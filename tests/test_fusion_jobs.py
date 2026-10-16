import hashlib
import io
import json
import pathlib
import random
import sys
import tempfile
import threading
import unittest
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.fusion import SCENARIOS, Anchor
from franco_eval.fusion_jobs import (EmptyReply, LockHeld, PlanError,
                                     ResumeRefused, RunGuard, RunStopped,
                                     StaleOutputError, check_resume,
                                     checkpoint_path, completed_job_ids,
                                     completed_keys, draw_v2_jobs,
                                     ensure_fresh, failure_kind, job_id,
                                     load_plan, plan_jobs, read_anchor_bank,
                                     read_checkpoint, repair_tail, run_lock,
                                     write_checkpoint, write_plan)
from franco_eval.model_clients import (FatalAPIError, QuotaExhausted,
                                       RequestRejected, TransientHTTPError)

ANCHORS = [Anchor("men"), Anchor("meen"), Anchor("3ayez")]

BANK_HEADER = "word\tarabic\tgloss_en\tcontext_type\tsource\tcount\tdoc_freq\tband\tpriority"
PINNED = list(enumerate(SCENARIOS))
EXTRA = [(16, "you're cooking lunch for the whole family today"),
         (17, "you're asking to borrow money until the end of the month"),
         (19, "you're telling a friend you can't come out tonight")]


def bank_row(word, arabic="", gloss="", context="general", source="bare",
             priority=1) -> str:
    return "\t".join([word, arabic, gloss, context, source, "10", "5", "200-500",
                      str(priority)])


def write_bank(path: pathlib.Path, rows: list[str]) -> pathlib.Path:
    path.write_text("\n".join([BANK_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def bank_rows(words):
    return [{"word": w, "arabic": "", "gloss_en": "", "context_type": "general",
             "source": "bare", "band": "200-500"} for w in words]


def http_error(status: int) -> urllib.error.HTTPError:
    error = urllib.error.HTTPError("u", status, "m", {}, io.BytesIO(b""))
    error.close()
    return error


def classified(cls, status=429):
    return cls(http_error(status), provider="p", model="m", error_code=None,
               message="x")


class PlanJobs(unittest.TestCase):
    def test_one_scenario_reproduces_the_pilot_draw(self):
        """K=1 must consume the RNG exactly like the pilot's per-anchor
        randrange, so pilot anchors keep their scenarios."""
        rng = random.Random(29)
        expected = [(a, rng.randrange(len(SCENARIOS))) for a in ANCHORS]
        self.assertEqual(plan_jobs(ANCHORS, 1, seed=29), expected)

    def test_k_scenarios_are_distinct_per_anchor(self):
        jobs = plan_jobs(ANCHORS, 4, seed=29)
        self.assertEqual(len(jobs), 12)
        for anchor in ANCHORS:
            picked = [s for a, s in jobs if a == anchor]
            self.assertEqual(len(picked), len(set(picked)), anchor.word)

    def test_deterministic(self):
        self.assertEqual(plan_jobs(ANCHORS, 3, seed=29),
                         plan_jobs(ANCHORS, 3, seed=29))

    def test_k_cannot_exceed_the_scenario_list(self):
        with self.assertRaises(ValueError):
            plan_jobs(ANCHORS, len(SCENARIOS) + 1, seed=29)


class OutputSafety(unittest.TestCase):
    """The b2 pilot appended a 20-anchor run onto 505 stale rows."""

    def test_non_empty_existing_output_is_refused_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "out.jsonl"
            path.write_text('{"x": 1}\n')
            with self.assertRaises(StaleOutputError):
                ensure_fresh([path], resume=False)
            ensure_fresh([path], resume=True)  # explicit resume is allowed

    def test_missing_or_empty_output_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = pathlib.Path(tmp) / "empty.jsonl"
            empty.write_text("")
            ensure_fresh([empty, pathlib.Path(tmp) / "absent.jsonl"],
                         resume=False)

    def test_completed_keys_read_from_the_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = pathlib.Path(tmp) / "audit.jsonl"
            audit.write_text(
                json.dumps({"anchor": "men", "scenario_index": 3}) + "\n"
                + json.dumps({"anchor": "meen", "scenario_index": 0}) + "\n")
            self.assertEqual(completed_keys(audit), {("men", 3), ("meen", 0)})
            self.assertEqual(completed_keys(pathlib.Path(tmp) / "none"), set())


class TmpCase(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)


class AnchorBank(TmpCase):
    def test_reads_rows_in_bank_order(self):
        path = write_bank(self.dir / "bank.tsv", [
            bank_row("2a2ema", "قائمة", "a list", "trap_contrast", "trap", 1),
            bank_row("def", priority=2)])
        rows = read_anchor_bank(path)
        self.assertEqual([r["word"] for r in rows], ["2a2ema", "def"])
        self.assertEqual(rows[0]["arabic"], "قائمة")
        self.assertEqual(rows[1]["arabic"], "")

    def test_wrong_header_or_bad_rows_fail_loudly(self):
        cases = {
            "header": "word\tarabic\n2a2ema\tx\n",
            "short row": BANK_HEADER + "\n2a2ema\tx\n",
            "empty word": BANK_HEADER + "\n" + bank_row("") + "\n",
            "space in word": BANK_HEADER + "\n" + bank_row("a b") + "\n",
        }
        for name, text in cases.items():
            with self.subTest(name):
                path = self.dir / "bad.tsv"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(PlanError):
                    read_anchor_bank(path)


class DrawV2(unittest.TestCase):
    SCEN = PINNED + EXTRA

    def draw(self, words, k=4, n=0, seed=29, exclude=frozenset(), rows=None):
        return draw_v2_jobs(rows or bank_rows(words), self.SCEN, n_anchors=n,
                            k=k, seed=seed, exclude=exclude)

    def test_job_ids_are_unique_and_k_distinct_accepted_scenarios_each(self):
        plan = self.draw(["men", "meen", "3ayez"], k=5)
        ids = [j.job_id for j in plan.jobs]
        self.assertEqual(len(ids), 15)
        self.assertEqual(len(set(ids)), 15)
        accepted = {sid for sid, _t in self.SCEN}
        for job in plan.jobs:
            self.assertEqual(job.job_id, f"{job.anchor.word}#{job.scenario_id}")
            self.assertIn(job.scenario_id, accepted)
            self.assertEqual(job.scenario, dict(self.SCEN)[job.scenario_id])

    def test_anchor_major_in_bank_order(self):
        plan = self.draw(["men", "meen", "3ayez"], k=2)
        self.assertEqual([j.anchor.word for j in plan.jobs],
                         ["men", "men", "meen", "meen", "3ayez", "3ayez"])

    def test_deterministic_and_seed_sensitive(self):
        a = self.draw(["men", "meen", "3ayez", "wa7ed"], k=6)
        self.assertEqual(a.jobs, self.draw(["men", "meen", "3ayez", "wa7ed"], k=6).jobs)
        self.assertNotEqual(a.jobs, self.draw(["men", "meen", "3ayez", "wa7ed"],
                                              k=6, seed=30).jobs)

    def test_an_anchors_draw_does_not_depend_on_other_anchors(self):
        """Per-anchor RNG: a bigger --anchors or an inserted row never
        reshuffles the scenarios of anchors already planned."""
        small = self.draw(["men", "meen"], k=4)
        big = self.draw(["x1", "men", "x2", "meen", "x3"], k=4)
        for word in ("men", "meen"):
            self.assertEqual([j.scenario_id for j in small.jobs if j.anchor.word == word],
                             [j.scenario_id for j in big.jobs if j.anchor.word == word])

    def test_first_n_unique_anchors(self):
        plan = self.draw(["men", "meen", "3ayez"], k=1, n=2)
        self.assertEqual([j.anchor.word for j in plan.jobs], ["men", "meen"])
        self.assertEqual(plan.stats["anchors_used"], 2)

    def test_duplicate_words_are_planned_once_first_row_wins(self):
        rows = bank_rows(["men", "meen"])
        rows.insert(1, {**rows[0], "arabic": "مين"})  # men again, other Arabic
        plan = self.draw(None, k=2, rows=rows)
        self.assertEqual([j.anchor.word for j in plan.jobs], ["men", "men", "meen", "meen"])
        self.assertEqual({j.anchor.arabic for j in plan.jobs}, {""})
        self.assertEqual(plan.stats["duplicate_rows_skipped"], 1)
        self.assertEqual(plan.stats["conflicting_duplicates"], ["men"])
        ids = [j.job_id for j in plan.jobs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_excluded_jobs_are_skipped_not_redrawn(self):
        full = self.draw(["men", "meen"], k=4)
        skip = {full.jobs[0].job_id, full.jobs[5].job_id}
        plan = self.draw(["men", "meen"], k=4, exclude=frozenset(skip))
        self.assertEqual([j for j in full.jobs if j.job_id not in skip], list(plan.jobs))
        self.assertEqual(plan.stats["jobs_excluded"], 2)
        self.assertEqual(plan.stats["jobs_drawn"], 8)

    def test_anchor_fields_travel_with_the_job(self):
        rows = [{"word": "2a2ema", "arabic": "قائمة", "gloss_en": "a list",
                 "context_type": "trap_contrast", "source": "trap", "band": "5k+"}]
        job = self.draw(None, k=1, rows=rows).jobs[0]
        self.assertEqual(job.anchor, Anchor("2a2ema", "قائمة", "a list", "trap_contrast"))
        self.assertEqual((job.source, job.band), ("trap", "5k+"))

    def test_k_bounds(self):
        with self.assertRaises(PlanError):
            self.draw(["men"], k=len(self.SCEN) + 1)
        with self.assertRaises(PlanError):
            self.draw(["men"], k=0)

    def test_pinned_ids_must_be_the_fusion_scenarios_verbatim(self):
        broken = [(0, SCENARIOS[0] + " today"), *PINNED[1:], *EXTRA]
        with self.assertRaises(PlanError):
            draw_v2_jobs(bank_rows(["men"]), broken, n_anchors=0, k=1, seed=29)


class PlanFile(TmpCase):
    def plan(self, words=("men", "meen"), k=3):
        return draw_v2_jobs(bank_rows(list(words)), PINNED + EXTRA, n_anchors=0,
                            k=k, seed=29).jobs

    def test_round_trip_and_sha(self):
        path = self.dir / "plan.jsonl"
        jobs = self.plan()
        sha = write_plan(path, jobs)
        self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual((self.dir / "plan.jsonl.sha256").read_text(),
                         f"{sha}  plan.jsonl\n")
        loaded = load_plan(path)
        self.assertEqual(loaded.sha256, sha)
        self.assertEqual(loaded.jobs, jobs)

    def test_same_inputs_same_bytes(self):
        write_plan(self.dir / "a.jsonl", self.plan())
        write_plan(self.dir / "b.jsonl", self.plan())
        self.assertEqual((self.dir / "a.jsonl").read_bytes(),
                         (self.dir / "b.jsonl").read_bytes())

    def test_frozen_plan_is_never_overwritten(self):
        path = self.dir / "plan.jsonl"
        write_plan(path, self.plan())
        with self.assertRaises(FileExistsError):
            write_plan(path, self.plan(k=2))

    def test_an_edited_plan_is_refused(self):
        path = self.dir / "plan.jsonl"
        write_plan(path, self.plan())
        path.write_text(path.read_text().replace('"men"', '"mEn"', 1))
        with self.assertRaises(PlanError):
            load_plan(path)

    def test_duplicate_or_malformed_job_ids_are_refused(self):
        path = self.dir / "plan.jsonl"
        write_plan(path, self.plan())
        first = path.read_text().splitlines()[0]
        for name, text in {"duplicate": first + "\n" + first + "\n",
                           "id mismatch": first.replace('"job_id": "men#', '"job_id": "meen#') + "\n"}.items():
            with self.subTest(name):
                bad = self.dir / f"{name}.jsonl"
                bad.write_text(text)
                with self.assertRaises(PlanError):
                    load_plan(bad)


class CompletedJobIds(TmpCase):
    def test_job_id_or_legacy_anchor_and_index(self):
        audit = self.dir / "audit.jsonl"
        audit.write_text(
            json.dumps({"anchor": "men", "scenario_index": 3}) + "\n"
            + json.dumps({"anchor": "meen", "scenario_index": 40,
                          "job_id": "meen#40"}) + "\n\n")
        self.assertEqual(completed_job_ids(audit), {"men#3", "meen#40"})
        self.assertEqual(completed_job_ids(self.dir / "absent.jsonl"), set())
        self.assertEqual(job_id("men", 3), "men#3")


class Checkpoint(TmpCase):
    def test_lives_beside_the_audit(self):
        self.assertEqual(checkpoint_path(self.dir / "fused_audit.jsonl"),
                         self.dir / "fused_audit.checkpoint.json")

    def test_atomic_round_trip(self):
        path = self.dir / "run.checkpoint.json"
        self.assertIsNone(read_checkpoint(path))
        write_checkpoint(path, {"status": "running", "jobs_done": 3})
        write_checkpoint(path, {"status": "done", "jobs_done": 4, "text": "قائمة"})
        self.assertEqual(read_checkpoint(path),
                         {"status": "done", "jobs_done": 4, "text": "قائمة"})
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["run.checkpoint.json"])


class RepairTail(TmpCase):
    def test_partial_last_line_is_cut(self):
        path = self.dir / "audit.jsonl"
        path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": ')
        self.assertEqual(repair_tail(path), 6)
        self.assertEqual(path.read_bytes(), b'{"a": 1}\n{"a": 2}\n')

    def test_complete_or_missing_files_are_untouched(self):
        path = self.dir / "audit.jsonl"
        path.write_bytes(b'{"a": 1}\n')
        self.assertEqual(repair_tail(path), 0)
        self.assertEqual(path.read_bytes(), b'{"a": 1}\n')
        self.assertEqual(repair_tail(self.dir / "absent.jsonl"), 0)
        single = self.dir / "one.jsonl"
        single.write_bytes(b'{"a"')
        self.assertEqual(repair_tail(single), 4)
        self.assertEqual(single.read_bytes(), b"")


class Lock(TmpCase):
    def test_second_holder_is_refused_until_release(self):
        path = self.dir / "audit.lock"
        with run_lock(path):
            with self.assertRaises(LockHeld):
                with run_lock(path):
                    pass
        with run_lock(path):  # released
            pass


class Resume(TmpCase):
    CKPT = {"plan": {"sha256": "abc"}, "out": "/x/out.jsonl",
            "teachers": ["a", "b"], "fusor": "f", "jobs_done": 3}

    def check(self, **overrides):
        kwargs = {"plan_sha": "abc", "out": pathlib.Path("/x/out.jsonl"),
                  "teachers": ["a", "b"], "fusor": "f", "done": 3}
        kwargs.update(overrides)
        check_resume(self.CKPT, **kwargs)

    def test_matching_run_resumes(self):
        self.check()
        self.check(done=5)  # the audit may be ahead of the last checkpoint
        self.check(done=None)  # before the audit is read (tail not yet repaired)

    def test_mismatches_are_refused(self):
        for name, override in {"plan": {"plan_sha": "abd"},
                               "out": {"out": pathlib.Path("/x/other.jsonl")},
                               "teachers": {"teachers": ["a"]},
                               "fusor": {"fusor": "g"},
                               "audit lost rows": {"done": 2}}.items():
            with self.subTest(name):
                with self.assertRaises(ResumeRefused):
                    self.check(**override)


class FailureKinds(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(failure_kind(classified(QuotaExhausted)), "quota")
        self.assertEqual(failure_kind(classified(FatalAPIError, 401)), "fatal")
        self.assertEqual(failure_kind(classified(RequestRejected, 400)), "rejected")
        self.assertEqual(failure_kind(classified(TransientHTTPError, 503)), "transient")
        self.assertEqual(failure_kind(urllib.error.URLError("refused")), "transient")
        self.assertEqual(failure_kind(EmptyReply("m")), "empty")
        self.assertEqual(failure_kind(KeyError("choices")), "error")


class Guard(unittest.TestCase):
    def test_quota_pauses_and_first_reason_wins(self):
        guard = RunGuard()
        guard.check()
        guard.failure("zai:glm-5.3", classified(QuotaExhausted), key="p1")
        guard.failure("deepseek:x", classified(FatalAPIError, 401), key="p2")
        self.assertEqual(guard.stop_status, "paused_quota:zai:glm-5.3")
        with self.assertRaises(RunStopped):
            guard.check()
        self.assertEqual(guard.last_error["model"], "deepseek:x")
        self.assertEqual(guard.failures["zai:glm-5.3"]["quota"], 1)

    def test_fatal_stops(self):
        guard = RunGuard()
        guard.failure("deepseek:x", classified(FatalAPIError, 401), key="p")
        self.assertEqual(guard.stop_status, "failed_fatal:deepseek:x")

    def test_consecutive_distinct_transient_failures_trip_the_breaker(self):
        guard = RunGuard(max_streak=3)
        for n in range(2):
            guard.failure("llama3.1:8b", TimeoutError(), key=f"p{n}")
        guard.success("llama3.1:8b")  # a success resets the streak
        for n in range(2, 4):
            guard.failure("llama3.1:8b", TimeoutError(), key=f"p{n}")
        self.assertIsNone(guard.stop_status)
        guard.failure("llama3.1:8b", TimeoutError(), key="p4")
        self.assertEqual(guard.stop_status, "paused_model_down:llama3.1:8b")

    def test_a_model_outage_pauses_with_its_own_exit_code(self):
        """User ruling 2026-09-26: an outage pauses and resumes (DeepSeek went
        empty for minutes, twice), distinct from a quota pause (75)."""
        from franco_eval.fusion_jobs import EXIT_MODEL_DOWN, EXIT_PAUSED, status_exit_code
        self.assertEqual(status_exit_code("paused_model_down:deepseek:x"), EXIT_MODEL_DOWN)
        self.assertEqual(EXIT_MODEL_DOWN, 76)
        self.assertNotEqual(EXIT_MODEL_DOWN, EXIT_PAUSED)

    def test_retrying_the_same_prompt_does_not_count_twice(self):
        guard = RunGuard(max_streak=2)
        for _ in range(5):
            guard.failure("m", TimeoutError(), key="same")
        self.assertIsNone(guard.stop_status)

    def test_rejections_never_trip_the_breaker(self):
        """A content filter on one anchor rejects all its scenarios in a row."""
        guard = RunGuard(max_streak=2)
        for n in range(10):
            guard.failure("zai:glm-5.3-flash", classified(RequestRejected, 400), key=f"p{n}")
        self.assertIsNone(guard.stop_status)
        self.assertEqual(guard.failures["zai:glm-5.3-flash"]["rejected"], 10)

    def test_thread_safe_counts(self):
        guard = RunGuard(max_streak=10 ** 6)

        def hammer(offset):
            for n in range(500):
                guard.failure("m", TimeoutError(), key=f"{offset}-{n}")

        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(guard.failures["m"]["transient"], 2000)


if __name__ == "__main__":
    unittest.main()
