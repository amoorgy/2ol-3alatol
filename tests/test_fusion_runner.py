"""build_dataset_b.py --plan mode: batches, checkpoint, quota pause, resume.

Every model is a fake (no network, no Ollama, no .env) and every path --
cache included -- is a temp file: the shared data/generated/fusion_cache.jsonl
is never opened by this suite.
"""

import collections
import contextlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.fusion import SCENARIOS
from franco_eval.fusion_jobs import (checkpoint_path, draw_v2_jobs, lock_path,
                                     run_lock, write_plan)
from franco_eval.model_clients import (FatalAPIError, QuotaExhausted,
                                       RequestRejected)

SCRIPT = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "build_dataset_b.py"
TEACHERS = ["llama3.1:8b", "gemma4:e2b", "deepseek:deepseek-v4-flash", "zai:glm-5.3-flash"]
FUSOR = "zai:glm-5.3"
WORDS = ["men", "meen", "3ayez"]
FILLER = ["ya", "sa7by", "embare7", "kan", "el", "yom", "gamed", "awy", "w",
          "ba2a", "keda", "fel", "share3", "3ashan", "bokra"]


def load_runner():
    spec = importlib.util.spec_from_file_location("build_dataset_b_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BDB = load_runner()


def anchor_in(prompt: str) -> str:
    return re.search(r'spelled exactly "([^"]+)"', prompt).group(1)


def good_sentence(model: str, prompt: str) -> str:
    """Passes every fusion gate: 12 tokens, anchor once, not first, ASCII."""
    word = anchor_in(prompt)
    seed = sum(map(ord, model + prompt))
    rest = [FILLER[(seed + i * 7) % len(FILLER)] for i in range(11)]
    return " ".join(rest[:3] + [word] + rest[3:])


def http_error(status: int) -> urllib.error.HTTPError:
    error = urllib.error.HTTPError("u", status, "m", {}, io.BytesIO(b""))
    error.close()
    return error


def quota(model: str) -> QuotaExhausted:
    return QuotaExhausted(http_error(429), provider="zai", model=model,
                          error_code="1308",
                          message="Usage limit reached for 5 hour. Your limit "
                                  "will reset at 2026-09-25 18:00:00")


class FakeModels:
    """call_model stand-in. behaviors[model](n, prompt) may raise, return a
    string, or return None for the default good sentence."""

    def __init__(self):
        self.calls = collections.Counter()
        self.behaviors = {}
        self._lock = threading.Lock()

    def __call__(self, model, prompt, *, timeout, cfg):
        with self._lock:
            self.calls[model] += 1
            n = self.calls[model]
        behavior = self.behaviors.get(model)
        if behavior is not None:
            result = behavior(n, prompt)
            if result is not None:
                return result
        return good_sentence(model, prompt)


class RunnerCase(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)
        self.plan = self.make_plan("plan.jsonl")
        self.out = self.dir / "out" / "fused_sentences.jsonl"
        self.audit = self.dir / "out" / "fused_audit.jsonl"
        self.cache = self.dir / "cache.jsonl"
        self.fake = FakeModels()

    def make_plan(self, name, seed=29, k=2):
        rows = [{"word": w, "arabic": "", "gloss_en": "", "context_type": "general",
                 "source": "bare", "band": "200-500"} for w in WORDS]
        jobs = draw_v2_jobs(rows, list(enumerate(SCENARIOS)), n_anchors=0, k=k,
                            seed=seed).jobs
        path = self.dir / name
        write_plan(path, jobs)
        self.plan_ids = [j.job_id for j in jobs] if name == "plan.jsonl" else getattr(
            self, "plan_ids", None)
        return path

    def argv(self, *extra, plan=None, teachers=TEACHERS, batch=4):
        return ["--plan", str(plan or self.plan), "--out", str(self.out),
                "--audit", str(self.audit), "--fusor", FUSOR,
                "--teachers", ",".join(teachers), "--workers", "1",
                "--batch-size", str(batch), *extra]

    def run_plan(self, *extra, **kwargs):
        args = BDB.build_parser().parse_args(self.argv(*extra, **kwargs))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = BDB.run_plan(args, call=self.fake, cache_path=self.cache)
        self.printed = printed.getvalue()
        return code

    def rows(self, path):
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def checkpoint(self):
        return json.loads(checkpoint_path(self.audit).read_text(encoding="utf-8"))


class FullRun(RunnerCase):
    def test_every_planned_job_is_audited_once(self):
        self.assertEqual(self.run_plan(), 0, self.printed)
        audit = self.rows(self.audit)
        self.assertEqual(sorted(r["job_id"] for r in audit), sorted(self.plan_ids))
        for row in audit:
            self.assertEqual(row["teachers"], TEACHERS)
            self.assertEqual(row["job_id"], f"{row['anchor']}#{row['scenario_id']}")
            self.assertEqual(row["scenario"], SCENARIOS[row["scenario_id"]])
        sentences = self.rows(self.out)
        ids = [s["sentence_id"] for s in sentences]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(sentences), 6)  # the fake always passes the gates
        self.assertTrue(all(re.fullmatch(r"fs_.+_\d\d", i) for i in ids))

    def test_checkpoint_records_the_run(self):
        self.run_plan()
        ckpt = self.checkpoint()
        self.assertEqual(ckpt["status"], "done")
        self.assertEqual(ckpt["exit_code"], 0)
        self.assertEqual((ckpt["jobs_done"], ckpt["jobs_total"], ckpt["jobs_remaining"]),
                         (6, 6, 0))
        self.assertEqual(ckpt["plan"]["sha256"],
                         (self.dir / "plan.jsonl.sha256").read_text().split()[0])
        self.assertEqual(ckpt["batches_this_invocation"], 2)
        self.assertEqual(ckpt["teachers"], TEACHERS)
        self.assertEqual(ckpt["fusor"], FUSOR)
        self.assertEqual(set(ckpt["latency"]), {*TEACHERS, FUSOR})
        self.assertEqual(ckpt["latency"][FUSOR]["calls"], 6)
        self.assertEqual(ckpt["totals"]["kept"], 6)
        self.assertIsNone(ckpt["last_error"])
        for key in ("started_at", "updated_at", "invocation_started_at", "command"):
            self.assertTrue(ckpt[key], key)

    def test_checkpoint_is_written_after_every_batch(self):
        seen = []
        real = BDB.write_checkpoint

        def spy(path, payload):
            seen.append((payload["status"], payload["jobs_done"], payload["exit_code"]))
            real(path, payload)

        with mock.patch.object(BDB, "write_checkpoint", spy):
            self.assertEqual(self.run_plan(batch=2), 0)
        self.assertEqual(seen, [("running", 0, None), ("running", 2, None),
                                ("running", 4, None), ("running", 6, None),
                                ("done", 6, 0)])


class QuotaPause(RunnerCase):
    def test_fusor_quota_pauses_after_the_current_job_and_resume_finishes(self):
        spent = {"yes": True}

        def fusor(n, prompt):
            if spent["yes"] and n > 3:
                raise quota(FUSOR)

        self.fake.behaviors[FUSOR] = fusor
        self.assertEqual(self.run_plan(batch=10), 75, self.printed)
        ckpt = self.checkpoint()
        self.assertEqual(ckpt["status"], f"paused_quota:{FUSOR}")
        self.assertEqual(ckpt["exit_code"], 75)
        self.assertIn("reset at 2026-09-25 18:00:00", ckpt["last_error"]["message"])
        self.assertEqual(ckpt["last_error"]["kind"], "quota")
        audit = self.rows(self.audit)
        self.assertEqual([r["job_id"] for r in audit], self.plan_ids[:3])
        self.assertEqual(ckpt["jobs_done"], 3)

        spent["yes"] = False  # the 5-hour window reset
        self.assertEqual(self.run_plan("--resume", batch=10), 0, self.printed)
        audit = self.rows(self.audit)
        self.assertEqual([r["job_id"] for r in audit], self.plan_ids)
        for teacher in TEACHERS:  # teacher drafts came from the cache
            self.assertEqual(self.fake.calls[teacher], 6, teacher)
        self.assertEqual(self.fake.calls[FUSOR], 4 + 3)
        ckpt = self.checkpoint()
        self.assertEqual((ckpt["status"], ckpt["invocations"]), ("done", 2))

    def test_teacher_quota_never_writes_a_degraded_job(self):
        spent = {"yes": True}
        teacher = "zai:glm-5.3-flash"

        def flash(n, prompt):
            if spent["yes"] and n > 2:
                raise quota(teacher)

        self.fake.behaviors[teacher] = flash
        self.assertEqual(self.run_plan(batch=10), 75)
        self.assertEqual(self.checkpoint()["status"], f"paused_quota:{teacher}")
        self.assertEqual(self.rows(self.audit), [])
        spent["yes"] = False
        self.assertEqual(self.run_plan("--resume", batch=10), 0)
        self.assertTrue(all(r["teachers"] == TEACHERS for r in self.rows(self.audit)))

    def test_quota_hit_during_assembly_is_a_pause_not_a_crash(self):
        """A draft that failed transiently in prefetch is re-asked live in
        assembly; if THAT call meets the quota, it is still a pause."""
        teacher = "zai:glm-5.3-flash"
        target = self.plan_ids[1]
        word, sid = target.split("#")
        attempts = collections.Counter()

        def flash(n, prompt):
            if anchor_in(prompt) == word and SCENARIOS[int(sid)] in prompt:
                attempts["target"] += 1
                if attempts["target"] <= 2:  # prefetch + fusor-phase re-draft
                    raise urllib.error.URLError("dropped")
                raise quota(teacher)  # the assembly re-draft

        self.fake.behaviors[teacher] = flash
        self.assertEqual(self.run_plan(batch=10), 75, self.printed)
        ckpt = self.checkpoint()
        self.assertEqual(ckpt["status"], f"paused_quota:{teacher}")
        self.assertEqual(attempts["target"], 3)
        self.assertEqual([r["job_id"] for r in self.rows(self.audit)], self.plan_ids[:1])

    def test_repeated_pauses_never_duplicate_or_degrade(self):
        self.fake.behaviors[FUSOR] = (
            lambda n, prompt: (_ for _ in ()).throw(quota(FUSOR)) if n % 3 == 0 else None)
        codes = []
        for _ in range(12):
            codes.append(self.run_plan("--resume", batch=2))
            if codes[-1] != 75:
                break
        self.assertEqual(codes[-1], 0, codes)
        self.assertIn(75, codes)
        audit = self.rows(self.audit)
        self.assertEqual(sorted(r["job_id"] for r in audit), sorted(self.plan_ids))
        self.assertTrue(all(r["teachers"] == TEACHERS for r in audit))
        ids = [s["sentence_id"] for s in self.rows(self.out)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_a_truncated_tail_is_repaired_on_resume(self):
        spent = {"yes": True}
        self.fake.behaviors[FUSOR] = (
            lambda n, prompt: (_ for _ in ()).throw(quota(FUSOR))
            if spent["yes"] and n > 3 else None)
        self.assertEqual(self.run_plan(batch=10), 75)
        with self.audit.open("a", encoding="utf-8") as handle:
            handle.write('{"anchor": "men", "scena')  # killed mid-write
        with self.out.open("a", encoding="utf-8") as handle:
            handle.write('{"sentence_id": "fs_')
        spent["yes"] = False
        self.assertEqual(self.run_plan("--resume", batch=10), 0, self.printed)
        self.assertEqual(sorted(r["job_id"] for r in self.rows(self.audit)),
                         sorted(self.plan_ids))
        self.assertEqual(len(self.rows(self.out)), 6)


class StopsThatAreNotPauses(RunnerCase):
    def test_fatal_error_stops_with_exit_1_and_writes_nothing_degraded(self):
        teacher = "deepseek:deepseek-v4-flash"

        def revoked(n, prompt):
            raise FatalAPIError(http_error(401), provider="deepseek", model=teacher,
                                error_code=None, message="Authentication Fails")

        self.fake.behaviors[teacher] = revoked
        self.assertEqual(self.run_plan(), 1)
        self.assertEqual(self.checkpoint()["status"], f"failed_fatal:{teacher}")
        self.assertEqual(self.rows(self.audit), [])

    def test_a_dead_model_trips_the_breaker(self):
        def down(n, prompt):
            raise urllib.error.URLError("connection refused")

        self.fake.behaviors["llama3.1:8b"] = down
        self.assertEqual(self.run_plan(batch=10), 76)   # pause, not stop
        self.assertEqual(self.checkpoint()["status"], "paused_model_down:llama3.1:8b")
        self.assertEqual(self.fake.calls["llama3.1:8b"], 5)
        self.assertEqual(self.rows(self.audit), [])

    def test_a_sporadic_failure_keeps_the_job_with_fewer_drafts(self):
        """Legacy semantics: one teacher down must not kill the anchor."""
        def flaky(n, prompt):
            if anchor_in(prompt) == "meen":
                raise urllib.error.URLError("dropped")

        self.fake.behaviors["gemma4:e2b"] = flaky
        self.assertEqual(self.run_plan(), 0, self.printed)
        for row in self.rows(self.audit):
            expected = [t for t in TEACHERS if t != "gemma4:e2b"] \
                if row["anchor"] == "meen" else TEACHERS
            self.assertEqual(row["teachers"], expected)
        failures = self.checkpoint()["model_failures"]["gemma4:e2b"]
        self.assertGreaterEqual(failures["transient"], 2)

    def test_a_content_filter_never_stops_the_run(self):
        teacher = "zai:glm-5.3-flash"

        def filtered(n, prompt):
            raise RequestRejected(http_error(400), provider="zai", model=teacher,
                                  error_code="1301", message="unsafe content")

        self.fake.behaviors[teacher] = filtered
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.rows(self.audit)), 6)

    def test_failed_jobs_leave_the_run_incomplete_not_done(self):
        target = self.plan_ids[0]
        word, sid = target.split("#")

        def empty_for_target(n, prompt):
            if anchor_in(prompt) == word and SCENARIOS[int(sid)] in prompt:
                return ""

        self.fake.behaviors[FUSOR] = empty_for_target
        self.assertEqual(self.run_plan(), 3, self.printed)
        ckpt = self.checkpoint()
        self.assertEqual(ckpt["status"], "incomplete")
        self.assertEqual(ckpt["jobs_remaining"], 1)
        self.assertEqual(ckpt["failed_jobs"], {target: "fusor_error"})
        self.assertNotIn(target, {r["job_id"] for r in self.rows(self.audit)})
        self.fake.behaviors.clear()
        self.assertEqual(self.run_plan("--resume"), 0)  # retried on resume
        self.assertEqual(len(self.rows(self.audit)), 6)


class Refusals(RunnerCase):
    def paused_run(self):
        self.fake.behaviors[FUSOR] = (
            lambda n, prompt: (_ for _ in ()).throw(quota(FUSOR)) if n > 3 else None)
        self.assertEqual(self.run_plan(batch=10), 75)
        self.fake.behaviors.clear()
        return self.audit.read_bytes()

    def test_resume_refuses_a_different_plan(self):
        before = self.paused_run()
        other = self.make_plan("other.jsonl", seed=30)
        self.assertEqual(self.run_plan("--resume", plan=other), 2)
        self.assertIn("plan sha256", self.printed)
        self.assertEqual(self.audit.read_bytes(), before)

    def test_an_edited_plan_is_refused(self):
        before = self.paused_run()
        self.plan.write_text(self.plan.read_text().replace("men", "mEn", 1))
        self.assertEqual(self.run_plan("--resume"), 2)
        self.assertEqual(self.audit.read_bytes(), before)

    def test_resume_refuses_a_changed_teacher_pool(self):
        self.paused_run()
        self.assertEqual(self.run_plan("--resume", teachers=TEACHERS[:3]), 2)
        self.assertIn("--teachers", self.printed)

    def test_existing_run_needs_resume(self):
        self.paused_run()
        self.assertEqual(self.run_plan(), 2)

    def test_foreign_outputs_are_refused_even_with_resume(self):
        """No checkpoint beside the audit: these are not this plan's files."""
        self.out.parent.mkdir(parents=True)
        self.out.write_text('{"sentence_id": "fs_x_00"}\n')
        self.assertEqual(self.run_plan("--resume"), 2)
        self.assertEqual(self.out.read_text(), '{"sentence_id": "fs_x_00"}\n')

    def test_audit_rows_outside_the_plan_are_refused(self):
        self.paused_run()
        with self.audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"anchor": "zzz", "scenario_index": 1}) + "\n")
        self.assertEqual(self.run_plan("--resume"), 2)

    def test_a_second_runner_is_locked_out(self):
        with run_lock(lock_path(self.audit)):
            self.assertEqual(self.run_plan(), 2)
        self.assertFalse(self.audit.exists())

    def test_plan_mode_rejects_the_pilot_flags(self):
        for extra in (["--split", "x.tsv"], ["--smoke", "3"],
                      ["--scenarios-per-anchor", "4"]):
            with self.subTest(extra):
                self.assertEqual(self.run_plan(*extra), 2)

    def test_resume_of_a_finished_run_is_a_no_op(self):
        self.assertEqual(self.run_plan(), 0)
        before = self.audit.read_bytes()
        calls = sum(self.fake.calls.values())
        self.assertEqual(self.run_plan("--resume"), 0)
        self.assertEqual(self.audit.read_bytes(), before)
        self.assertEqual(sum(self.fake.calls.values()), calls)


class LegacyPath(unittest.TestCase):
    """Without --plan the pilot/--split path writes the same rows as before."""

    LEGACY_AUDIT_KEYS = ["anchor", "arabic", "context_type", "scenario",
                         "scenario_index", "fusor", "teachers", "drafts", "fused",
                         "gates_ok", "gate_reasons", "contribution", "bon_pick",
                         "bon_teacher"]

    def test_pilot_rows_keep_their_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            traps = tmp / "trap_votes.tsv"
            traps.write_text(
                "word_type\ttrap\tvalid\tarabic_word\tarabic_trap\t"
                "meaning_en_word\tmeaning_en_trap\n"
                "men\tmeen\ttrue\tمن\tمين\tfrom\twho\n", encoding="utf-8")
            fake = FakeModels()
            argv = ["--smoke", "2", "--fusor", FUSOR, "--teachers", ",".join(TEACHERS),
                    "--out", str(tmp / "o.jsonl"), "--audit", str(tmp / "a.jsonl")]
            with mock.patch.object(BDB, "TRAPS", traps), \
                    mock.patch.object(BDB, "CACHE", tmp / "cache.jsonl"), \
                    mock.patch.object(BDB, "call_model", fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(BDB.main(argv), 0)
            audit = [json.loads(l) for l in (tmp / "a.jsonl").read_text().splitlines()]
            self.assertEqual([list(r) for r in audit], [self.LEGACY_AUDIT_KEYS] * 2)
            sentences = [json.loads(l) for l in (tmp / "o.jsonl").read_text().splitlines()]
            self.assertEqual([s["sentence_id"][:6] for s in sentences], ["fs_men", "fs_mee"])
            self.assertFalse(checkpoint_path(tmp / "a.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
