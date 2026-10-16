"""Council verification tooling: jobs listing, calibration, audit sampling.

Fixtures only -- no model, no workflow, no live council directory is read.
"""

import contextlib
import hashlib
import io
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COUNCIL = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "council"
sys.path.insert(0, str(COUNCIL))

import compare_councils  # noqa: E402
import council_io  # noqa: E402
import council_jobs  # noqa: E402
import sample_audit  # noqa: E402

FUSED = council_io.COUNCILS["fused"]
PAIRS = council_io.COUNCILS["pairs"]


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def fused_item(sid: str, anchor: str = "x") -> dict:
    return {"sid": sid, "rid": sid, "anchor": anchor, "arabic": "ا",
            "gloss": "g", "sentence": f"sentence {sid}"}


def write_panel(out_dir: pathlib.Path, chunks: dict[str, list[dict]],
                votes: dict[str, str]) -> None:
    """votes: lens -> 'TFT..' over all items in chunk order."""
    order = [i for c in chunks for i in chunks[c]]
    for lens, pattern in votes.items():
        verdict = dict(zip((i["sid"] for i in order), pattern))
        for chunk, items in chunks.items():
            write_json(out_dir / f"{lens}_{chunk}.json", [
                {"sid": i["sid"], "rid": i["rid"], "v": verdict[i["sid"]] == "T",
                 "why": "" if verdict[i["sid"]] == "T" else f"{lens} says no"}
                for i in items])


def run_main(main, argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class Statistics(unittest.TestCase):
    def test_cohen_kappa_textbook_example(self):
        # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
        a = [True] * 25 + [False] * 25
        b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
        self.assertAlmostEqual(council_io.cohen_kappa(a, b), 0.4, places=4)

    def test_cohen_kappa_is_undefined_without_variance(self):
        self.assertIsNone(council_io.cohen_kappa([True] * 4, [True] * 4))
        self.assertIsNone(council_io.cohen_kappa([], []))

    def test_cohen_kappa_rejects_unequal_lengths(self):
        with self.assertRaises(ValueError):
            council_io.cohen_kappa([True], [True, False])

    def test_wilson_interval_known_values(self):
        lo, hi = council_io.wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)
        lo, hi = council_io.wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)

    def test_wilson_endpoints_are_exact_at_zero_and_all_successes(self):
        # center - half and center + half are mathematically 0 and 1 here, but
        # float rounding leaves 2.8e-17 / 0.9999999999999999 for many n (which
        # n depends on the platform's libm), so results differ across machines.
        for n in (4, 10, 15, 21, 29, 33, 40, 1000):
            with self.subTest(n=n):
                self.assertEqual(council_io.wilson_interval(0, n)[0], 0.0)
                self.assertEqual(council_io.wilson_interval(n, n)[1], 1.0)

    def test_wilson_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            council_io.wilson_interval(3, 0)
        with self.assertRaises(ValueError):
            council_io.wilson_interval(11, 10)

    def test_stratified_rate_weights_strata_and_uses_kish_n_eff(self):
        est = council_io.stratified_rate([(900, 30, 3), (100, 30, 15)])
        self.assertAlmostEqual(est["rate"], 0.14, places=10)
        self.assertAlmostEqual(est["n_eff"], 30 / 0.82, places=6)
        lo, hi = council_io.wilson_interval(0.14 * 30 / 0.82, 30 / 0.82)
        self.assertAlmostEqual(est["wilson95"][0], lo, places=10)
        self.assertAlmostEqual(est["wilson95"][1], hi, places=10)

    def test_proportional_allocation_gives_pooled_n(self):
        est = council_io.stratified_rate([(200, 20, 2), (100, 10, 1)])
        self.assertAlmostEqual(est["n_eff"], 30.0, places=10)
        self.assertAlmostEqual(est["rate"], 0.1, places=10)

    def test_every_stratum_must_be_sampled(self):
        with self.assertRaises(ValueError):
            council_io.stratified_rate([(100, 0, 0), (50, 5, 1)])


class ChunkAndVerdictFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.chunks = self.tmp / "chunks"
        write_json(self.chunks / "c10.json", [fused_item("s9")])
        write_json(self.chunks / "c2.json", [fused_item("s1"), fused_item("s2"),
                                             fused_item("s3")])
        write_json(self.chunks / "sample.json", {"not": "a chunk"})

    def test_chunks_load_in_numeric_order_and_ignore_other_files(self):
        chunks = council_io.load_chunks(self.chunks)
        self.assertEqual(list(chunks), ["c2", "c10"])

    def test_an_item_in_two_chunks_is_refused(self):
        write_json(self.chunks / "c3.json", [fused_item("s1")])
        with self.assertRaises(council_io.CouncilDataError):
            council_io.load_chunks(self.chunks)

    def test_an_empty_chunk_dir_is_refused(self):
        with self.assertRaises(council_io.CouncilDataError):
            council_io.load_chunks(self.tmp / "nowhere")

    def status(self, content) -> council_io.JobStatus:
        path = self.tmp / "votes" / "anchor_c2.json"
        if content is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if isinstance(content, str)
                            else json.dumps(content), encoding="utf-8")
        items = council_io.load_chunks(self.chunks)["c2"]
        return council_io.job_status(path, items)

    def rows(self, *spec):
        return [{"sid": s, "rid": s, "v": v, "why": ""} for s, v in spec]

    def test_missing_file_is_pending(self):
        st = self.status(None)
        self.assertFalse(st.complete)
        self.assertTrue(st.reason.startswith("missing"))

    def test_unparsable_file_is_pending(self):
        st = self.status('[{"sid": "s1", ')
        self.assertFalse(st.complete)
        self.assertTrue(st.reason.startswith("unparsable"))

    def test_short_file_is_pending(self):
        st = self.status(self.rows(("s1", True), ("s2", False)))
        self.assertFalse(st.complete)
        self.assertTrue(st.reason.startswith("short: 2/3"))

    def test_right_count_but_wrong_items_is_pending(self):
        st = self.status(self.rows(("s1", True), ("s2", True), ("s7", True)))
        self.assertFalse(st.complete)
        self.assertTrue(st.reason.startswith("mismatch"))

    def test_non_boolean_verdict_is_pending(self):
        # "false" is truthy: merged as-is it would count as an ACCEPT
        st = self.status(self.rows(("s1", True), ("s2", "false"), ("s3", True)))
        self.assertFalse(st.complete)
        self.assertIn("non-boolean", st.reason)

    def test_conflicting_duplicates_are_pending(self):
        st = self.status(self.rows(("s1", True), ("s1", False), ("s2", True),
                                   ("s3", True)))
        self.assertFalse(st.complete)
        self.assertIn("conflicting", st.reason)

    def test_complete_file_with_stray_rows_is_complete_but_counted(self):
        st = self.status(self.rows(("s1", True), ("s2", True), ("s3", False),
                                   ("zz", True)))
        self.assertTrue(st.complete)
        self.assertEqual(st.strays, 1)


class CouncilJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.chunks = self.tmp / "chunks"
        self.votes = self.tmp / "votes"
        self.c = {"c0": [fused_item("s1"), fused_item("s2")],
                  "c1": [fused_item("s3")]}
        for name, items in self.c.items():
            write_json(self.chunks / f"{name}.json", items)

    def args_for(self, *extra) -> tuple[int, dict | None, str]:
        code, out, err = run_main(council_jobs.main, [
            "--council", "fused", "--chunk-dir", str(self.chunks),
            "--out-dir", str(self.votes), *extra])
        return code, (json.loads(out) if code == 0 else None), err

    def test_fresh_council_lists_every_chunk_lens_job(self):
        code, args, _ = self.args_for("--model", "sonnet", "--effort", "high")
        self.assertEqual(code, 0)
        self.assertEqual(len(args["jobs"]), 2 * len(FUSED))
        self.assertEqual(args["jobs"][0], {"chunk": "c0", "lens": FUSED[0]})
        self.assertEqual(args["sizes"], {"c0": 2, "c1": 1})
        self.assertEqual(args["council"], "fused")
        self.assertEqual((args["model"], args["effort"]), ("sonnet", "high"))
        self.assertTrue(pathlib.Path(args["chunkDir"]).is_absolute())
        self.assertEqual(pathlib.Path(args["outDir"]), self.votes.resolve())

    def test_resume_lists_only_incomplete_jobs(self):
        write_panel(self.votes, self.c, {lens: "TTT" for lens in FUSED})
        write_json(self.votes / "anchor_c0.json", [
            {"sid": "s1", "rid": "s1", "v": True, "why": ""}])      # short
        (self.votes / "fluency_c1.json").write_text("[{", encoding="utf-8")
        (self.votes / "register_c0.json").unlink()
        _, args, err = self.args_for("--model", "sonnet", "--effort", "high")
        self.assertEqual(args["jobs"], [{"chunk": "c0", "lens": "anchor"},
                                        {"chunk": "c0", "lens": "register"},
                                        {"chunk": "c1", "lens": "fluency"}])
        self.assertIn("short", err)
        self.assertIn("unparsable", err)

    def test_lens_and_chunk_subsets(self):
        _, args, _ = self.args_for("--lenses", "anchor,register", "--chunks", "c1",
                                   "--model", "opus", "--effort", "xhigh")
        self.assertEqual(args["jobs"], [{"chunk": "c1", "lens": "anchor"},
                                        {"chunk": "c1", "lens": "register"}])
        self.assertEqual(args["sizes"], {"c0": 2, "c1": 1})

    def test_unknown_lens_is_refused(self):
        code, _, err = self.args_for("--lenses", "anchor,skeptic")
        self.assertEqual(code, 2)
        self.assertIn("skeptic", err)

    def test_stamp_locks_the_verdict_dir_to_one_panel(self):
        code, _, _ = self.args_for("--model", "sonnet", "--effort", "high", "--stamp")
        self.assertEqual(code, 0)
        stamp = json.loads((self.votes / council_io.STAMP).read_text())
        self.assertEqual((stamp["model"], stamp["effort"], stamp["council"]),
                         ("sonnet", "high", "fused"))
        # resuming without --model picks the panel up from the stamp
        code, args, _ = self.args_for()
        self.assertEqual((args["model"], args["effort"]), ("sonnet", "high"))
        # resuming with another model is refused
        code, _, err = self.args_for("--model", "opus", "--effort", "xhigh")
        self.assertEqual(code, 2)
        self.assertIn("panel.meta", err)

    def test_stamp_is_not_read_as_a_verdict_file(self):
        self.args_for("--model", "sonnet", "--effort", "high", "--stamp")
        self.assertEqual(list(self.votes.glob("*.json")), [])

    def test_complete_council_has_no_jobs(self):
        write_panel(self.votes, self.c, {lens: "TFT" for lens in FUSED})
        code, args, err = self.args_for("--model", "sonnet", "--effort", "high")
        self.assertEqual((code, args["jobs"]), (0, []))
        self.assertIn("complete", err)


def calibration_fixture(tmp: pathlib.Path) -> dict:
    chunks = {"c0": [fused_item("s1", "x"), fused_item("s2", "x"), fused_item("s3", "y")],
              "c1": [fused_item("s4", "y"), fused_item("s5", "y")]}
    for name, items in chunks.items():
        write_json(tmp / "chunks" / f"{name}.json", items)
    # A (reference) accepts s1, s3; B (lenient) accepts all five
    write_panel(tmp / "a", chunks, {"anchor": "TFTFT", "coherence": "TFTTF",
                                    "degeneracy": "TTTFT", "fluency": "TTFTF",
                                    "register": "TTTTT"})
    write_panel(tmp / "b", chunks, {"anchor": "TTTFT", "coherence": "TTTTT",
                                    "degeneracy": "TTTTT", "fluency": "TTFTF",
                                    "register": "TTTTT"})
    return chunks


class CompareCouncils(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        calibration_fixture(self.tmp)
        self.results = self.tmp / "results"

    def compare(self, *extra) -> tuple[int, dict | None, str]:
        code, _, err = run_main(compare_councils.main, [
            "--council", "fused", "--chunk-dir", str(self.tmp / "chunks"),
            "--a-dir", str(self.tmp / "a"), "--a-label", "opus/xhigh",
            "--b-dir", str(self.tmp / "b"), "--b-label", "sonnet/high",
            "--name", "fixture", "--results-dir", str(self.results), *extra])
        path = self.results / "council_calibration_fixture.json"
        return code, (json.loads(path.read_text()) if path.exists() else None), err

    def test_final_confusion_and_leniency(self):
        code, report, _ = self.compare()
        self.assertEqual(code, 0)
        final = report["final"]
        self.assertEqual(final["n"], 5)
        self.assertEqual(final["confusion"], {"a_accept_b_accept": 2, "a_accept_b_reject": 0,
                                              "a_reject_b_accept": 3, "a_reject_b_reject": 0})
        self.assertAlmostEqual(final["a"]["reject_rate"], 0.6)
        self.assertAlmostEqual(final["b"]["reject_rate"], 0.0)
        self.assertAlmostEqual(final["leniency_pp"], 60.0)
        self.assertAlmostEqual(final["cohen_kappa"], 0.0)
        self.assertEqual((final["b_false_accept_vs_a"]["k"], final["b_false_accept_vs_a"]["n"]), (3, 5))
        self.assertEqual((final["b_miss_rate_vs_a"]["k"], final["b_miss_rate_vs_a"]["n"]), (3, 3))

    def test_per_lens_rates_and_kappa(self):
        _, report, _ = self.compare()
        lenses = report["lenses"]
        self.assertAlmostEqual(lenses["anchor"]["a_accept_rate"], 0.6)
        self.assertAlmostEqual(lenses["anchor"]["b_accept_rate"], 0.8)
        self.assertAlmostEqual(lenses["anchor"]["leniency_pp"], 20.0)
        self.assertAlmostEqual(lenses["anchor"]["cohen_kappa"], 0.5455, places=4)
        self.assertAlmostEqual(lenses["fluency"]["cohen_kappa"], 1.0)
        self.assertIsNone(lenses["register"]["cohen_kappa"])
        self.assertEqual(lenses["degeneracy"]["counts"],
                         {"both_accept": 4, "a_only_accept": 0, "b_only_accept": 1,
                          "both_reject": 0})

    def test_threshold_sensitivity_and_by_field(self):
        _, report, _ = self.compare("--by", "anchor")
        by_k = {row["b_min_agree"]: row for row in report["b_threshold_sensitivity"]}
        self.assertAlmostEqual(by_k[5]["b_accept_rate"], 0.4)   # s1, s2 unanimous in B
        self.assertEqual(by_k[4]["b_accept_rate"], 1.0)
        by = report["by"]["anchor"]
        self.assertEqual(by["x"]["n"], 2)
        self.assertAlmostEqual(by["y"]["a_reject_rate"], 2 / 3, places=4)

    def test_disagreements_name_the_flipped_items(self):
        _, report, _ = self.compare()
        self.assertEqual([d["sid"] for d in report["disagreements"]], ["s2", "s4", "s5"])
        self.assertIn("anchor", report["disagreements"][0]["a_rejected_on"])

    def test_report_records_reproducibility_fields(self):
        _, report, _ = self.compare()
        self.assertIn("compare_councils.py", report["command"])
        self.assertEqual(report["corpus"]["n_items"], 5)
        self.assertEqual(set(report["corpus"]["chunk_sha256"]), {"c0", "c1"})
        self.assertTrue(report["coverage"]["complete"])
        md = (self.results / "council_calibration_fixture.md").read_text()
        self.assertIn("compare_councils.py", md)
        self.assertIn("+60.0 pp", md)

    def test_output_is_byte_identical_on_rerun(self):
        self.compare()
        first = [(self.results / f"council_calibration_fixture.{e}").read_bytes()
                 for e in ("json", "md")]
        self.compare()
        second = [(self.results / f"council_calibration_fixture.{e}").read_bytes()
                  for e in ("json", "md")]
        self.assertEqual(first, second)

    def test_incomplete_panel_is_flagged_not_hidden(self):
        (self.tmp / "b" / "register_c1.json").unlink()
        code, report, _ = self.compare()
        self.assertEqual(code, 0)
        self.assertFalse(report["coverage"]["complete"])
        self.assertEqual(report["final"]["n"], 3)
        self.assertEqual(report["coverage"]["b_pending_jobs"],
                         [{"chunk": "c1", "lens": "register", "reason": "missing"}])
        md = (self.results / "council_calibration_fixture.md").read_text()
        self.assertIn("INCOMPLETE", md)

    def test_require_complete_refuses(self):
        (self.tmp / "b" / "register_c1.json").unlink()
        code, report, _ = self.compare("--require-complete")
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_bad_name_is_refused(self):
        code, _, _ = run_main(compare_councils.main, [
            "--council", "fused", "--chunk-dir", str(self.tmp / "chunks"),
            "--a-dir", str(self.tmp / "a"), "--a-label", "a",
            "--b-dir", str(self.tmp / "b"), "--b-label", "b",
            "--name", "../escape", "--results-dir", str(self.results)])
        self.assertEqual(code, 2)


def audit_fixture(tmp: pathlib.Path) -> dict:
    """40 items: i%4 == 0 unanimous, 1 dissent anchor, 2 dissent fluency,
    3 two dissents (rejected). 30 accepted, 10 per stratum."""
    items = [fused_item(f"s{i:02d}", anchor=f"w{i % 2}") for i in range(40)]
    chunks = {"c0": items[:20], "c1": items[20:]}
    for name, rows in chunks.items():
        write_json(tmp / "chunks" / f"{name}.json", rows)
    pattern = {lens: "" for lens in FUSED}
    for i in range(40):
        dissent = {0: set(), 1: {"anchor"}, 2: {"fluency"}, 3: {"anchor", "fluency"}}[i % 4]
        for lens in FUSED:
            pattern[lens] += "F" if lens in dissent else "T"
    write_panel(tmp / "bulk", chunks, pattern)
    (tmp / "source.jsonl").write_text("".join(
        json.dumps({"sentence_id": it["sid"],
                    "anchor": {"context_type": "trap" if int(it["sid"][1:]) < 20 else "plain"}})
        + "\n" for it in items), encoding="utf-8")
    return chunks


class SampleAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.chunks = audit_fixture(self.tmp)
        self.all_items = {i["sid"]: i for c in self.chunks.values() for i in c}

    def draw(self, out: str, *extra) -> tuple[int, dict | None, str]:
        code, _, err = run_main(sample_audit.main, [
            "draw", "--council", "fused", "--chunk-dir", str(self.tmp / "chunks"),
            "--verdict-dir", str(self.tmp / "bulk"), "--label", "sonnet/high",
            "--out-dir", str(self.tmp / out), *extra])
        path = self.tmp / out / "sample.json"
        return code, (json.loads(path.read_text()) if path.exists() else None), err

    def test_draw_samples_only_accepted_items_by_stratum(self):
        code, manifest, _ = self.draw("audit", "--n", "12", "--min-per-stratum", "2",
                                      "--seed", "7")
        self.assertEqual(code, 0)
        strata = {s["stratum"]: (s["N"], s["n"]) for s in manifest["strata"]}
        self.assertEqual(strata, {"unanimous": (10, 4), "dissent:anchor": (10, 4),
                                  "dissent:fluency": (10, 4)})
        self.assertEqual(manifest["population"]["n_accepted"], 30)
        self.assertEqual(manifest["population"]["n_rejected"], 10)
        for entry in manifest["sample"]:
            self.assertNotEqual(int(entry["sid"][1:]) % 4, 3, "a rejected item was sampled")

    def test_audit_chunks_hold_the_original_items_unchanged(self):
        self.draw("audit", "--n", "12", "--min-per-stratum", "2", "--chunk-size", "5")
        chunks = council_io.load_chunks(self.tmp / "audit")
        self.assertEqual([len(c) for c in chunks.values()], [5, 5, 2])
        for items in chunks.values():
            for item in items:
                self.assertEqual(item, self.all_items[item["sid"]])

    def test_draw_is_deterministic_and_seed_sensitive(self):
        for out, seed in (("a1", "7"), ("a2", "7"), ("a3", "8")):
            self.draw(out, "--n", "12", "--min-per-stratum", "2", "--seed", seed)

        def read(d):
            manifest = json.loads((self.tmp / d / "sample.json").read_text())
            del manifest["command"]          # names the out dir, which differs
            return manifest, (self.tmp / d / "c0.json").read_bytes()

        self.assertEqual(read("a1"), read("a2"))
        self.assertNotEqual(read("a1")[1], read("a3")[1])
        self.assertNotEqual(read("a1")[0]["sample"], read("a3")[0]["sample"])

    def test_floor_oversamples_small_strata(self):
        _, manifest, _ = self.draw("audit", "--n", "3", "--min-per-stratum", "2")
        self.assertEqual({s["n"] for s in manifest["strata"]}, {2})

    def test_source_and_item_strata(self):
        _, manifest, _ = self.draw(
            "audit", "--n", "30", "--min-per-stratum", "1",
            "--strata", "source:anchor.context_type", "--strata", "item:anchor",
            "--source", str(self.tmp / "source.jsonl"))
        names = {s["stratum"] for s in manifest["strata"]}
        self.assertIn("anchor.context_type=trap | anchor=w0", names)
        self.assertEqual(sum(s["N"] for s in manifest["strata"]), 30)

    def test_incomplete_bulk_panel_is_refused(self):
        (self.tmp / "bulk" / "anchor_c1.json").unlink()
        code, manifest, err = self.draw("audit")
        self.assertEqual(code, 2)
        self.assertIsNone(manifest)
        self.assertIn("incomplete", err)

    def test_existing_audit_dir_is_not_overwritten(self):
        self.draw("audit", "--n", "12")
        code, _, err = self.draw("audit", "--n", "12", "--seed", "99")
        self.assertEqual(code, 2)
        self.assertIn("--replace", err)

    def estimate(self, *extra) -> tuple[int, dict | None, str]:
        code, _, err = run_main(sample_audit.main, [
            "estimate", "--sample-dir", str(self.tmp / "audit"),
            "--verdict-dir", str(self.tmp / "ref"), "--label", "opus/xhigh",
            "--name", "fixture", "--results-dir", str(self.tmp / "results"), *extra])
        path = self.tmp / "results" / "council_audit_fixture.json"
        return code, (json.loads(path.read_text()) if path.exists() else None), err

    def reference_rejects(self, reject) -> None:
        chunks = council_io.load_chunks(self.tmp / "audit")
        manifest = json.loads((self.tmp / "audit" / "sample.json").read_text())
        stratum = {e["sid"]: e["stratum"] for e in manifest["sample"]}
        order = [i for c in chunks.values() for i in c]
        pattern = "".join("F" if reject(stratum[i["sid"]]) else "T" for i in order)
        write_panel(self.tmp / "ref", chunks, {
            lens: pattern if lens in ("anchor", "fluency") else "T" * len(order)
            for lens in FUSED})

    def test_estimate_weights_strata_and_reports_wilson(self):
        self.draw("audit", "--n", "12", "--min-per-stratum", "2", "--seed", "7")
        self.reference_rejects(lambda s: s == "dissent:anchor")
        code, report, _ = self.estimate()
        self.assertEqual(code, 0)
        est = report["false_accept"]
        self.assertAlmostEqual(est["rate"], 1 / 3, places=4)
        self.assertAlmostEqual(est["n_eff"], 12.0, places=4)
        lo, hi = council_io.wilson_interval(4, 12)
        self.assertAlmostEqual(est["wilson95"][0], round(lo, 4), places=4)
        self.assertAlmostEqual(est["wilson95"][1], round(hi, 4), places=4)
        self.assertEqual(len(report["false_accepts"]), 4)
        md = (self.tmp / "results" / "council_audit_fixture.md").read_text()
        self.assertIn("sample_audit.py", md)

    def test_estimate_refuses_an_incomplete_audit(self):
        self.draw("audit", "--n", "12")
        self.reference_rejects(lambda s: False)
        (self.tmp / "ref" / "register_c0.json").unlink()
        code, report, err = self.estimate()
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertIn("incomplete", err)

    def test_estimate_flags_a_bulk_dir_changed_after_the_draw(self):
        self.draw("audit", "--n", "12")
        self.reference_rejects(lambda s: False)
        path = self.tmp / "bulk" / "anchor_c0.json"
        rows = json.loads(path.read_text())
        rows[0]["why"] = "edited after the draw"
        path.write_text(json.dumps(rows), encoding="utf-8")
        code, report, _ = self.estimate()
        self.assertEqual(code, 0)
        self.assertTrue(report["warnings"])
        self.assertIn("changed since the draw", " ".join(report["warnings"]))


# ---------------------------------------------------------------- JS scripts

SCRIPTS = {"fused": COUNCIL / "fused_council.js", "pairs": COUNCIL / "pair_council.js"}

# sha256 of the LENSES, prompt and SUMMARY blocks of the Opus session council
# (council-fused-v2, run wf_c8f42ae0-f11, 2026-09-25). The committed fused
# council must stay byte-identical to it, or its verdicts stop being comparable.
FUSED_BLOCK_SHA256 = {
    "LENSES": "20d628553cbd5c7e23ce0447be6c1b62a27b5a019d652456c30ab8cef143e6af",
    "prompt": "19a261331d81d35e7ce120990ebd313441723fc68afdf9c970981497eca049f7",
    "SUMMARY": "7646a6df3284015ef60aca4c71608bc9577462095d7e34cc1dcdf72a249b3d98",
}


def js_block(text: str, name: str) -> str:
    start = {"LENSES": "const LENSES = {", "prompt": "const prompt = ",
             "SUMMARY": "const SUMMARY = {"}[name]
    lines = text.splitlines(keepends=True)
    i = next(k for k, line in enumerate(lines) if line.startswith(start))
    if name == "prompt":
        j = next(k for k in range(i, len(lines))
                 if lines[k].rstrip("\n").endswith("marked false.`"))
    else:
        j = next(k for k in range(i, len(lines)) if lines[k].rstrip("\n") == "}")
    return "".join(lines[i:j + 1])


def meta_literal(text: str) -> dict:
    """Parse the `export const meta = {...}` header as JSON -- which only
    works if it is a pure literal (strings, lists, objects; nothing computed)."""
    match = re.match(r"export const meta = (\{.*?\n\})\n", text, re.S)
    if not match:
        raise AssertionError("script does not open with `export const meta = {...}`")
    body, strings = match.group(1), []

    def stash(m: re.Match) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    body = re.sub(r"'([^'\\]*)'", stash, body)
    if re.search(r"[`$()]|\bnew\b|=>", body):
        raise AssertionError("meta header is not a pure literal")
    body = re.sub(r"(\w+)\s*:", r'"\1":', body)
    body = re.sub(r",(\s*[}\]])", r"\1", body)
    body = re.sub(r"\x00(\d+)\x00", lambda m: json.dumps(strings[int(m.group(1))]), body)
    return json.loads(body)


class WorkflowScriptsStatic(unittest.TestCase):
    def test_meta_is_a_pure_literal(self):
        for council, path in SCRIPTS.items():
            meta = meta_literal(path.read_text(encoding="utf-8"))
            self.assertEqual(set(meta), {"name", "description", "phases"}, council)

    def test_no_clock_randomness_or_filesystem(self):
        forbidden = ["Date.now", "new Date", "Math.random", "performance.now",
                     "require(", "import(", "process.", "fs.", "Deno.", "Bun."]
        for path in SCRIPTS.values():
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name}: {token}")
            self.assertIsNone(re.search(r"^import ", text, re.M), path.name)

    def test_model_and_effort_come_from_args(self):
        for path in SCRIPTS.values():
            text = path.read_text(encoding="utf-8")
            self.assertIn("model: args.model", text)
            self.assertIn("effort: args.effort", text)
            self.assertIsNone(re.search(r"model:\s*'", text), path.name)

    def test_lens_names_match_the_python_tools(self):
        for council, path in SCRIPTS.items():
            block = js_block(path.read_text(encoding="utf-8"), "LENSES")
            names = tuple(sorted(re.findall(r"^  (\w+): '", block, re.M)))
            self.assertEqual(names, council_io.COUNCILS[council])

    def test_fused_lenses_prompt_and_schema_are_verbatim(self):
        text = SCRIPTS["fused"].read_text(encoding="utf-8")
        for name, digest in FUSED_BLOCK_SHA256.items():
            got = hashlib.sha256(js_block(text, name).encode("utf-8")).hexdigest()
            self.assertEqual(got, digest, f"fused {name} drifted from the Opus council")

    def test_pair_prompt_frames_one_word_form_pair(self):
        text = SCRIPTS["pairs"].read_text(encoding="utf-8")
        prompt = js_block(text, "prompt")
        for needle in ("word, form, why_flagged, examples", "Judge ONLY that one pair",
                       "3amr -> amr", "leha -> lha", "3nd -> and",
                       "FORM is the same word as WORD"):
            self.assertIn(needle, prompt)
        lenses = js_block(text, "LENSES")
        self.assertIn("Default to v = false", lenses)


NODE = shutil.which("node")

HARNESS = r"""
const fs = require('fs')
const src = fs.readFileSync(process.argv[2], 'utf8').replace(/^export const meta/m, 'const meta')
const args = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))
const calls = []
const agent = async (prompt, opts) => {
  calls.push({ prompt, opts })
  const [lens, chunk] = opts.label.split(':')
  return { chunk, lens, n_judged: args.sizes[chunk], n_invalid: 0 }
}
const parallel = async thunks => Promise.all(thunks.map(t => t()))
const noop = () => {}
const AsyncFunction = (async () => {}).constructor
new AsyncFunction('args', 'agent', 'parallel', 'log', 'phase', src)(args, agent, parallel, noop, noop)
  .then(result => console.log(JSON.stringify({ ok: true, result, calls })),
        error => console.log(JSON.stringify({ ok: false, error: String(error.message) })))
"""


@unittest.skipUnless(NODE, "node not installed")
class WorkflowScriptsRun(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "harness.js").write_text(HARNESS, encoding="utf-8")

    def run_script(self, council: str, args: dict) -> dict:
        (self.tmp / "args.json").write_text(json.dumps(args), encoding="utf-8")
        proc = subprocess.run([NODE, str(self.tmp / "harness.js"), str(SCRIPTS[council]),
                               str(self.tmp / "args.json")],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def base_args(self, council: str) -> dict:
        lenses = council_io.COUNCILS[council]
        return {"council": council, "model": "sonnet", "effort": "high",
                "chunkDir": "/chunks", "outDir": "/votes", "sizes": {"c0": 3, "c1": 2},
                "jobs": [{"chunk": "c1", "lens": lenses[0]},
                         {"chunk": "c0", "lens": lenses[-1]}]}

    def test_each_job_becomes_one_judge_on_the_requested_panel(self):
        for council in SCRIPTS:
            out = self.run_script(council, self.base_args(council))
            self.assertTrue(out["ok"], out)
            self.assertEqual(len(out["calls"]), 2)
            lens = council_io.COUNCILS[council][0]
            first = out["calls"][0]
            self.assertEqual((first["opts"]["model"], first["opts"]["effort"]),
                             ("sonnet", "high"))
            self.assertIn(f"/chunks/c1.json", first["prompt"])
            self.assertIn(f"/votes/{lens}_c1.json", first["prompt"])
            self.assertEqual(out["result"]["returned"], 2)
            self.assertEqual(out["result"]["short"], [])
            self.assertEqual(out["result"]["model"], "sonnet")

    def test_missing_model_unknown_lens_or_wrong_council_fail_loudly(self):
        for council in SCRIPTS:
            args = self.base_args(council)
            del args["model"]
            self.assertFalse(self.run_script(council, args)["ok"])
            args = self.base_args(council)
            args["jobs"][0]["lens"] = "toString"
            self.assertFalse(self.run_script(council, args)["ok"])
            args = self.base_args(council)
            args["council"] = "pairs" if council == "fused" else "fused"
            self.assertFalse(self.run_script(council, args)["ok"])
            args = self.base_args(council)
            args["jobs"][0]["chunk"] = "c9"
            self.assertFalse(self.run_script(council, args)["ok"])


if __name__ == "__main__":
    unittest.main()
