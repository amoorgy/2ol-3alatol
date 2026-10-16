"""plan_fusion_v2.py: a frozen, deterministic plan from the two banks.

Fixture banks only (temp files in the contract's raw-TSV format); the
committed banks are never read or written here.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.fusion import SCENARIOS
from franco_eval.fusion_jobs import load_plan

SCRIPT = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "plan_fusion_v2.py"

try:
    from franco_eval import scenario_bank  # noqa: F401  (another agent's loader)
    HAVE_LOADER = True
except ImportError:
    HAVE_LOADER = False


def load_planner():
    spec = importlib.util.spec_from_file_location("plan_fusion_v2_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANCHOR_HEADER = "word\tarabic\tgloss_en\tcontext_type\tsource\tcount\tdoc_freq\tband\tpriority"
ANCHORS = [
    ("2a2ema", "قائمة", "a list", "trap_contrast", "trap"),
    ("2a2em", "قائم", "standing", "trap_contrast", "trap"),
    ("def", "", "", "general", "bare"),
    ("2a2ema", "قيمة", "value", "trap_contrast", "trap"),  # duplicate word
    ("men", "", "", "general", "bare"),
    ("meen", "", "", "general", "bare"),
]
EXTRA = [(16, "food_cooking", "you're cooking lunch for the whole family today", "accepted"),
         (17, "money", "you're asking to borrow money until the end of the month", "accepted"),
         (18, "weather", "you're complaining about the heat all week long", "pending_review"),
         (19, "work", "you're asking your manager for a day off next week", "rejected"),
         (20, "friends", "you're telling a friend you can't come out tonight", "accepted")]


@unittest.skipUnless(HAVE_LOADER, "franco_eval.scenario_bank not present yet")
class Planner(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)
        self.anchor_bank = self.dir / "anchors.tsv"
        self.anchor_bank.write_text("\n".join(
            [ANCHOR_HEADER] + ["\t".join([*row, "10", "5", "200-500", str(i)])
                               for i, row in enumerate(ANCHORS, start=1)]) + "\n",
            encoding="utf-8")
        self.scenario_bank = self.dir / "scenarios.tsv"
        rows = [(i, "daily_life", text, "accepted") for i, text in enumerate(SCENARIOS)]
        self.scenario_bank.write_text("\n".join(
            ["scenario_id\tdomain\ttext\tstatus"]
            + ["\t".join(map(str, row)) for row in rows + EXTRA]) + "\n",
            encoding="utf-8")
        self.planner = load_planner()

    def plan(self, name="plan.jsonl", *extra):
        out = self.dir / name
        argv = ["--anchor-bank", str(self.anchor_bank),
                "--scenario-bank", str(self.scenario_bank),
                "--anchors", "4", "--scenarios-per-anchor", "5",
                "--out", str(out), *extra]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = self.planner.main(argv)
        self.printed = printed.getvalue()
        return code, out

    def test_writes_a_frozen_verifiable_plan(self):
        code, out = self.plan()
        self.assertEqual(code, 0, self.printed)
        plan = load_plan(out)
        self.assertTrue(plan.sidecar_checked)
        self.assertEqual(plan.sha256, hashlib.sha256(out.read_bytes()).hexdigest())
        words = []
        for job in plan.jobs:
            if job.anchor.word not in words:
                words.append(job.anchor.word)
        self.assertEqual(words, ["2a2ema", "2a2em", "def", "men"])  # 4 unique, bank order
        self.assertEqual(len(plan.jobs), 20)
        first = next(j for j in plan.jobs if j.anchor.word == "2a2ema")
        self.assertEqual(first.anchor.arabic, "قائمة")  # first row of the word wins
        used = {j.scenario_id for j in plan.jobs}
        self.assertTrue(used <= set(range(16)) | {16, 17, 20}, used)

    def test_meta_records_config_sources_and_command(self):
        _code, out = self.plan()
        meta = json.loads((self.dir / "plan.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sha256"], load_plan(out).sha256)
        self.assertEqual((meta["seed"], meta["scenarios_per_anchor"], meta["jobs"]),
                         (29, 5, 20))
        self.assertEqual(meta["anchor_bank"]["sha256"],
                         hashlib.sha256(self.anchor_bank.read_bytes()).hexdigest())
        self.assertEqual(meta["scenario_bank"]["accepted"], 19)
        self.assertEqual(meta["stats"]["duplicate_rows_skipped"], 1)
        self.assertEqual(meta["stats"]["conflicting_duplicates"], ["2a2ema"])
        self.assertIn("plan_fusion_v2.py", meta["command"])
        self.assertIn("--scenarios-per-anchor 5", meta["command"])

    def test_deterministic(self):
        _c, a = self.plan("a.jsonl")
        _c, b = self.plan("b.jsonl")
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_a_plan_is_never_overwritten(self):
        _c, out = self.plan()
        before = out.read_bytes()
        code, _ = self.plan("plan.jsonl", "--seed", "30")
        self.assertEqual(code, 2)
        self.assertEqual(out.read_bytes(), before)

    def test_exclude_audit_skips_done_jobs(self):
        _c, full = self.plan("full.jsonl")
        jobs = load_plan(full).jobs
        audit = self.dir / "old_audit.jsonl"
        legacy = next(j for j in jobs if j.scenario_id < 16)
        audit.write_text(
            json.dumps({"anchor": legacy.anchor.word, "scenario_index": legacy.scenario_id}) + "\n"
            + json.dumps({"anchor": jobs[-1].anchor.word, "scenario_index": 99,
                          "job_id": jobs[-1].job_id}) + "\n", encoding="utf-8")
        code, out = self.plan("rest.jsonl", "--exclude-audit", str(audit))
        self.assertEqual(code, 0, self.printed)
        rest = [j.job_id for j in load_plan(out).jobs]
        self.assertEqual(rest, [j.job_id for j in jobs
                                if j.job_id not in {legacy.job_id, jobs[-1].job_id}])
        meta = json.loads((self.dir / "rest.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["stats"]["jobs_excluded"], 2)
        self.assertEqual(meta["exclude_audits"][0]["jobs"], 2)

    def test_missing_exclude_audit_is_an_error_not_a_silent_no_op(self):
        code, out = self.plan("x.jsonl", "--exclude-audit", str(self.dir / "typo.jsonl"))
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())

    def test_k_above_the_accepted_scenarios_is_refused(self):
        code, out = self.plan("x.jsonl", "--scenarios-per-anchor", "20")
        self.assertEqual(code, 2)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
