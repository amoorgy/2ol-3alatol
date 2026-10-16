"""End to end: run_fusion_v2.sh -> build_dataset_b.py --plan -> fake models.

The runner process is real (argparse, lock, checkpoint, signals); only the
model layer is replaced, by a wrapper that injects a fake `call` whose
quota state lives on disk so it survives across the loop's invocations.
The cache is a temp file, never data/generated/fusion_cache.jsonl.
"""

import json
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.fusion import SCENARIOS
from franco_eval.fusion_jobs import checkpoint_path, draw_v2_jobs, write_plan

LOOP = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "run_fusion_v2.sh"
RUNNER = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "build_dataset_b.py"

WRAPPER = r'''
import importlib.util, io, json, pathlib, re, signal, sys, time, urllib.error
STATE = pathlib.Path(%(state)r)
sys.path.insert(0, %(src)r)
spec = importlib.util.spec_from_file_location("bdb_e2e", %(runner)r)
bdb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdb)
from franco_eval.model_clients import QuotaExhausted

def fake(model, prompt, *, timeout, cfg):
    word = re.search(r'spelled exactly "([^"]+)"', prompt).group(1)
    if (STATE / "slow").exists():
        time.sleep(0.5)
    if model == "zai:glm-5.3":
        counter = STATE / "fusor_calls"
        n = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(n))
        if n == int((STATE / "quota_at").read_text()):
            error = urllib.error.HTTPError("u", 429, "m", {}, io.BytesIO(b""))
            error.close()
            raise QuotaExhausted(error, provider="zai", model="glm-5.3",
                                 error_code="1308", message="Usage limit reached")
    return f"ya sa7by embare7 {word} kan el yom gamed awy w ba2a keda"

signal.signal(signal.SIGTERM, bdb._terminate)
args = bdb.build_parser().parse_args(sys.argv[1:])
sys.exit(bdb.run_plan(args, call=fake, cache_path=STATE / "cache.jsonl"))
'''


class E2ECase(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)
        rows = [{"word": w, "arabic": "", "gloss_en": "", "context_type": "general",
                 "source": "bare", "band": ""} for w in ("men", "meen", "3ayez", "wa7ed")]
        jobs = draw_v2_jobs(rows, list(enumerate(SCENARIOS)), n_anchors=0, k=2,
                            seed=29).jobs
        self.plan = self.dir / "plan.jsonl"
        write_plan(self.plan, jobs)
        self.plan_ids = sorted(j.job_id for j in jobs)
        self.wrapper = self.dir / "runner_with_fakes.py"
        self.wrapper.write_text(WRAPPER % {"state": str(self.dir), "src": str(ROOT / "src"),
                                           "runner": str(RUNNER)})
        self.audit = self.dir / "fused_audit.jsonl"
        self.out = self.dir / "fused_sentences.jsonl"
        (self.dir / "quota_at").write_text("0")

    def runner_args(self):
        return ["--plan", str(self.plan), "--out", str(self.out), "--audit", str(self.audit),
                "--fusor", "zai:glm-5.3", "--teachers", "llama3.1:8b,deepseek:deepseek-v4-flash",
                "--workers", "1", "--batch-size", "3"]

    def audit_ids(self):
        return sorted(json.loads(l)["job_id"] for l in self.audit.read_text().splitlines())

    def checkpoint(self):
        return json.loads(checkpoint_path(self.audit).read_text())


@unittest.skipUnless(shutil.which("caffeinate"), "macOS caffeinate required")
class LoopAroundTheRealRunner(E2ECase):
    def test_quota_pause_then_resume_to_done(self):
        (self.dir / "quota_at").write_text("5")  # the 5th fusor call hits the limit
        result = subprocess.run(
            ["/bin/bash", str(LOOP), "--foreground", "--wait-minutes", "0",
             "--python", sys.executable, "--runner", str(self.wrapper), *self.runner_args()],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn("runner exit 75: quota pause 1", result.stdout)
        self.assertEqual(self.audit_ids(), self.plan_ids)
        ckpt = self.checkpoint()
        self.assertEqual((ckpt["status"], ckpt["invocations"], ckpt["quota_pauses"]),
                         ("done", 2, 1))
        self.assertEqual([h["status"] for h in ckpt["history"]],
                         ["paused_quota:zai:glm-5.3", "done"])
        self.assertEqual(ckpt["history"][0]["jobs_done"], 4)  # jobs 1-4, then the pause


class SigtermIsCheckpointed(E2ECase):
    def test_sigterm_mid_batch_records_interrupted_and_leaves_no_partial_row(self):
        (self.dir / "slow").write_text("")
        proc = subprocess.Popen([sys.executable, str(self.wrapper), *self.runner_args()],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        ckpt = checkpoint_path(self.audit)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not ckpt.exists():
            time.sleep(0.1)
        time.sleep(1.0)  # somewhere inside the first batch's prefetch
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, output[-2000:])
        self.assertEqual(self.checkpoint()["status"], "interrupted")
        text = self.audit.read_text() if self.audit.exists() else ""
        self.assertTrue(text == "" or text.endswith("\n"))
        (self.dir / "slow").unlink()
        resumed = subprocess.run([sys.executable, str(self.wrapper), *self.runner_args(),
                                  "--resume"], capture_output=True, text=True, timeout=60)
        self.assertEqual(resumed.returncode, 0, resumed.stdout[-2000:])
        self.assertEqual(self.audit_ids(), self.plan_ids)


if __name__ == "__main__":
    unittest.main()
