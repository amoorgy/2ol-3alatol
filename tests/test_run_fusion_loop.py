"""run_fusion_v2.sh: the resume loop around build_dataset_b.py --plan.

A fake runner replays a scripted list of exit codes and records its argv,
so the loop's contract is tested without any model: 75 -> sleep and resume,
0 -> stop, anything else -> stop and log; the loop always adds --resume
exactly once; launching detaches exactly once and refuses a second loop.
Run under /bin/bash on purpose: macOS ships bash 3.2.
"""

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "run_fusion_v2.sh"

FAKE_RUNNER = r'''
import json, pathlib, sys, time
argv = sys.argv[1:]
audit = pathlib.Path(argv[argv.index("--audit") + 1])
state = audit.parent
calls = state / "calls.jsonl"
with calls.open("a") as handle:
    handle.write(json.dumps(argv) + "\n")
n = len(calls.read_text().splitlines())
codes = json.loads((state / "codes.json").read_text())
delay = state / "sleep.txt"
if delay.exists():
    time.sleep(float(delay.read_text()))
if (state / "progress").exists():   # simulate jobs completing each call
    audit.with_name(audit.stem + ".checkpoint.json").write_text(
        json.dumps({"jobs_done": n * 10}))
print(f"fake runner call {n}")
sys.exit(codes[min(n, len(codes)) - 1])
'''


@unittest.skipUnless(shutil.which("caffeinate") and pathlib.Path("/bin/bash").exists(),
                     "macOS caffeinate + /bin/bash required")
class Loop(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)
        self.runner = self.dir / "fake_runner.py"
        self.runner.write_text(FAKE_RUNNER)
        self.audit = self.dir / "fused_audit.jsonl"
        self.log = self.dir / "fused_audit.run.log"

    def script(self, codes, *extra, foreground=True, timeout=30):
        (self.dir / "codes.json").write_text(json.dumps(codes))
        argv = ["/bin/bash", str(SCRIPT), "--python", sys.executable,
                "--runner", str(self.runner), "--wait-minutes", "0",
                "--plan", str(self.dir / "plan.jsonl"), "--audit", str(self.audit),
                "--out", str(self.dir / "out.jsonl"), "--fusor", "zai:glm-5.3",
                "--teachers", "a,b", *extra]
        if foreground:
            argv.append("--foreground")
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=self.dir)

    def calls(self):
        path = self.dir / "calls.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def test_quota_pauses_are_resumed_until_done(self):
        result = self.script([75, 75, 0])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        for argv in calls:
            self.assertEqual(argv.count("--resume"), 1)
            self.assertEqual(argv[argv.index("--plan") + 1], str(self.dir / "plan.jsonl"))
            self.assertIn("--teachers", argv)
            self.assertNotIn("--wait-minutes", argv)  # loop options stay in the loop
        self.assertIn("quota pause 2", result.stdout)
        self.assertIn("runner exit 0", result.stdout)

    def test_any_other_exit_stops_and_is_logged(self):
        for code in (1, 2, 3):
            with self.subTest(code=code):
                (self.dir / "calls.jsonl").unlink(missing_ok=True)
                result = self.script([code, 0])
                self.assertEqual(result.returncode, code)
                self.assertEqual(len(self.calls()), 1)
                self.assertIn(f"runner exit {code}: stopping", result.stdout)

    def test_model_outages_are_resumed_until_done(self):
        result = self.script([76, 76, 0], "--outage-wait-minutes", "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 3)
        self.assertIn("outage pause 2", result.stdout)

    def test_outages_without_progress_give_up_at_the_cap(self):
        result = self.script([76] * 6, "--outage-wait-minutes", "0",
                             "--max-outage-pauses", "2")
        self.assertEqual(result.returncode, 76)
        self.assertEqual(len(self.calls()), 3)   # 1 run + 2 resumes, then give up

    def test_progress_resets_the_outage_count(self):
        (self.dir / "progress").write_text("")
        result = self.script([76, 76, 76, 0], "--outage-wait-minutes", "0",
                             "--max-outage-pauses", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 4)

    def test_max_pauses_gives_up(self):
        result = self.script([75, 75, 0], "--max-pauses", "1")
        self.assertEqual(result.returncode, 75)
        self.assertEqual(len(self.calls()), 2)

    def test_a_passed_resume_is_not_doubled(self):
        self.script([0], "--resume")
        self.assertEqual(self.calls()[0].count("--resume"), 1)

    def test_missing_required_options_never_start_the_runner(self):
        result = subprocess.run(["/bin/bash", str(SCRIPT), "--python", sys.executable,
                                 "--runner", str(self.runner), "--foreground",
                                 "--audit", str(self.audit)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--plan", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_launch_detaches_once_and_refuses_a_second_loop(self):
        (self.dir / "sleep.txt").write_text("4")
        started = time.monotonic()
        first = self.script([0], foreground=False)
        self.assertLess(time.monotonic() - started, 4, "the launcher must not wait")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("LAUNCHED", first.stdout)
        pidfile = self.dir / "fused_audit.loop.pid"
        pgid = int(pidfile.read_text())
        self.addCleanup(self._kill_group, pgid)
        second = self.script([0], foreground=False)
        self.assertEqual(second.returncode, 2)
        self.assertIn("already running", second.stderr)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and pidfile.exists():
            time.sleep(0.2)
        self.assertFalse(pidfile.exists(), "the loop removes its pidfile on exit")
        self.assertEqual(len(self.calls()), 1)
        log = self.log.read_text()
        self.assertIn("fake runner call 1", log)
        self.assertIn("loop end: exit 0", log)

    @staticmethod
    def _kill_group(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


if __name__ == "__main__":
    unittest.main()
