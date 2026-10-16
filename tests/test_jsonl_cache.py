import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.model_clients import JsonlCache

KEYS = ("model", "rendering_id", "prompt_sha")


def record(i: int) -> dict:
    return {"model": "m", "rendering_id": f"r{i}", "prompt_sha": "s", "raw": f"answer {i}"}


class TornWrite(unittest.TestCase):
    """A kill mid-append leaves a half line; the next start must not die on it
    (a multi-day fusion run pauses and resumes many times)."""

    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.path = pathlib.Path(handle.name) / "cache.jsonl"
        self.path.write_text(json.dumps(record(1)) + "\n" + '{"model": "m", "rend',
                             encoding="utf-8")

    def test_loads_every_complete_record_and_counts_the_torn_one(self):
        cache = JsonlCache(self.path, KEYS)
        self.assertEqual(cache.get(record(1))["raw"], "answer 1")
        self.assertEqual(cache.skipped_lines, 1)

    def test_next_put_starts_on_a_fresh_line(self):
        JsonlCache(self.path, KEYS).put(record(2))
        reloaded = JsonlCache(self.path, KEYS)
        self.assertEqual(reloaded.get(record(2))["raw"], "answer 2")
        self.assertEqual(reloaded.get(record(1))["raw"], "answer 1")
        self.assertEqual(reloaded.skipped_lines, 1)   # the fragment, still isolated

    def test_the_file_is_never_rewritten(self):
        before = self.path.read_bytes()
        JsonlCache(self.path, KEYS)
        self.assertEqual(self.path.read_bytes(), before)


class CleanFile(unittest.TestCase):
    def test_round_trip_without_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "cache.jsonl"
            JsonlCache(path, KEYS).put(record(1))
            cache = JsonlCache(path, KEYS)
            self.assertEqual(cache.get(record(1))["raw"], "answer 1")
            self.assertEqual(cache.skipped_lines, 0)


if __name__ == "__main__":
    unittest.main()
