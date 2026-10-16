import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COUNCIL_DIR = ROOT / "experiments" / "d5_cohere_track" / "training_data" / "council"
SPEC = importlib.util.spec_from_file_location("glm_council", COUNCIL_DIR / "glm_council.py")
gc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gc)

ITEMS = [{"sid": f"s{i}", "rid": f"s{i}", "anchor": "meen", "arabic": "مين",
          "gloss": "who", "sentence": f"sentence {i}"} for i in range(5)]


class PromptFromTheMethodOfRecord(unittest.TestCase):
    """The GLM council must judge by the SAME criteria as the committed
    council scripts: lens texts and framing are read from the .js files."""

    def test_fused_lenses_and_framing_come_from_the_script(self):
        framing, lenses = gc.load_council("fused")
        self.assertEqual(set(lenses), {"fluency", "anchor", "degeneracy", "coherence", "register"})
        self.assertIn("ANCHOR SENSE.", lenses["anchor"])
        self.assertIn("{lens}", framing)
        self.assertNotIn("Read the chunk file", framing)

    def test_pair_framing_keeps_the_native_ruling(self):
        framing, lenses = gc.load_council("pairs")
        self.assertEqual(set(lenses), {"coherence", "identity", "plausibility", "register", "skeptic"})
        self.assertIn("A native speaker ruled", framing)

    def test_prompt_inlines_every_item_and_asks_for_json_only(self):
        framing, lenses = gc.load_council("fused")
        prompt = gc.build_prompt(framing, lenses["anchor"], ITEMS)
        for item in ITEMS:
            self.assertIn(f'"sentence": "{item["sentence"]}"', prompt)
            self.assertNotIn(f'"{item["sid"]}"', prompt)   # indexed, not by sid
        self.assertIn(lenses["anchor"], prompt)
        self.assertNotIn("Write tool", prompt)
        self.assertIn("JSON array", prompt)


class ParseVerdicts(unittest.TestCase):
    def reply(self, rows, wrap="```json\n{}\n```"):
        return wrap.format(json.dumps(rows))

    def test_items_are_keyed_by_a_small_index_not_the_sid(self):
        """GLM garbled near-identical sids (fs_f2dan_00 -> fs_f2dan_0) and a
        batch failed 3 retries; small integers are copied reliably."""
        framing, lenses = gc.load_council("fused")
        prompt = gc.build_prompt(framing, lenses["anchor"], ITEMS)
        self.assertIn('"i": 1', prompt)
        self.assertIn("keys i, v, why", prompt)

    def test_fenced_reply_with_chatter_parses(self):
        rows = [{"i": n + 1, "v": True, "why": ""} for n, _ in enumerate(ITEMS)]
        got = gc.parse_verdicts("Here you go:\n" + self.reply(rows), ITEMS)
        self.assertEqual([r["sid"] for r in got], [i["sid"] for i in ITEMS])
        self.assertEqual(got[0]["rid"], ITEMS[0]["rid"])

    def test_string_booleans_are_accepted(self):
        rows = [{"i": str(n + 1), "v": "false", "why": "x"} for n, _ in enumerate(ITEMS)]
        self.assertFalse(gc.parse_verdicts(self.reply(rows), ITEMS)[0]["v"])

    def test_missing_or_out_of_range_items_are_rejected(self):
        rows = [{"i": n + 1, "v": True, "why": ""} for n, _ in enumerate(ITEMS)]
        with self.assertRaises(ValueError):
            gc.parse_verdicts(self.reply(rows[:-1]), ITEMS)
        bad = [dict(r) for r in rows]; bad[2]["i"] = 99
        with self.assertRaises(ValueError):
            gc.parse_verdicts(self.reply(bad), ITEMS)

    def test_a_reordered_reply_is_matched_by_index(self):
        rows = [{"i": n + 1, "v": n != 1, "why": ""} for n, _ in enumerate(ITEMS)]
        got = gc.parse_verdicts(self.reply(list(reversed(rows))), ITEMS)
        self.assertEqual([r["sid"] for r in got], [i["sid"] for i in ITEMS])
        self.assertFalse(got[1]["v"])


class JudgeChunk(unittest.TestCase):
    def test_batches_are_stitched_in_order_and_a_bad_batch_is_retried(self):
        calls = []

        def ask(prompt):
            calls.append(prompt)
            sids = [i["sid"] for i in ITEMS if f'"sentence": "{i["sentence"]}"' in prompt]
            if len(calls) == 1:
                return "not json"
            return json.dumps([{"i": n + 1, "v": s != "s3", "why": "" if s != "s3" else "bad"}
                               for n, s in enumerate(sids)])
        rows = gc.judge_chunk(ITEMS, "framing {lens}", "LENS", ask, batch_size=2, workers=1)
        self.assertEqual([r["sid"] for r in rows], [i["sid"] for i in ITEMS])
        self.assertFalse(rows[3]["v"])
        self.assertEqual(len(calls), 4)   # 3 batches + 1 retry of the malformed one

    def test_a_complete_file_is_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "anchor_c0.json"
            out.write_text(json.dumps([{"sid": i["sid"], "rid": i["rid"], "v": True, "why": ""}
                                       for i in ITEMS]))
            self.assertTrue(gc.is_complete(out, ITEMS))
            out.write_text(json.dumps([]))
            self.assertFalse(gc.is_complete(out, ITEMS))


if __name__ == "__main__":
    unittest.main()
