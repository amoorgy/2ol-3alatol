import csv
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MASTER = ROOT / "data" / "generated" / "v2" / "master.jsonl"
SPLIT = ROOT / "data" / "stimulus" / "nilechat" / "word_split.tsv"
STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"


def heldout_words() -> set[str]:
    with SPLIT.open(encoding="utf-8") as handle:
        return {r["word"] for r in csv.DictReader(handle, delimiter="\t")
                if r["side"] == "heldout"}


@unittest.skipUnless(SPLIT.exists(), "word split not generated")
class SplitFile(unittest.TestCase):
    def test_every_probe_word_and_trap_is_assigned_a_side(self):
        with SPLIT.open(encoding="utf-8") as handle:
            sides = {r["word"]: r["side"] for r in csv.DictReader(handle, delimiter="\t")}
        with STIMULI.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.assertIn(row["word_type"], sides)
                if row["trap_word"]:
                    self.assertEqual(sides[row["trap_word"]], sides[row["word_type"]],
                                     "a trap pair straddles the split")


@unittest.skipUnless(MASTER.exists(), "v2 master file not generated yet")
class MasterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heldout = heldout_words()
        cls.protected = set(cls.heldout)
        with STIMULI.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["word_type"] in cls.heldout:
                    cls.protected.add(row["rendering"])

    def rows(self):
        with MASTER.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    def test_no_held_out_word_is_respelled_or_created(self):
        for row in self.rows():
            for sub in row["substitutions"]:
                with self.subTest(rid=row["rendering_id"]):
                    self.assertNotIn(sub["word"], self.heldout)
                    self.assertNotIn(sub["rendering"], self.protected)

    def test_every_substitution_sits_at_its_span(self):
        for row in self.rows():
            for sub in row["substitutions"]:
                self.assertEqual(row["text"][sub["start"]:sub["end"]],
                                 sub["rendering"], row["rendering_id"])

    def test_split_is_per_sentence_and_renderings_are_distinct(self):
        split_of: dict[str, str] = {}
        texts: dict[str, set] = {}
        for row in self.rows():
            sid = row["sentence_id"]
            self.assertEqual(split_of.setdefault(sid, row["split"]), row["split"])
            bucket = texts.setdefault(sid, set())
            self.assertNotIn(row["text"], bucket)
            bucket.add(row["text"])

    def test_provenance_and_source_on_every_row(self):
        for row in self.rows():
            self.assertIn(row["source"], ("nilechat", "fusion"))
            self.assertIn(row["provenance"],
                          ("nilechat_machine_derived", "fusion_synthetic"))


if __name__ == "__main__":
    unittest.main()
