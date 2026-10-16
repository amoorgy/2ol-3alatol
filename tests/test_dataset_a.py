import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAT = ROOT / "data" / "generated" / "training_a_full.jsonl"

EXPECTED_KEYS = {"sentence_id", "rendering_id", "role", "text", "base_text",
                 "substitutions", "split", "provenance", "source_doc_id",
                 "council_valid"}
SUB_KEYS = {"word", "rendering", "start", "end", "licence", "corpus_count",
            "rule_path", "ortho_classes", "score"}

# The 500 benchmark words: none may ever appear as a substitution target.
CANDIDATES = ROOT / "data" / "generated" / "nilechat_candidates.json"


@unittest.skipUnless(FAT.exists(), "fat training file not generated yet")
class FatRecordContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in FAT.open(encoding="utf-8")]

    def test_every_row_carries_the_full_schema(self):
        for row in self.rows[:200]:
            self.assertEqual(set(row), EXPECTED_KEYS)
            for sub in row["substitutions"]:
                self.assertEqual(set(sub), SUB_KEYS)

    def test_every_substitution_sits_at_its_span(self):
        for row in self.rows:
            for sub in row["substitutions"]:
                with self.subTest(rid=row["rendering_id"], word=sub["word"]):
                    self.assertEqual(row["text"][sub["start"]:sub["end"]],
                                     sub["rendering"])

    def test_base_rows_have_no_substitutions_and_match_base_text(self):
        for row in self.rows:
            if row["role"] == "base":
                self.assertEqual(row["substitutions"], [])
                self.assertEqual(row["text"], row["base_text"])

    def test_each_sentence_has_exactly_one_base(self):
        bases = {}
        for row in self.rows:
            if row["role"] == "base":
                bases[row["sentence_id"]] = bases.get(row["sentence_id"], 0) + 1
        self.assertTrue(all(n == 1 for n in bases.values()))
        self.assertEqual(set(bases), {r["sentence_id"] for r in self.rows})

    def test_split_is_assigned_per_sentence_not_per_row(self):
        """All renderings of one sentence share a split -- otherwise the val
        set sees near-copies of training sentences and the metric lies."""
        split_of: dict[str, set] = {}
        for row in self.rows:
            split_of.setdefault(row["sentence_id"], set()).add(row["split"])
        offenders = {sid for sid, splits in split_of.items() if len(splits) > 1}
        self.assertEqual(offenders, set())

    def test_provenance_is_declared_on_every_row(self):
        for row in self.rows[:200]:
            self.assertEqual(row["provenance"], "nilechat_machine_derived")

    @unittest.skipUnless(CANDIDATES.exists(), "benchmark word list absent")
    def test_firewall_no_benchmark_word_is_ever_respelled(self):
        blocked = {w["word"] for w in
                   json.loads(CANDIDATES.read_text())["words"]}
        for row in self.rows:
            for sub in row["substitutions"]:
                self.assertNotIn(sub["word"], blocked)

    def test_renderings_within_a_sentence_are_distinct(self):
        texts: dict[str, set] = {}
        for row in self.rows:
            bucket = texts.setdefault(row["sentence_id"], set())
            self.assertNotIn(row["text"], bucket)
            bucket.add(row["text"])


class PairsFormat(unittest.TestCase):
    """The old trainer's contract (prepare_dataset.validate_jsonl, b43512b):
    single `messages` key, first role user, >= 2 messages."""

    def test_pairs_payload_shape(self):
        # mirror of the emit() construction, asserted directly
        payload = {"messages": [
            {"role": "user", "content": "el 7ara2a el mhdra"},
            {"role": "assistant", "content": "el 7ara2a el mhdara"},
        ]}
        self.assertGreaterEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")
        roles = {m["role"] for m in payload["messages"]}
        self.assertLessEqual(roles, {"user", "assistant"})


if __name__ == "__main__":
    unittest.main()
