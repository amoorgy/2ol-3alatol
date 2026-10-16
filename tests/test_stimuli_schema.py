import csv
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval import phenomena

STIMULI = (pathlib.Path(__file__).resolve().parents[1]
           / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv")

EXPECTED_FIELDS = [
    "rendering_id", "item_id", "word_type", "rendering", "role",
    "licence", "corpus_count", "collision", "collision_count",
    "rule_path", "phenomena", "ortho_classes", "n_edits", "score",
    "trap_word", "trap_reason",
    "sentence", "target_start", "target_end",
    "provenance", "source_doc_id",
    "council_valid", "council_reason",
]

KNOWN_CLASSES = {getattr(phenomena, name) for name in dir(phenomena)
                 if name.isupper() and isinstance(getattr(phenomena, name), str)}


@unittest.skipUnless(STIMULI.exists(), "stimulus file not built yet")
class StimulusContract(unittest.TestCase):
    """The frozen interface between the builders and both runners."""

    @classmethod
    def setUpClass(cls):
        with STIMULI.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.fields = reader.fieldnames
            cls.rows = list(reader)

    def test_header_is_exactly_the_contract(self):
        self.assertEqual(self.fields, EXPECTED_FIELDS)

    def test_rendering_ids_are_unique(self):
        ids = [r["rendering_id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_item_has_exactly_one_base(self):
        bases = {}
        for row in self.rows:
            if row["role"] == "base":
                bases[row["item_id"]] = bases.get(row["item_id"], 0) + 1
        for item_id, count in bases.items():
            self.assertEqual(count, 1, item_id)
        items = {r["item_id"] for r in self.rows}
        self.assertEqual(items, set(bases), "an item without a base row")

    def test_the_rendering_sits_at_its_span(self):
        for row in self.rows:
            start, end = int(row["target_start"]), int(row["target_end"])
            with self.subTest(rendering_id=row["rendering_id"]):
                self.assertEqual(row["sentence"][start:end], row["rendering"])

    def test_base_rendering_is_the_word_type(self):
        for row in self.rows:
            if row["role"] == "base":
                self.assertEqual(row["rendering"], row["word_type"])

    def test_roles_are_known(self):
        self.assertLessEqual({r["role"] for r in self.rows},
                             {"base", "variant", "trap"})

    def test_provenance_is_declared_on_every_row(self):
        """Hard rule 1: machine-derived stimulus must say so, row by row."""
        for row in self.rows:
            self.assertEqual(row["provenance"], "nilechat_machine_derived")

    def test_variant_classes_are_phenomena_constants(self):
        """The join key for the E1xE2 figure."""
        for row in self.rows:
            if row["role"] != "variant" or not row["ortho_classes"]:
                continue
            for cls in row["ortho_classes"].split("+"):
                with self.subTest(rendering_id=row["rendering_id"], cls=cls):
                    self.assertIn(cls, KNOWN_CLASSES)

    def test_traps_carry_their_word_and_reason(self):
        for row in self.rows:
            if row["role"] == "trap":
                self.assertTrue(row["trap_word"])
                self.assertEqual(row["ortho_classes"], "TRAP")

    def test_variants_never_equal_their_base(self):
        base_of = {r["item_id"]: r["rendering"] for r in self.rows
                   if r["role"] == "base"}
        for row in self.rows:
            if row["role"] == "variant":
                self.assertNotEqual(row["rendering"], base_of[row["item_id"]])


if __name__ == "__main__":
    unittest.main()
