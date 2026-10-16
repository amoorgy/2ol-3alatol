import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.licensing import (BY_ID, DENIED, LICENSED_OPS,
                                   apply_op, enumerate_variants, licensed_ops)
from franco_eval.phenomena import EXCLUDED_DIGITS, classify_edge


class TableInvariants(unittest.TestCase):
    def test_op_ids_are_unique(self):
        ids = [op.op_id for op in LICENSED_OPS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_pattern_compiles(self):
        for op in LICENSED_OPS:
            with self.subTest(op=op.op_id):
                re.compile(op.pattern)

    def test_every_weight_is_positive(self):
        for op in LICENSED_OPS:
            with self.subTest(op=op.op_id):
                self.assertGreater(op.weight, 0.0)

    def test_denied_table_documents_its_reasons(self):
        for op_id, reason in DENIED.items():
            with self.subTest(op=op_id):
                self.assertNotIn(op_id, BY_ID, "an op cannot be both licensed and denied")
                self.assertGreater(len(reason), 20, "a denial needs a real reason")


class DigitLicensing(unittest.TestCase):
    """The asymmetry that makes generation safe without gold Arabic."""

    def test_reversible_digits_are_licensed_in_both_directions(self):
        khasam = {apply_op("khasam", op, m) for op, m in licensed_ops("khasam")}
        back = {apply_op("5asam", op, m) for op, m in licensed_ops("5asam")}
        self.assertIn("5asam", khasam)
        self.assertIn("khasam", back)

    def test_ambiguous_digit_to_latin_is_always_licensed(self):
        forms = {apply_op("7aga", op, m) for op, m in licensed_ops("7aga")}
        self.assertIn("haga", forms)

    def test_latin_to_ambiguous_digit_is_denied_without_self_licence(self):
        """hena (هنا) must never yield 7ena: that h realises ه, not ح."""
        for variant in enumerate_variants("hena", max_depth=2):
            self.assertNotIn("7", variant.form)

    def test_latin_to_digit_is_denied_even_with_a_seven_in_the_base(self):
        """Council pass 1 (0/5 keep) overturned the original self-licence:
        in a word already spelling ح as 7, any REMAINING h is by construction
        ه -- 7abaha's -ha is a clitic -- so h->7 is precisely the merge the
        licence existed to prevent. H_TO_7 is DENIED outright."""
        for word in ("7abaha", "7aha", "gamil", "hena"):
            with self.subTest(word=word):
                for variant in enumerate_variants(word, max_depth=2):
                    # 7 -> h may LOWER the count (licensed direction); no h
                    # may ever become a 7, so the count never rises.
                    self.assertLessEqual(variant.form.count("7"),
                                         word.count("7"))

    def test_the_denial_is_recorded_with_its_reason(self):
        self.assertIn("H_TO_7", DENIED)
        self.assertIn("INVERTED", DENIED["H_TO_7"])

    def test_excluded_digits_are_never_generated(self):
        for word in ("tab3an", "khalas", "sa7", "shanta"):
            for variant in enumerate_variants(word, max_depth=2):
                for digit in EXCLUDED_DIGITS:
                    self.assertNotIn(digit, variant.form)

    def test_medial_hamza_is_never_dropped(self):
        """a2ollak's 2 is a root consonant (see skeleton.py's own tests)."""
        for variant in enumerate_variants("a2ollak", max_depth=2):
            self.assertIn("2", variant.form)

    def test_digit_four_is_denied_entirely(self):
        """Council 3/5 deny: 4=ش is Gulf/Levantine, not Egyptian. It was
        rare-gated before; now it is out of the table with a written reason."""
        self.assertIn("SH_TO_4", DENIED)
        for rare in (False, True):
            with self.subTest(allow_rare=rare):
                forms = {v.form for v in enumerate_variants("mesh", max_depth=1,
                                                            allow_rare=rare)}
                self.assertNotIn("me4", forms)


class NativeRulings(unittest.TestCase):
    """Rules signed off by the native speaker, with corpus evidence."""

    def test_u_is_never_generated_adjacent_to_w(self):
        """2026-08-24: howa must never become huwa (628,539:125 in corpus;
        guwa unattested). The w carries the /u/; writing u beside it is MSA
        transliteration style, not Egyptian Franco."""
        for word in ("howa", "gowa", "worsha", "elwosool", "tawos"):
            for variant in enumerate_variants(word, max_depth=2):
                with self.subTest(word=word, form=variant.form):
                    self.assertNotIn("uw", variant.form)
                    self.assertNotIn("wu", variant.form)

    def test_u_to_o_beside_w_stays_licensed(self):
        """The reverse direction moves toward the attested form."""
        forms = {v.form for v in enumerate_variants("huwa", max_depth=1)}
        self.assertIn("howa", forms)


class Determinism(unittest.TestCase):
    def test_generation_is_deterministic_without_a_seed(self):
        first = enumerate_variants("3ashan", max_depth=2)
        second = enumerate_variants("3ashan", max_depth=2)
        self.assertEqual(first, second)

    def test_the_module_does_not_import_random(self):
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "franco_eval" / "licensing.py").read_text()
        self.assertNotIn("import random", source)


class RoundTrip(unittest.TestCase):
    """The generator and the classifier must not drift apart."""

    def test_every_generated_variant_converges_under_classify_edge(self):
        for word in ("3ashan", "7aga", "beit", "khasam", "elly", "shanta"):
            for variant in enumerate_variants(word, max_depth=2):
                with self.subTest(pair=f"{word}/{variant.form}"):
                    self.assertTrue(classify_edge(word, variant.form).is_variant)

    def test_classes_field_carries_the_classifier_view(self):
        for variant in enumerate_variants("3ashan", max_depth=1):
            expected = classify_edge("3ashan", variant.form).classes
            self.assertEqual(variant.classes, expected)

    def test_derivation_and_explanation_may_legitimately_differ(self):
        """benhawel -> benhawl derives by deleting the epenthetic e, but other
        derivations read differently after canonicalisation. Both descriptions
        are true; analyses key on `classes`, provenance keeps `phenomena`.

        (The old witness was 3ashan -> 3shan, which died with the council's
        e/i-only restriction on medial deletion -- that a is a real vowel.)"""
        by_form = {v.form: v for v in enumerate_variants("benhawel", max_depth=1)}
        self.assertIn("benhawl", by_form)
        variant = by_form["benhawl"]
        self.assertEqual(variant.phenomena, ("VOWEL_DROP_MED",))


class MedialVowelBounds(unittest.TestCase):
    def test_at_most_one_medial_deletion_per_variant(self):
        for variant in enumerate_variants("mawdoo3", max_depth=2):
            drops = sum(1 for op_id, _at in variant.path
                        if op_id == "DROP_MEDIAL_VOWEL")
            self.assertLessEqual(drops, 1)

    def test_the_last_vowel_is_never_deleted(self):
        """mesh has one vowel, so no generated variant may be vowelless.

        Known cost: the heavily attested `msh` (25k occurrences) is therefore
        not generatable from `mesh`. Safety wins -- deleting the only vowel of
        an arbitrary word produces consonant husks far more often than real
        variants, and msh-style forms can enter through the attested tier of a
        larger base word instead.
        """
        for variant in enumerate_variants("mesh", max_depth=2):
            with self.subTest(form=variant.form):
                self.assertTrue(any(ch in "aeiou" for ch in variant.form))

    def test_deletion_never_fires_at_index_zero(self):
        for variant in enumerate_variants("ezay", max_depth=2):
            drops = [at for op_id, at in variant.path if op_id == "DROP_MEDIAL_VOWEL"]
            for at in drops:
                self.assertGreater(at, 0)

    def test_collisions_are_flagged_not_dropped(self):
        """men -> meen lands on a real different word (مين). Keep it, label it."""
        vocabulary = {"meen": 1552, "min": 628}
        by_form = {v.form: v for v in
                   enumerate_variants("men", max_depth=1, vocabulary=vocabulary)}
        self.assertIn("meen", by_form)
        self.assertEqual(by_form["meen"].collision, "meen")
        self.assertEqual(by_form["meen"].collision_count, 1552)

    def test_sub_threshold_collisions_are_not_flagged(self):
        vocabulary = {"meen": 10}
        by_form = {v.form: v for v in
                   enumerate_variants("men", max_depth=1, vocabulary=vocabulary,
                                      collision_min=50)}
        self.assertEqual(by_form["meen"].collision, "")


if __name__ == "__main__":
    unittest.main()
