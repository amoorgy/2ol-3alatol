import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval import phenomena
from franco_eval.meaning_metrics import ItemMeanings, modal_meaning, score


def item(**kwargs):
    defaults = dict(item_id="s1", word_type="3ayez", base="عايز",
                    variants=("عايز", "عايز", "عايز", "عايز"),
                    variant_classes=(("VOWEL_SWAP",),) * 4)
    defaults.update(kwargs)
    return ItemMeanings(**defaults)


class VariantAgreement(unittest.TestCase):
    def test_full_agreement_scores_zero_under_robustness(self):
        result = score([item()])
        self.assertEqual(result["variant_pairwise_agreement"], 1.0)
        self.assertEqual(result["under_robust_rate"], 0.0)
        self.assertEqual(result["variant_consistency_at_k"], 1.0)

    def test_total_disagreement_scores_one(self):
        result = score([item(variants=("ا", "ب", "ج", "د"), base="ه")])
        self.assertEqual(result["variant_pairwise_agreement"], 0.0)
        self.assertEqual(result["under_robust_rate"], 1.0)

    def test_a_none_among_variants_drops_agreement(self):
        """Inherited from pairwise_agreement: None never agrees. For variants
        a refusal IS a failure to demonstrate robustness."""
        full = score([item()])["variant_pairwise_agreement"]
        with_none = score([item(variants=("عايز", "عايز", "عايز", None))])
        self.assertLess(with_none["variant_pairwise_agreement"], full)

    def test_consistency_at_k_is_stricter_than_pairwise(self):
        result = score([item(variants=("عايز", "عايز", "عايز", "عاوز"))])
        self.assertLess(result["variant_consistency_at_k"],
                        result["variant_pairwise_agreement"])


class TrapScoring(unittest.TestCase):
    """The None asymmetry: refusing a trap must not count as detecting it."""

    def test_trap_matching_modal_meaning_is_over_normalisation(self):
        result = score([item(trap="عايز", has_trap=True)])
        self.assertEqual(result["over_norm_rate"], 1.0)
        self.assertEqual(result["n_traps_scored"], 1)

    def test_trap_with_distinct_meaning_is_correct(self):
        result = score([item(trap="عين", has_trap=True)])
        self.assertEqual(result["over_norm_rate"], 0.0)

    def test_a_none_trap_is_excluded_from_the_denominator(self):
        """The single most important assertion in this file."""
        result = score([item(trap=None, has_trap=True)])
        self.assertEqual(result["n_traps_scored"], 0)
        self.assertIsNone(result["over_norm_rate"])
        self.assertEqual(result["trap_non_answer_rate"], 1.0)

    def test_all_traps_none_yields_none_not_zero(self):
        result = score([item(trap=None, has_trap=True),
                        item(item_id="s2", trap=None, has_trap=True)])
        self.assertIsNone(result["over_norm_rate"])

    def test_no_modal_meaning_means_the_trap_cannot_be_scored(self):
        result = score([item(base=None, variants=(None, None, None, None),
                             trap="عايز", has_trap=True)])
        self.assertEqual(result["n_traps_scored"], 0)

    def test_items_without_traps_do_not_enter_trap_denominators(self):
        result = score([item()])
        self.assertEqual(result["n_trap_items"], 0)
        self.assertIsNone(result["over_norm_rate"])
        self.assertIsNone(result["trap_non_answer_rate"])


class Ceiling(unittest.TestCase):
    def test_repeat_agreement_is_the_stability_ceiling(self):
        stable = score([item(base_repeat="عايز", has_repeat=True)])
        drift = score([item(base_repeat="عاوز", has_repeat=True)])
        self.assertEqual(stable["ceiling_agreement"], 1.0)
        self.assertEqual(drift["ceiling_agreement"], 0.0)

    def test_double_non_answer_says_nothing_about_stability(self):
        result = score([item(base=None, base_repeat=None, has_repeat=True)])
        self.assertEqual(result["ceiling_n"], 0)
        self.assertIsNone(result["ceiling_agreement"])


class ByClass(unittest.TestCase):
    def test_keys_are_phenomena_constants(self):
        """The join key for the E1xE2 figure. Guard it."""
        result = score([item(variants=("عايز", "عاوز", None, "عايز"),
                             variant_classes=(("VOWEL_SWAP",), ("DIPHTHONG",),
                                              ("DEGEMINATION",), ("FINAL_Y_I",)))])
        known = {getattr(phenomena, name) for name in dir(phenomena)
                 if name.isupper() and isinstance(getattr(phenomena, name), str)}
        for key in result["by_class"]:
            self.assertIn(key, known)

    def test_flip_rate_counts_none_as_a_flip(self):
        result = score([item(variants=("عايز", None, "عايز", "عايز"),
                             variant_classes=(("VOWEL_SWAP",),) * 4)])
        cell = result["by_class"]["VOWEL_SWAP"]
        self.assertEqual(cell["n_edges"], 4)
        self.assertEqual(cell["flip_rate"], 0.25)
        self.assertEqual(cell["n_none"], 1)

    def test_a_compound_edge_counts_under_both_classes(self):
        result = score([item(variants=("عاوز",) + ("عايز",) * 3,
                             variant_classes=(("DIPHTHONG", "VOWEL_LENGTH"),
                                              ("VOWEL_SWAP",), ("VOWEL_SWAP",),
                                              ("VOWEL_SWAP",)))])
        self.assertEqual(result["by_class"]["DIPHTHONG"]["flip_rate"], 1.0)
        self.assertEqual(result["by_class"]["VOWEL_LENGTH"]["flip_rate"], 1.0)


class GoldAnchor(unittest.TestCase):
    def test_meaning_accuracy_only_where_gold_exists(self):
        no_gold = score([item()])
        self.assertEqual(no_gold["n_gold"], 0)
        self.assertIsNone(no_gold["meaning_accuracy"])

    def test_gold_scores_only_parsed_answers(self):
        result = score([item(variants=("عايز", "سيارة", None, "عايز"),
                             gold=frozenset({"عايز"}))])
        self.assertEqual(result["n_gold"], 4)          # base + 3 parsed variants
        self.assertEqual(result["meaning_accuracy"], 0.75)


class ModalMeaning(unittest.TestCase):
    def test_most_common_non_none_wins(self):
        # base عايز + one variant عايز = 2; عاوز = 2. Counter breaks the tie
        # by first-seen order, which is the base -- deterministic either way.
        meaning, count = modal_meaning(item(variants=("عاوز", "عاوز", "عايز", None)))
        self.assertEqual(meaning, "عايز")
        self.assertEqual(count, 2)

    def test_all_none_reports_none(self):
        meaning, count = modal_meaning(item(base=None, variants=(None,) * 4))
        self.assertIsNone(meaning)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
