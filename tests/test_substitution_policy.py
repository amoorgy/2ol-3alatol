import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.licensing import Variant
from franco_eval.substitution_policy import (Policy, check, needs_judge,
                                             parse_bad_substitutions)


def v(form: str, *ops: str) -> Variant:
    return Variant(form=form, path=tuple((op, 0) for op in ops))


POLICY = Policy(
    trap_partners={"men": frozenset({"meen"}), "meen": frozenset({"men"})},
    bad_pairs=frozenset({("7adid", "7aadd")}),
    protected_forms=frozenset({"3ayz"}),
    english=frozenset({"offline"}),
)


class PairLevelBlocks(unittest.TestCase):
    def test_known_trap_partner_is_never_a_respelling(self):
        self.assertIn("trap", check(POLICY, "men", v("meen", "LENGTHEN"),
                                    sentence_final=False))

    def test_council_named_pair_is_blocked(self):
        self.assertIn("council", check(POLICY, "7adid", v("7aadd", "SHORTEN"),
                                       sentence_final=False))

    def test_held_out_stimulus_form_is_never_created(self):
        self.assertIn("held-out", check(POLICY, "3ayez", v("3ayz", "DROP_MEDIAL_VOWEL"),
                                        sentence_final=False))

    def test_english_token_is_never_respelled(self):
        self.assertIn("English", check(POLICY, "offline", v("ufflene", "O_TO_U"),
                                       sentence_final=False))

    def test_an_ordinary_swap_passes(self):
        self.assertEqual(check(POLICY, "ketir", v("kitir", "E_TO_I"),
                               sentence_final=False), "")


class ReviewRulings(unittest.TestCase):
    """The a_v1 review's rulings, turned from row verdicts into code."""

    def test_elongation_is_allowed_sentence_finally(self):
        self.assertEqual(check(POLICY, "aktar", v("aktaaar", "LENGTHEN"),
                               sentence_final=True), "")

    def test_elongation_is_denied_mid_sentence(self):
        self.assertIn("elongation", check(POLICY, "aktar", v("aktaaar", "LENGTHEN"),
                                          sentence_final=False))

    def test_word_already_elongated_is_not_penalised(self):
        self.assertEqual(check(POLICY, "yaaah", v("yaaa", "AH_TO_A"),
                               sentence_final=False), "")

    def test_7_to_h_after_t_fabricates_a_th_digraph(self):
        for word, form in [("mot7araka", "motharaka"), ("bt7tag", "bthtag"),
                           ("byt7awl", "bythawl")]:
            with self.subTest(word=word):
                self.assertIn("th", check(POLICY, word, v(form, "7_TO_H"),
                                          sentence_final=False))

    def test_7_to_h_elsewhere_passes(self):
        self.assertEqual(check(POLICY, "7adid", v("hadid", "7_TO_H"),
                               sentence_final=False), "")

    def test_existing_th_is_not_blamed_on_7_to_h(self):
        self.assertEqual(check(POLICY, "ta2ther7", v("ta2therh", "7_TO_H"),
                               sentence_final=False), "")

    def test_fabricated_long_vowel_onset_is_denied(self):
        self.assertIn("onset", check(POLICY, "elwa2t", v("eelwa2t", "LENGTHEN"),
                                     sentence_final=False))


def at(form: str, op: str, position: int) -> Variant:
    return Variant(form=form, path=((op, position),))


class NativeClassRulings(unittest.TestCase):
    """Native ruling 2026-09-25: ع-deletion is word-initial only; medial
    vowel drop is allowed."""

    def test_word_initial_3_to_a_is_allowed(self):
        for word, form in [("3amr", "amr"), ("3and", "aand"), ("3elm", "aelm")]:
            with self.subTest(word=word):
                self.assertEqual(check(POLICY, word, at(form, "3_TO_A", 0),
                                       sentence_final=False), "")

    def test_3_to_a_inside_the_word_is_denied(self):
        for word, form, pos in [("n3m", "nam", 1), ("sam3t", "samat", 3),
                                ("el3ard", "elaard", 2), ("rab3", "raba", 3)]:
            with self.subTest(word=word):
                self.assertIn("word-initial", check(POLICY, word, at(form, "3_TO_A", pos),
                                                    sentence_final=False))

    def test_medial_vowel_drop_is_allowed(self):
        for word, form in [("leha", "lha"), ("3alameya", "3alamya")]:
            with self.subTest(word=word):
                self.assertEqual(check(POLICY, word, at(form, "DROP_MEDIAL_VOWEL", 1),
                                       sentence_final=False), "")


class NeedsJudge(unittest.TestCase):
    """Native ruling 2026-09-25: these read fine in context, so they are not
    blocked -- but no rule can prove them, so each pair goes to a judge."""

    def test_every_word_initial_3_to_a_needs_a_judge(self):
        self.assertTrue(needs_judge("3arf", at("arf", "3_TO_A", 0)))
        self.assertTrue(needs_judge("3nd", at("and", "3_TO_A", 0)))

    def test_vowel_dropped_beside_a_vowel_needs_a_judge(self):
        for word, form, pos in [("radio", "rado", 3), ("3amalia", "3amala", 5)]:
            with self.subTest(word=word):
                self.assertTrue(needs_judge(word, at(form, "DROP_MEDIAL_VOWEL", pos)))

    def test_vowel_dropped_between_consonants_is_trusted(self):
        self.assertEqual(needs_judge("leha", at("lha", "DROP_MEDIAL_VOWEL", 1)), "")
        self.assertEqual(needs_judge("3alameya", at("3alamya", "DROP_MEDIAL_VOWEL", 5)), "")

    def test_ordinary_swap_is_trusted(self):
        self.assertEqual(needs_judge("ketir", at("kitir", "E_TO_I", 1)), "")

    def test_position_is_read_on_the_intermediate_form(self):
        """A later op's position indexes the word AFTER earlier ops ran."""
        variant = Variant(form="3lameya", path=(("SHORTEN", 99), ("DROP_MEDIAL_VOWEL", 1)))
        # unreplayable first step: flag conservatively rather than guess
        self.assertTrue(needs_judge("3alameya", variant))


class BadSubstitutionParsing(unittest.TestCase):
    def test_parses_multi_pair_cells_with_counts(self):
        self.assertEqual(parse_bad_substitutions("gedida->gidda:5 farz->faaarz:2"),
                         [("gedida", "gidda", 5), ("farz", "faaarz", 2)])

    def test_empty_cell(self):
        self.assertEqual(parse_bad_substitutions(""), [])


if __name__ == "__main__":
    unittest.main()
