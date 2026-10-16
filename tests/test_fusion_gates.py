import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.fusion import (Anchor, check_sentence, contribution,
                                fusion_prompt, teacher_prompt)

ANCHOR = Anchor("2ahwa", arabic="قهوة", gloss_en="coffee")
GOOD = "kol yom el sob7 bashrab 2ahwa so5na 3ala el balakona"


class Gates(unittest.TestCase):
    def test_a_clean_sentence_passes(self):
        self.assertTrue(check_sentence(GOOD, ANCHOR).ok)

    def test_non_egyptian_digit_letters_fail(self):
        """6 (ط) and 9 (ص) are EXCLUDED_DIGITS and 4 = ش is the Gulf/Levantine
        habit licensing denies (SH_TO_4); glm-5.3 wrote all three in v2."""
        for word in ("maba5las4", "fe6or", "9a7by"):
            with self.subTest(word=word):
                report = check_sentence(f"kol yom el sob7 {word} bashrab 2ahwa so5na", ANCHOR)
                self.assertFalse(report.ok)
                self.assertTrue(any("non-Egyptian digit" in r for r in report.reasons))

    def test_numerals_are_not_digit_letters(self):
        self.assertTrue(check_sentence(
            "kol yom el sob7 3ala el sa3a 6 bashrab 2ahwa so5na 24 sa3a", ANCHOR).ok)

    def test_missing_anchor_is_a_byte_identity_failure(self):
        report = check_sentence(
            "kol yom bashrab shai so5n 3ala el balakona beta3etna", ANCHOR)
        self.assertFalse(report.ok)
        self.assertTrue(any("byte-identity" in r for r in report.reasons))

    def test_a_respelled_anchor_fails_the_gate(self):
        """The fusor 'improving' 2ahwa to ahwa is exactly what must not pass:
        orthography belongs to the rules, never to the fusor."""
        report = check_sentence(
            "kol yom el sob7 bashrab ahwa so5na 3ala el balakona", ANCHOR)
        self.assertFalse(report.ok)

    def test_arabic_script_drift_is_rejected(self):
        report = check_sentence(
            "kol yom bashrab قهوة so5na 3ala el balakona", ANCHOR)
        self.assertFalse(report.ok)
        self.assertTrue(any("Arabic script" in r for r in report.reasons))

    def test_duplicated_anchor_is_rejected(self):
        report = check_sentence(
            "2ahwa 2ahwa bashrab kol yom el sob7 3ala el balakona", ANCHOR)
        self.assertFalse(report.ok)

    def test_length_bounds_are_enforced(self):
        self.assertFalse(check_sentence("bashrab 2ahwa", ANCHOR).ok)

    def test_all_reasons_are_reported_not_just_the_first(self):
        report = check_sentence("قهوة", ANCHOR)
        self.assertGreaterEqual(len(report.reasons), 2)

    def test_latin_language_drift_is_rejected_via_ascii(self):
        """Observed live: a fusor answered in French. Franco is ASCII by
        construction, so one accented character is a sufficient tell."""
        report = check_sentence(
            "bashrab 2ahwa présenter des informations kol yom aho", ANCHOR)
        self.assertFalse(report.ok)
        self.assertTrue(any("non-ASCII" in r for r in report.reasons))

    def test_english_function_words_are_rejected(self):
        report = check_sentence(
            "bashrab 2ahwa because this is the word we want here aho", ANCHOR)
        self.assertFalse(report.ok)
        self.assertTrue(any("English drift" in r for r in report.reasons))

    def test_prompt_scaffolding_echo_is_rejected(self):
        """Observed live: the fusor copied 'el 7a2ay2 elly mamno3 tetghayar'
        from our own instruction into the data."""
        report = check_sentence(
            "ba7eb 2ahwa bas el 7a2ay2 elly mamno3 tetghayar mohemma", ANCHOR)
        self.assertFalse(report.ok)

    def test_sentence_initial_anchor_is_rejected(self):
        report = check_sentence(
            "2ahwa keteer bashrabha kol yom el sob7 3ala el balakona", ANCHOR)
        self.assertFalse(report.ok)

    def test_first_word_merely_prefixed_by_the_anchor_is_not_initial(self):
        """`2ahwaha` is a different word, not the anchor in first position.

        The check compares tokens, never a string prefix: Egyptian clitics
        (-ha -ak -ni -lo) make a stem a prefix of its own inflected forms,
        so startswith() here silently discarded valid sentences.
        """
        report = check_sentence(
            "2ahwaha kanet sa2a3 bas ana shrebt 2ahwa so5na fel maktab",
            ANCHOR)
        self.assertTrue(report.ok, report.reasons)


class Contribution(unittest.TestCase):
    def test_verbatim_copy_attributes_fully_to_its_draft(self):
        shares = contribution([GOOD, "7aga tanya 5ales"], GOOD)
        self.assertGreater(shares["draft_0"], 0.95)
        self.assertLessEqual(shares["novel"], 0.05)

    def test_novel_text_is_reported_as_novel(self):
        shares = contribution(["kalam mo5talef tamaman aho"],
                              "gomla gedida ma3andahash ay 3elaqa")
        self.assertGreater(shares["novel"], 0.7)

    def test_novel_is_a_fraction(self):
        shares = contribution([GOOD], GOOD[:20] + " kalam gedid")
        self.assertGreaterEqual(shares["novel"], 0.0)
        self.assertLessEqual(shares["novel"], 1.0)


SCENARIO = "you're describing something that happened yesterday"


class Prompts(unittest.TestCase):
    def test_prompts_carry_the_anchor_and_its_arabic(self):
        for prompt in (fusion_prompt(ANCHOR, ["a b c d e"], SCENARIO),
                       teacher_prompt(ANCHOR, SCENARIO)):
            self.assertIn("2ahwa", prompt)
            self.assertIn("قهوة", prompt)

    def test_prompts_carry_the_scenario(self):
        """The whole point of threading scenario through: teachers and the
        fusor need something to write ABOUT, not just a word to place."""
        for prompt in (fusion_prompt(ANCHOR, ["a b c d e"], SCENARIO),
                       teacher_prompt(ANCHOR, SCENARIO)):
            self.assertIn(SCENARIO, prompt)

    def test_fusion_prompt_numbers_every_draft(self):
        prompt = fusion_prompt(ANCHOR, ["gomla wa7da aho", "gomla tanya aho"],
                               SCENARIO)
        self.assertIn("1. gomla wa7da aho", prompt)
        self.assertIn("2. gomla tanya aho", prompt)


if __name__ == "__main__":
    unittest.main()
