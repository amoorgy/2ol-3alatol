import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.arabic_norm import first_arabic_word, normalize_arabic


class Folds(unittest.TestCase):
    def test_alef_family_folds_to_one_form(self):
        for variant in ("أهم", "إهم", "آهم", "اهم"):
            self.assertEqual(normalize_arabic(variant), "اهم")

    def test_alef_maksura_and_farsi_yeh_fold_to_yeh(self):
        self.assertEqual(normalize_arabic("عايزى"), normalize_arabic("عايزي"))
        self.assertEqual(normalize_arabic("مصطفی"), normalize_arabic("مصطفي"))

    def test_taa_marbuta_folds_to_heh(self):
        self.assertEqual(normalize_arabic("شقة"), normalize_arabic("شقه"))

    def test_the_fold_direction_never_invents_a_taa(self):
        """اه must stay اه; only ة -> ه, never ه -> ة."""
        self.assertEqual(normalize_arabic("اه"), "اه")

    def test_diacritics_and_shadda_are_stripped(self):
        self.assertEqual(normalize_arabic("جِدًّا"), "جدا")

    def test_tatweel_is_stripped(self):
        self.assertEqual(normalize_arabic("عــايز"), "عايز")

    def test_presentation_forms_fold_via_nfkc(self):
        self.assertEqual(normalize_arabic("ﻻ"), "لا")   # lam-alef ligature

    def test_hamza_carriers_fold(self):
        self.assertEqual(normalize_arabic("مسؤول"), "مسوول")
        self.assertEqual(normalize_arabic("شئ"), "شي")

    def test_latin_digits_and_punctuation_are_dropped(self):
        self.assertEqual(normalize_arabic('  "عايز" (car) 123 ٤٥٦ '), "عايز")

    def test_letter_run_collapse_is_off_by_default(self):
        """Merging elongation also merges genuinely doubled roots -- opt in."""
        self.assertEqual(normalize_arabic("مسوول"), "مسوول")
        self.assertEqual(normalize_arabic("مسوول", collapse_repeats=True), "مسول")

    def test_idempotent(self):
        for text in ("أهلاً وسهلاً", "شقة", "جِدًّا", "عــايز يعني"):
            once = normalize_arabic(text)
            self.assertEqual(normalize_arabic(once), once)

    def test_nothing_arabic_yields_none_not_empty(self):
        for text in ("", "   ", "hello", "3ayez", "123", '"()"'):
            with self.subTest(text=text):
                self.assertIsNone(normalize_arabic(text))


class MeaningExtraction(unittest.TestCase):
    def test_filler_openers_are_stripped(self):
        self.assertEqual(first_arabic_word("يعني عايز"), "عايز")
        self.assertEqual(first_arabic_word("معناها سيارة"), "سياره")

    def test_up_to_three_tokens_join(self):
        """Real meanings can be short phrases (اسم علم, على طول)."""
        self.assertEqual(first_arabic_word("على طول"), "علي طول")

    def test_a_bare_filler_word_is_an_answer_not_a_preamble(self):
        """A model glossing `howa` may answer exactly "هو"; the filler
        stripper must not eat an entire one-word answer."""
        self.assertEqual(first_arabic_word("هو"), "هو")
        self.assertEqual(first_arabic_word("هو عايز"), "عايز")

    def test_more_than_three_tokens_is_a_non_answer(self):
        """The prompt asked for one word; parsing a sentence manufactures data."""
        self.assertIsNone(first_arabic_word("الكلمة دي معناها حاجة كبيرة اوي"))

    def test_latin_only_answer_is_a_non_answer(self):
        self.assertIsNone(first_arabic_word("it means car"))

    def test_arabic_word_with_quotes_survives(self):
        self.assertEqual(first_arabic_word('"عايز"'), "عايز")


if __name__ == "__main__":
    unittest.main()
