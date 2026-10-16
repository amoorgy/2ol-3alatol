import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.script_id import arabic_ratio, is_arabic_answer


class ArabicRatio(unittest.TestCase):
    def test_pure_arabic_is_one(self):
        self.assertEqual(arabic_ratio("عايز"), 1.0)

    def test_pure_latin_is_zero(self):
        self.assertEqual(arabic_ratio("3ayez"), 0.0)

    def test_digits_and_punctuation_sit_outside_the_denominator(self):
        """An Arabic meaning with a full stop or smiley is still fully Arabic."""
        self.assertEqual(arabic_ratio("عايز."), 1.0)
        self.assertEqual(arabic_ratio("عايز (:"), 1.0)
        self.assertEqual(arabic_ratio("عايز 123"), 1.0)

    def test_mixed_scripts_are_proportional(self):
        self.assertAlmostEqual(arabic_ratio("عايز car"), 4 / 7)

    def test_no_letters_at_all_is_zero(self):
        self.assertEqual(arabic_ratio("123 !?"), 0.0)
        self.assertEqual(arabic_ratio(""), 0.0)


class Gate(unittest.TestCase):
    def test_both_sides_of_the_threshold(self):
        self.assertTrue(is_arabic_answer("عايز"))
        self.assertFalse(is_arabic_answer("3ayez"))
        self.assertFalse(is_arabic_answer("عايز means want", threshold=0.8))
        self.assertTrue(is_arabic_answer("عايز x", threshold=0.5))

    def test_empty_answer_fails_the_gate(self):
        self.assertFalse(is_arabic_answer(""))


if __name__ == "__main__":
    unittest.main()
