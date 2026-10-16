import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco import skeleton
from franco.corpus import FEATURES, features, is_clean, sentences, tokens, word_strings
from franco_eval import phenomena


class Tokenising(unittest.TestCase):
    """The corpus is machine-transliterated, and it is dirty in two known ways."""

    def test_prime_digraph_tokens_are_rejected(self):
        """skeleton.words() splits et3'ayar into et3 + ayar; both are junk."""
        self.assertEqual(word_strings("et3'ayar kan"), ["kan"])
        self.assertEqual(word_strings("e3'lam we 3'eer"), ["we"])
        self.assertIn("et3", skeleton.words("et3'ayar kan"))  # the bug being avoided

    def test_typographic_apostrophe_is_rejected_too(self):
        self.assertEqual(word_strings("et3’ayar kan"), ["kan"])

    def test_arabic_script_adjacent_tokens_are_rejected(self):
        self.assertEqual(word_strings("felدرس dah"), ["dah"])

    def test_a_contaminated_run_is_dropped_whole_not_trimmed(self):
        """Trimming would manufacture a shorter word that nobody wrote."""
        self.assertEqual(word_strings("wa7'eda"), [])

    def test_clean_text_is_unaffected(self):
        self.assertEqual(word_strings("dah 7aga kwayesa"), ["dah", "7aga", "kwayesa"])

    def test_offsets_point_at_the_original_text(self):
        text = "dah 7aga"
        for word, offset, _cap in tokens(text):
            self.assertEqual(text[offset:offset + len(word)].lower(), word)


class Capitalisation(unittest.TestCase):
    """cap_ratio is the proper-noun filter, so sentence-initial caps must not count."""

    def test_line_initial_capital_is_not_reported(self):
        self.assertEqual(tokens("Masr kwayesa")[0], ("masr", 0, False))

    def test_capital_after_terminal_punctuation_is_not_reported(self):
        self.assertFalse(tokens("dah kan. Masr kwayesa")[-2][2])

    def test_mid_sentence_capital_is_reported(self):
        self.assertTrue(dict((t, c) for t, _o, c in tokens("dah kan fe America"))["america"])

    def test_capital_after_an_opening_quote_is_not_reported(self):
        self.assertFalse(tokens('"Masr kwayesa"')[0][2])


class Sentences(unittest.TestCase):
    def test_line_breaks_split_even_without_punctuation(self):
        got = [s for _li, _si, s in sentences("- awel haga\n- tany haga")]
        self.assertEqual(got, ["- awel haga", "- tany haga"])

    def test_terminal_punctuation_is_kept(self):
        got = [s for _li, _si, s in sentences("dah kan. w dah kaman!")]
        self.assertEqual(got, ["dah kan.", "w dah kaman!"])

    def test_blank_segments_are_skipped(self):
        self.assertEqual(list(sentences("\n\n  \n")), [])

    def test_is_clean_rejects_the_three_contaminants(self):
        self.assertTrue(is_clean("dah 7aga kwayesa"))
        self.assertFalse(is_clean("felدرس dah"))
        self.assertFalse(is_clean("et3'ayar"))
        self.assertFalse(is_clean("shoof http://x.com"))


class Features(unittest.TestCase):
    """Features drive the stratified 500-word selection."""

    def test_diphthong_feature_matches_the_phenomena_step(self):
        """The coupling point between the generator and the classifier.

        src/franco/ may not import franco_eval, so nothing but this assertion
        stops the two definitions of "diphthong" from drifting apart.
        """
        expected = f"{phenomena._DIPH_FRONT.pattern}|{phenomena._DIPH_BACK.pattern}"
        self.assertEqual(FEATURES["diph"].pattern, expected)

    def test_excluded_digits_are_not_a_feature(self):
        for digit in phenomena.EXCLUDED_DIGITS:
            self.assertNotIn(digit, FEATURES["digit"].pattern)

    def test_each_feature_fires_on_its_witness(self):
        for name, word in (("digit", "3ashan"), ("diph", "beit"), ("vlong", "aayez"),
                           ("gem", "elly"), ("digraph", "mesh"), ("finalyi", "elli")):
            with self.subTest(feature=name):
                self.assertIn(name, features(word))

    def test_a_featureless_word_is_reported_empty(self):
        self.assertEqual(features("bnt"), ())

    def test_features_are_returned_in_table_order(self):
        got = features("3ashan")
        self.assertEqual(list(got), [n for n in FEATURES if n in got])


if __name__ == "__main__":
    unittest.main()
