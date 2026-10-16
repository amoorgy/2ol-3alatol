import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco.skeleton import skeleton, words


class SkeletonEquivalence(unittest.TestCase):
    """Attested spelling variants must share one fingerprint."""

    CLUSTERS = [
        ("a7a", "aha"),
        ("2alb", "alb", "qalb"),
        ("khalas", "5alas"),
        ("gedan", "geddan", "gedaan"),
        ("ezayak", "ezzayak", "izayak", "ezayyak"),
        ("mesh", "msh", "mish", "mch"),
        ("3ayez", "3aiz", "3ayz"),
        ("tamam", "tmam", "tamaam"),
        ("ghali", "8ali"),
        ("sa7", "sah", "saa7"),
        ("enty", "enti"),
        ("habibi", "habiby"),
    ]

    def test_variants_collide(self):
        for cluster in self.CLUSTERS:
            keys = {skeleton(w) for w in cluster}
            self.assertEqual(len(keys), 1, f"{cluster} -> {keys}")

    def test_distinct_words_do_not_collide(self):
        self.assertNotEqual(skeleton("khalas"), skeleton("kalam"))
        self.assertNotEqual(skeleton("gamed"), skeleton("kwayes"))

    def test_inflection_is_not_a_spelling_variant(self):
        # Final -y/-i is a morpheme (2fs/possessive): enta and enty are
        # different words and must NOT share a fingerprint.
        self.assertNotEqual(skeleton("enta"), skeleton("enty"))
        self.assertNotEqual(skeleton("masr"), skeleton("masry"))

    def test_word_medial_glottal_survives(self):
        # Only a WORD-INITIAL glottal stop is optional in writing; the ق/ء
        # inside "a2ollak" is a root consonant and must stay.
        self.assertIn("q", skeleton("a2ollak"))
        self.assertEqual(skeleton("2alb"), skeleton("alb"))

    def test_vowel_only_words_survive(self):
        self.assertTrue(skeleton("eeeh"))
        self.assertEqual(skeleton("aiwa"), skeleton("aywa"))

    def test_empty_input(self):
        self.assertEqual(skeleton(""), "")


class Words(unittest.TestCase):
    def test_tokenizes_digits_inside_words(self):
        self.assertEqual(words("ezayak ya 3am!"), ["ezayak", "ya", "3am"])


if __name__ == "__main__":
    unittest.main()
