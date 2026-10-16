import pathlib
import random
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco.perturb import perturb, perturb_word, variants
from franco.skeleton import skeleton

SAMPLE = "ezayak ya 3am, elhamdullellah kwayes we enta 3amel eh"


class Determinism(unittest.TestCase):
    def test_same_seed_same_output(self):
        a = perturb(SAMPLE, 0.5, random.Random(7))
        b = perturb(SAMPLE, 0.5, random.Random(7))
        self.assertEqual(a, b)

    def test_zero_intensity_is_identity(self):
        self.assertEqual(perturb(SAMPLE, 0.0, random.Random(7)), SAMPLE)


class Plausibility(unittest.TestCase):
    def test_output_alphabet_stays_franco(self):
        rng = random.Random(3)
        for _ in range(50):
            out = perturb(SAMPLE, 1.0, rng)
            self.assertTrue(re.fullmatch(r"[a-z0-9 ,]+", out), out)

    def test_most_single_rewrites_preserve_skeleton(self):
        """Taxonomy rules are spelling habits, not word changes: a clear
        majority of single-word rewrites must keep the phonemic skeleton
        (not all: medial vowel deletion can legitimately shift it) — AND
        must actually rewrite (an identity perturber must fail this test)."""
        rng = random.Random(11)
        w = "elhamdullellah"
        outputs = [perturb_word(w, rng) for _ in range(200)]
        same = sum(skeleton(o) == skeleton(w) for o in outputs)
        changed = sum(o != w for o in outputs)
        self.assertGreater(same / 200, 0.6)
        self.assertGreater(changed / 200, 0.8)

    def test_uppercase_words_are_perturbed_too(self):
        rng = random.Random(4)
        outputs = {perturb_word("KHALAS", rng) for _ in range(30)}
        self.assertTrue(any(o != "khalas" for o in outputs), outputs)


class Variants(unittest.TestCase):
    def test_distinct_and_never_original(self):
        vs = variants(SAMPLE, 3, 0.4, random.Random(5))
        self.assertEqual(len(vs), 3)
        self.assertEqual(len(set(vs)), 3)
        self.assertNotIn(SAMPLE, vs)


if __name__ == "__main__":
    unittest.main()
