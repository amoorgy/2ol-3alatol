import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco.normalize import normalize
from franco.perturb import perturb


class Canon(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(normalize("KHALAS"), "5alas")
        self.assertEqual(normalize("ezzayyak"), "ezayak")
        self.assertEqual(normalize("mich"), "mesh")
        self.assertEqual(normalize("qalb"), "2alb")
        self.assertEqual(normalize("gamedah"), "gameda")
        self.assertEqual(normalize("gedddaaan"), "gedan")

    def test_idempotent(self):
        rng = random.Random(2)
        base = "ezayak ya 3am, elhamdullellah kwayes we enta 3amel eh"
        for _ in range(100):
            text = perturb(base, 0.8, rng)
            once = normalize(text)
            self.assertEqual(normalize(once), once, text)

    def test_collapses_variant_pairs(self):
        pairs = [("khalas", "5alas"), ("ezayak", "ezzayak"), ("shokran", "chukran")]
        for a, b in pairs:
            self.assertEqual(normalize(a), normalize(b), (a, b))


if __name__ == "__main__":
    unittest.main()
