import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.respell import (Applied, Span, eligible_spans, respell,
                                 sample_rendering)

SENTENCE = "dah kan 3ashan el mawdoo3 mesh wade7"
SPANS = (
    Span("3ashan", 8, 14),
    Span("mawdoo3", 18, 25),
    Span("mesh", 26, 30),
)


class Respelling(unittest.TestCase):
    def test_identity_choices_reproduce_the_input_byte_identically(self):
        text, applied = respell(SENTENCE, SPANS, {})
        self.assertEqual(text, SENTENCE)
        for a in applied:
            self.assertTrue(a.is_base)
            self.assertEqual(text[a.start:a.end], a.rendering)

    def test_every_applied_rendering_sits_at_its_recorded_span(self):
        """The multi-span invariant. Lengths change, offsets must follow."""
        choices = {"3ashan": "3shan", "mawdoo3": "elmawdooo3", "mesh": "mish"}
        text, applied = respell(SENTENCE, SPANS, choices)
        for a in applied:
            with self.subTest(word=a.word):
                self.assertEqual(text[a.start:a.end], a.rendering)

    def test_earlier_substitution_shifts_do_not_corrupt_later_spans(self):
        """Regression case for the right-to-left rule: a LONGER first
        rendering must not displace where the later ones land."""
        choices = {"3ashan": "3alashaaan", "mesh": "msh"}
        text, applied = respell(SENTENCE, SPANS, choices)
        self.assertIn("3alashaaan", text)
        by_word = {a.word: a for a in applied}
        self.assertEqual(text[by_word["mesh"].start:by_word["mesh"].end], "msh")
        self.assertEqual(
            text[by_word["mawdoo3"].start:by_word["mawdoo3"].end], "mawdoo3")

    def test_unknown_words_in_choices_are_ignored(self):
        text, _ = respell(SENTENCE, SPANS, {"kalb": "kelb"})
        self.assertEqual(text, SENTENCE)

    def test_span_text_mismatch_is_an_error_not_a_silent_corruption(self):
        with self.assertRaises(ValueError):
            respell(SENTENCE, (Span("3ashan", 0, 6),), {})

    def test_overlapping_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            respell(SENTENCE, (Span("3ashan", 8, 14), Span("an", 12, 14)), {})


class Sampling(unittest.TestCase):
    VARIANTS = {
        "3ashan": [("3shan", 3.0), ("3ashaan", 1.0)],
        "mesh": [("mish", 2.0), ("msh", 1.0)],
    }

    def test_same_seed_gives_byte_identical_choices(self):
        first = sample_rendering(SPANS, self.VARIANTS, random.Random(13))
        second = sample_rendering(SPANS, self.VARIANTS, random.Random(13))
        self.assertEqual(first, second)

    def test_different_seeds_eventually_differ(self):
        outcomes = {tuple(sorted(sample_rendering(
            SPANS, self.VARIANTS, random.Random(seed)).items()))
            for seed in range(30)}
        self.assertGreater(len(outcomes), 1)

    def test_blocked_words_are_never_respelled(self):
        """The contamination firewall: eval words keep base spelling always."""
        for seed in range(50):
            choices = sample_rendering(
                SPANS, self.VARIANTS, random.Random(seed), p_respell=1.0,
                blocked=frozenset({"3ashan"}))
            self.assertNotIn("3ashan", choices)

    def test_p_respell_zero_keeps_everything_base(self):
        choices = sample_rendering(SPANS, self.VARIANTS, random.Random(13),
                                   p_respell=0.0)
        self.assertEqual(choices, {})

    def test_words_without_variants_stay_base(self):
        choices = sample_rendering(SPANS, {"mesh": [("mish", 1.0)]},
                                   random.Random(13), p_respell=1.0)
        self.assertEqual(set(choices), {"mesh"})

    def test_weights_bias_the_draw(self):
        heavy = sum(
            sample_rendering(SPANS, {"mesh": [("mish", 50.0), ("msh", 1.0)]},
                             random.Random(seed), p_respell=1.0).get("mesh") == "mish"
            for seed in range(200))
        self.assertGreater(heavy, 150)


class Eligibility(unittest.TestCase):
    def test_eligible_spans_respect_the_firewall_and_variant_coverage(self):
        variants = {"3ashan": [("3shan", 1.0)], "mesh": [("mish", 1.0)]}
        spans = eligible_spans(SPANS, variants,
                               blocked=frozenset({"3ashan"}))
        self.assertEqual([s.word for s in spans], ["mesh"])


if __name__ == "__main__":
    unittest.main()
