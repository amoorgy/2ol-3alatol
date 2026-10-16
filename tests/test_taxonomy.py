import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco.taxonomy import COMPILED, RULES
from franco_eval.phenomena import DIPHTHONG, FINAL_Y_I, classify_edge


class RuleTable(unittest.TestCase):
    """Structural invariants of the rule table itself."""

    def test_every_rule_pattern_compiles(self):
        for rule in RULES:
            with self.subTest(pattern=rule.pattern):
                re.compile(rule.pattern)

    def test_every_rule_weight_is_positive(self):
        for rule in RULES:
            with self.subTest(pattern=rule.pattern):
                self.assertGreater(rule.weight, 0.0)

    def test_no_duplicate_rewrites(self):
        """Two rules with the same pattern AND replacement double its weight."""
        seen = [(r.pattern, r.replacement) for r in RULES]
        self.assertEqual(len(seen), len(set(seen)))

    def test_compiled_mirrors_rules(self):
        self.assertEqual(len(COMPILED), len(RULES))
        for (pattern, rule), source in zip(COMPILED, RULES):
            self.assertEqual(pattern.pattern, source.pattern)
            self.assertIs(rule, source)


class FinalYIAnchoring(unittest.TestCase):
    """Regression guard for a bug that shipped unnoticed.

    Both y/i rules name the FINAL_Y_I phenomenon, but were written unanchored,
    so they fired mid-word: ya3ny -> ia3ny, which is not a habit anyone has.
    Egyptians alternate the *final* y/i (enty/enti, ya3ny/ya3ni).
    """

    def _rules_for(self, replacement):
        return [(p, r) for p, r in COMPILED
                if r.replacement == replacement and r.phenomenon.startswith("final")]

    def test_final_y_rules_are_anchored(self):
        for pattern, rule in self._rules_for("i") + self._rules_for("y"):
            with self.subTest(pattern=rule.pattern):
                self.assertIn(r"\b", rule.pattern, "y/i alternation must be word-final")

    def test_mid_word_y_is_left_alone(self):
        """ya3ny has two y's; only the one at index 4 may be rewritten."""
        for pattern, rule in self._rules_for("i"):
            with self.subTest(pattern=rule.pattern):
                match = pattern.search("ya3ny")
                self.assertIsNotNone(match)
                self.assertEqual(match.start(), 4)

    def test_final_alternation_still_fires_both_ways(self):
        for word, replacement, expected in (("ya3ny", "i", "ya3ni"),
                                            ("ya3ni", "y", "ya3ny"),
                                            ("enty", "i", "enti")):
            with self.subTest(word=word):
                pattern, rule = self._rules_for(replacement)[0]
                match = pattern.search(word)
                self.assertIsNotNone(match, f"{rule.pattern} did not fire on {word}")
                got = word[:match.start()] + match.expand(rule.replacement) + word[match.end():]
                self.assertEqual(got, expected)
                self.assertIn(FINAL_Y_I, classify_edge(word, got).classes)


class DiphthongRules(unittest.TestCase):
    """The generator and the classifier must agree about what a diphthong is.

    taxonomy.py produces variants; phenomena.py explains them. If the two drift
    apart, every per-phenomenon number downstream is silently wrong -- so each
    new rule is asserted to round-trip into a DIPHTHONG edge.
    """

    PROBES = {
        "ay(?![aeiou])": "zay",
        "ei(?![aeiou])": "3ein",
        "ee": "beet",
        "aw(?![aeiou])": "lawn",
        "ou(?![aeiou])": "youm",
    }

    def test_diphthong_rules_produce_diphthong_edges(self):
        fired = 0
        for pattern, rule in COMPILED:
            probe = self.PROBES.get(rule.pattern)
            if probe is None or "diphthong" not in rule.phenomenon.lower() \
                    and "monophthong" not in rule.phenomenon.lower():
                continue
            match = pattern.search(probe)
            self.assertIsNotNone(match, f"{rule.pattern} did not fire on {probe}")
            got = probe[:match.start()] + match.expand(rule.replacement) + probe[match.end():]
            with self.subTest(rewrite=f"{probe}->{got}"):
                self.assertIn(DIPHTHONG, classify_edge(probe, got).classes)
            fired += 1
        self.assertEqual(fired, 6, "expected all six diphthong rules to be probed")

    def test_onset_glides_are_not_rewritten(self):
        """tayeb keeps its y: the lookahead must reject a following vowel."""
        for pattern, rule in COMPILED:
            if rule.pattern.startswith(("ay(", "aw(")):
                with self.subTest(pattern=rule.pattern):
                    self.assertIsNone(pattern.search("tayeb"))


if __name__ == "__main__":
    unittest.main()
