import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.word_split import HELDOUT, TRAIN, components, split_words


class Components(unittest.TestCase):
    def test_a_word_and_its_trap_form_one_component(self):
        comps = components(["men", "ana"], [("men", "meen")])
        self.assertIn(frozenset({"men", "meen"}), comps)
        self.assertIn(frozenset({"ana"}), comps)

    def test_a_trap_shared_by_two_words_joins_all_three(self):
        comps = components(["a", "b"], [("a", "t"), ("b", "t")])
        self.assertEqual(comps, [frozenset({"a", "b", "t"})])

    def test_every_word_lands_in_exactly_one_component(self):
        words = ["a", "b", "c", "d"]
        comps = components(words, [("a", "x"), ("c", "d")])
        members = [w for comp in comps for w in comp]
        self.assertEqual(sorted(members), sorted(set(members)))
        self.assertEqual(set(members), {"a", "b", "c", "d", "x"})


class SplitWords(unittest.TestCase):
    WORDS = [f"w{i}" for i in range(400)]
    PAIRS = [(f"w{i}", f"t{i}") for i in range(0, 400, 3)]

    def test_pair_never_straddles_the_split(self):
        sides = split_words(self.WORDS, self.PAIRS, salt="s")
        for word, trap in self.PAIRS:
            self.assertEqual(sides[word], sides[trap])

    def test_deterministic_and_order_independent(self):
        a = split_words(self.WORDS, self.PAIRS, salt="s")
        b = split_words(list(reversed(self.WORDS)),
                        list(reversed(self.PAIRS)), salt="s")
        self.assertEqual(a, b)

    def test_salt_changes_the_split(self):
        a = split_words(self.WORDS, self.PAIRS, salt="s1")
        b = split_words(self.WORDS, self.PAIRS, salt="s2")
        self.assertNotEqual(a, b)

    def test_fraction_is_roughly_honoured(self):
        sides = split_words(self.WORDS, self.PAIRS, salt="s")
        share = sum(1 for w in self.WORDS if sides[w] == HELDOUT) / len(self.WORDS)
        self.assertTrue(0.4 < share < 0.6, share)

    def test_forced_words_pull_their_whole_component_to_heldout(self):
        sides = split_words(self.WORDS, self.PAIRS, salt="s",
                            forced_heldout=frozenset({"t0", "w1"}))
        self.assertEqual(sides["w0"], HELDOUT)   # via its trap t0
        self.assertEqual(sides["t0"], HELDOUT)
        self.assertEqual(sides["w1"], HELDOUT)

    def test_forced_words_outside_the_benchmark_are_still_listed(self):
        sides = split_words(["a"], [], salt="s",
                            forced_heldout=frozenset({"human"}))
        self.assertEqual(sides["human"], HELDOUT)
        self.assertIn(sides["a"], (TRAIN, HELDOUT))


if __name__ == "__main__":
    unittest.main()
