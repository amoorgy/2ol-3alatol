import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.diversity import pairwise_distance, select_diverse

WORDS = ["a", "b", "c", "d"]
BASE = {}


class PairwiseDistance(unittest.TestCase):
    def test_identical_choices_are_zero_apart(self):
        self.assertEqual(pairwise_distance({"a": "a2"}, {"a": "a2"}, WORDS), 0)

    def test_missing_key_means_base_spelling(self):
        # {} and {"a": "a"} both mean "word a kept its base spelling"
        self.assertEqual(pairwise_distance({}, {"a": "a"}, WORDS), 0)

    def test_counts_each_disagreeing_word_once(self):
        a = {"a": "a2", "b": "b2"}
        b = {"a": "a2", "c": "c2"}
        # disagree on b (b2 vs base) and c (base vs c2); agree on a and d
        self.assertEqual(pairwise_distance(a, b, WORDS), 2)


class SelectDiverse(unittest.TestCase):
    def test_near_copies_are_excluded_even_to_fill_quota(self):
        near_copy = {"a": "a2"}          # distance 1 from base
        eligible = {"a": "a2", "b": "b2"}  # distance 2 from base
        picked = select_diverse([BASE], [near_copy, eligible], WORDS, k=5,
                                min_distance_from_base=2)
        self.assertEqual(picked, [1])

    def test_greedy_maximin_selection_worked_example(self):
        c0 = {"a": "a2"}                          # dist(base)=1 -> excluded
        c1 = {"a": "a2", "b": "b2"}                # dist(base)=2
        c2 = {"c": "c2", "d": "d2"}                # dist(base)=2
        c3 = {"a": "a2", "b": "b2", "c": "c2"}     # dist(base)=3
        picked = select_diverse([BASE], [c0, c1, c2, c3], WORDS, k=2,
                                min_distance_from_base=2)
        # c3 is farthest from base, picked first; c2 is farthest from c3
        # (distance 3) vs c1 (distance 1 from c3), so c2 is picked second.
        self.assertEqual(picked, [3, 2])

    def test_stops_at_k_even_with_more_eligible_candidates(self):
        candidates = [{"a": "a2", "b": "b2"}, {"c": "c2", "d": "d2"},
                      {"a": "a2", "c": "c2"}]
        picked = select_diverse([BASE], candidates, WORDS, k=1,
                                min_distance_from_base=2)
        self.assertEqual(len(picked), 1)

    def test_ties_break_on_lowest_index_deterministically(self):
        candidates = [{"a": "a2", "b": "b2"}, {"c": "c2", "d": "d2"}]
        picked = select_diverse([BASE], candidates, WORDS, k=1,
                                min_distance_from_base=2)
        self.assertEqual(picked, [0])

    def test_avoids_redundancy_with_an_already_kept_rendering(self):
        already_kept = {"a": "a2", "b": "b2"}
        # candidate 0 nearly repeats the already-kept rendering (dist 1 from
        # it, even though it's far from base); candidate 1 is far from both.
        near_kept = {"a": "a2", "b": "b2", "c": "c2"}
        far_from_both = {"c": "c2", "d": "d2"}
        picked = select_diverse([BASE, already_kept],
                                [near_kept, far_from_both], WORDS, k=1,
                                min_distance_from_base=2)
        self.assertEqual(picked, [1])


if __name__ == "__main__":
    unittest.main()
