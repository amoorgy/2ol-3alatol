import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.tokenizer_metrics import (
    aggregate_by_class,
    auc,
    fertility,
    fragment_edge,
)


def char_encode(text: str) -> list[int]:
    """Fake tokenizer: one token per character. Deterministic and dependency-free."""
    return [ord(c) for c in text]


def whole_word_encode(text: str) -> list[int]:
    """Fake tokenizer with a vocabulary that happens to contain both spellings."""
    vocab = {"a5sam": 10, "akhsam": 10, "hena": 11, "7ena": 12}
    return [vocab.get(text, 99)]


class Fragmentation(unittest.TestCase):
    def test_identical_spellings_do_not_fragment(self):
        edge = fragment_edge(char_encode, "akhsam", "akhsam")
        self.assertTrue(edge.same_seq)
        self.assertEqual(edge.frag, 0.0)
        self.assertEqual(edge.len_delta, 0)

    def test_digit_digraph_pair_shares_little(self):
        """5asam vs khasam: the digit and the digraph share no characters."""
        edge = fragment_edge(char_encode, "5asam", "khasam")
        self.assertFalse(edge.same_seq)
        self.assertGreater(edge.frag, 0.0)
        self.assertEqual(edge.len_delta, 1)

    def test_a_tokenizer_that_unifies_spellings_reports_zero_fragmentation(self):
        edge = fragment_edge(whole_word_encode, "a5sam", "akhsam")
        self.assertTrue(edge.same_seq)
        self.assertEqual(edge.frag, 0.0)

    def test_disjoint_tokens_give_full_fragmentation(self):
        edge = fragment_edge(whole_word_encode, "hena", "7ena")
        self.assertEqual(edge.frag, 1.0)

    def test_empty_strings_are_treated_as_identical(self):
        self.assertEqual(fragment_edge(char_encode, "", "").frag, 0.0)


class Fertility(unittest.TestCase):
    def test_tokens_per_word(self):
        self.assertEqual(fertility(char_encode, ["abc", "de"]), 2.5)

    def test_blank_words_are_ignored(self):
        self.assertEqual(fertility(char_encode, ["abc", ""]), 3.0)

    def test_no_words_is_zero_not_a_crash(self):
        self.assertEqual(fertility(char_encode, []), 0.0)


class Aggregation(unittest.TestCase):
    def test_edge_counts_under_every_class_it_belongs_to(self):
        edges = [fragment_edge(char_encode, "a5sam", "akhsam", ("DIGIT_SAFE", "X"))]
        summaries = {s.phenomenon: s for s in aggregate_by_class(edges)}
        self.assertEqual(summaries["DIGIT_SAFE"].n_edges, 1)
        self.assertEqual(summaries["X"].n_edges, 1)

    def test_sorted_worst_fragmentation_first(self):
        edges = [
            fragment_edge(whole_word_encode, "a5sam", "akhsam", ("easy",)),
            fragment_edge(whole_word_encode, "hena", "7ena", ("hard",)),
        ]
        self.assertEqual([s.phenomenon for s in aggregate_by_class(edges)],
                         ["hard", "easy"])

    def test_unlabelled_edges_still_aggregate(self):
        edges = [fragment_edge(char_encode, "a", "b")]
        self.assertEqual(aggregate_by_class(edges)[0].phenomenon, "UNLABELLED")


class FragmentEdgeToAggregation(unittest.TestCase):
    """The end-to-end path the E1 runner uses.

    aggregate_by_class sat unreachable for one missing argument at the runner
    call site -- fragment_edge was never handed `classes` -- so this asserts
    the full path, not just the two functions in isolation."""

    def test_classes_flow_from_fragment_edge_into_the_aggregate(self):
        edges = [
            fragment_edge(char_encode, "5asam", "khasam",
                          classes=("DIGIT_SAFE",)),
            fragment_edge(char_encode, "akhsam", "2a5sam",
                          classes=("DIGIT_SAFE", "INITIAL_HAMZA")),
            fragment_edge(char_encode, "elly", "elli", classes=("FINAL_Y_I",)),
        ]
        summary = {s.phenomenon: s for s in aggregate_by_class(edges)}
        self.assertEqual(summary["DIGIT_SAFE"].n_edges, 2)
        self.assertEqual(summary["INITIAL_HAMZA"].n_edges, 1)
        # a compound edge counts under both classes, so the total exceeds
        # the number of edges
        self.assertGreater(sum(c.n_edges for c in summary.values()), len(edges))


class Separability(unittest.TestCase):
    """AUC answers: can this score tell the two sections apart?"""

    def test_perfect_separation(self):
        self.assertEqual(auc([0.9, 0.8], [0.2, 0.1]), 1.0)

    def test_no_separation_when_identical(self):
        self.assertEqual(auc([0.5, 0.5], [0.5, 0.5]), 0.5)

    def test_reversed_separation(self):
        self.assertEqual(auc([0.1], [0.9]), 0.0)

    def test_ties_count_half(self):
        self.assertEqual(auc([0.5], [0.5, 0.1]), 0.75)

    def test_empty_side_is_uninformative_not_a_crash(self):
        self.assertEqual(auc([], [0.3]), 0.5)
        self.assertEqual(auc([0.3], []), 0.5)


if __name__ == "__main__":
    unittest.main()
