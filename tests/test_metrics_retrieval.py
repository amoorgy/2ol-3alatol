import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco.metrics import consistency_at_k, pairwise_agreement, recall_at_1
from franco.mine import edit_distance, mine_variants
from franco.retrieval import CharNgramIndex

DOCS = [
    "ezayak ya 3am kwayes",
    "el mat3am dah gamed gedan",
    "yalla beena 3ala el nadi",
    "mesh 3aref asaraha khalas",
]


class Retrieval(unittest.TestCase):
    def test_exact_self_retrieval(self):
        index = CharNgramIndex().fit(DOCS)
        for i, doc in enumerate(DOCS):
            self.assertEqual(index.search(doc, topk=1)[0][0], i)

    def test_variant_query_still_close(self):
        index = CharNgramIndex().fit(DOCS)
        self.assertEqual(index.search("ezayk ya 3am kwayis")[0][0], 0)


class Metrics(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(recall_at_1([1, 2, None], [1, 1, 3]), 1 / 3)

    def test_recall_length_mismatch(self):
        with self.assertRaises(ValueError):
            recall_at_1([1], [1, 2])

    def test_consistency(self):
        self.assertAlmostEqual(consistency_at_k([[1, 1, 1], [1, 2, 1]]), 0.5)
        self.assertEqual(consistency_at_k([]), 0.0)

    def test_total_failure_is_not_consistent(self):
        self.assertEqual(consistency_at_k([[None, None, None]]), 0.0)
        self.assertEqual(pairwise_agreement([[None, None, None]]), 0.0)

    def test_pairwise(self):
        self.assertAlmostEqual(pairwise_agreement([[1, 1, 2]]), 1 / 3)

    def test_search_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            CharNgramIndex().search("ezayak")


class Mining(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("khalas", "5alas"), 2)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_mines_planted_variants(self):
        texts = ["khalas ba2a"] * 3 + ["5alas yasta"] * 2 + ["kalam tany"] * 3
        clusters = mine_variants(texts)
        top = {w for c in clusters for w, _ in c["forms"]}
        self.assertIn("khalas", top)
        self.assertIn("5alas", top)
        self.assertNotIn("kalam", top)

    def test_inflectional_suffix_pairs_rejected(self):
        # gamed/gameda differ by the feminine -a morpheme, not spelling:
        # the prefix guard must keep them out of one cluster.
        texts = ["gamed awy"] * 3 + ["gameda awy"] * 2
        clusters = mine_variants(texts)
        forms = {w for c in clusters for w, _ in c["forms"]}
        self.assertFalse({"gamed", "gameda"} <= forms, clusters)


if __name__ == "__main__":
    unittest.main()
