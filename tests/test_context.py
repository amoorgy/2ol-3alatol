import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.context import (accumulate, build_model, calibrate,
                                 context_vocabulary)

# A frozen miniature corpus. "3ayez" and "3ayz" are one word spelled two ways,
# so they are given the same company; "3ayn" is a different word and keeps its
# own. "el" and "w" are deliberately everywhere, to be skipped as contexts.
CORPUS = (
    [["ana", "3ayez", "akl", "el", "w"]] * 4
    + [["ana", "3ayz", "akl", "el", "w"]] * 4
    + [["howa", "3ayez", "shorb", "el", "w"]] * 3
    + [["howa", "3ayz", "shorb", "el", "w"]] * 3
    + [["shaf", "3ayn", "gamila", "el", "w"]] * 4
    + [["fata7", "3ayn", "wa7da", "el", "w"]] * 3
)
TARGETS = {"3ayez", "3ayz", "3ayn"}


def model_from(corpus, skip_top=2, min_count=1):
    counts = {}
    for words in corpus:
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    index = context_vocabulary(counts, skip_top=skip_top, size=50)
    return build_model(*accumulate(corpus, TARGETS, index), min_count=min_count), index


class Separation(unittest.TestCase):
    """The property the trap ranking depends on."""

    def test_ppmi_cosine_ranks_known_variants_above_known_traps(self):
        model, _index = model_from(CORPUS)
        variant = model.cosine("3ayez", "3ayz")
        trap = model.cosine("3ayez", "3ayn")
        self.assertIsNotNone(variant)
        self.assertIsNotNone(trap)
        self.assertGreater(variant, trap)

    def test_context_vectors_ignore_the_top_frequency_words(self):
        """el and w sit beside everything and would swamp every vector."""
        model, index = model_from(CORPUS, skip_top=2)
        self.assertNotIn("el", index)
        self.assertNotIn("w", index)

    def test_offsets_are_kept_signed(self):
        """`el kitab` and `kitab el` are different evidence."""
        model, index = model_from(CORPUS)
        offsets = {offset for feature_map in model.vectors.values()
                   for offset, _ctx in feature_map}
        self.assertTrue({-1, 1}.issubset(offsets))


class CosineContract(unittest.TestCase):
    def test_cosine_is_symmetric(self):
        model, _ = model_from(CORPUS)
        self.assertAlmostEqual(model.cosine("3ayez", "3ayz"),
                               model.cosine("3ayz", "3ayez"))

    def test_cosine_is_bounded(self):
        model, _ = model_from(CORPUS)
        for a, b in (("3ayez", "3ayz"), ("3ayez", "3ayn"), ("3ayez", "3ayez")):
            with self.subTest(pair=f"{a}/{b}"):
                self.assertGreaterEqual(model.cosine(a, b), 0.0)
                self.assertLessEqual(model.cosine(a, b), 1.0 + 1e-9)

    def test_identical_words_score_one(self):
        model, _ = model_from(CORPUS)
        self.assertAlmostEqual(model.cosine("3ayez", "3ayez"), 1.0, places=6)

    def test_unseen_word_returns_none_not_zero(self):
        """A missing vector is absent evidence, not evidence of difference."""
        model, _ = model_from(CORPUS)
        self.assertIsNone(model.cosine("3ayez", "kalb"))
        self.assertIsNone(model.cosine("kalb", "kalb"))

    def test_membership_reflects_a_usable_vector(self):
        model, _ = model_from(CORPUS)
        self.assertIn("3ayez", model)
        self.assertNotIn("kalb", model)


class Calibration(unittest.TestCase):
    """The threshold comes from the human-authored arm, per hard rule 1."""

    HEADER = "pair_id,label,subcategory,a,b\n"

    def _calibrate(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pairs.csv"
            path.write_text(self.HEADER + "".join(rows), encoding="utf-8")
            model, _ = model_from(CORPUS)
            return calibrate(model, path)

    def test_perfect_separation_scores_auc_one(self):
        result = self._calibrate(["p1,SAME,x,3ayez,3ayz\n",
                                  "p2,DIFFERENT,x,3ayez,3ayn\n"])
        self.assertEqual(result.auc, 1.0)
        self.assertEqual((result.n_same, result.n_different), (1, 1))

    def test_pairs_without_vectors_are_skipped_not_scored(self):
        result = self._calibrate(["p1,SAME,x,3ayez,3ayz\n",
                                  "p2,DIFFERENT,x,kalb,bent\n"])
        self.assertEqual(result.n_skipped, 1)

    def test_no_usable_rows_reports_chance_not_a_crash(self):
        result = self._calibrate(["p1,SAME,x,kalb,bent\n"])
        self.assertEqual(result.auc, 0.5)
        self.assertFalse(result.usable)

    def test_usable_requires_both_signal_and_sample_size(self):
        result = self._calibrate(["p1,SAME,x,3ayez,3ayz\n",
                                  "p2,DIFFERENT,x,3ayez,3ayn\n"])
        self.assertFalse(result.usable, "2 pairs is not enough to calibrate on")


if __name__ == "__main__":
    unittest.main()
