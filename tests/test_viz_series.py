import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from visualization.series import (DotRow, agreement_ladder_series, bootstrap_ci,
                                  capability_tiles, flip_by_phenomenon_panels,
                                  frag_vs_flip_points, fragmentation_ceiling_series,
                                  phenomenon_matrix, robustness_points,
                                  separability_series, spearman)
from visualization.theme import annotate_n, cell_alpha

# A miniature results fixture as a literal -- no fixture files.
LEVEL1 = {
    "tokenizers": {
        "tok-a": {"group": "production", "family": "qwen",
                  "arms": {"human": {
                      "n_edges": 100, "n_same": 60, "n_different": 40,
                      "mean_frag_same": 0.7, "mean_frag_different": 0.79,
                      "separability_auc": 0.61, "separability_ci95": [0.55, 0.67],
                      "by_class": [
                          {"phenomenon": "DIGIT_SAFE", "n_edges": 30,
                           "mean_frag": 0.8, "same_seq_rate": 0.0,
                           "mean_len_delta": 1.0},
                          {"phenomenon": "FINAL_Y_I", "n_edges": 3,
                           "mean_frag": 0.4, "same_seq_rate": 0.2,
                           "mean_len_delta": 0.5}],
                      "by_subcategory": {}},
                      "nilechat": {
                      "n_edges": 200, "n_same": 200, "n_different": 0,
                      "mean_frag_same": 0.68, "mean_frag_different": None,
                      "separability_auc": None, "separability_ci95": None,
                      "by_class": [
                          {"phenomenon": "DIGIT_SAFE", "n_edges": 80,
                           "mean_frag": 0.75, "same_seq_rate": 0.0,
                           "mean_len_delta": 1.1}],
                      "by_subcategory": {}}}},
        "byt5": {"group": "control", "family": "byte-level",
                 "arms": {"human": {
                     "n_edges": 100, "n_same": 60, "n_different": 40,
                     "mean_frag_same": 0.22, "mean_frag_different": 0.28,
                     "separability_auc": 0.59, "separability_ci95": [0.51, 0.66],
                     "by_class": [], "by_subcategory": {}}}},
    },
}

LEVEL3 = {
    "models": {
        "model-x": {"n_items": 150, "control": {"accuracy": 0.8, "capability": "PASS"},
                    "variant_pairwise_agreement": 0.62,
                    "variant_consistency_at_k": 0.18,
                    "under_robust_rate": 0.38, "over_norm_rate": 0.25,
                    "ceiling_agreement": 0.9, "non_answer_rate": 0.05,
                    "by_class": {"DIGIT_SAFE": {"n_edges": 80, "n_none": 2,
                                                "flip_rate": 0.3},
                                 "FINAL_Y_I": {"n_edges": 4, "n_none": 0,
                                               "flip_rate": 0.5}}},
        "model-y": {"n_items": 150, "control": {"accuracy": 0.3, "capability": "FAIL"},
                    "variant_pairwise_agreement": 0.4,
                    "variant_consistency_at_k": 0.05,
                    "under_robust_rate": 0.6, "over_norm_rate": None,
                    "ceiling_agreement": None, "non_answer_rate": 0.4,
                    "by_class": {"DIGIT_SAFE": {"n_edges": 80, "n_none": 30,
                                                "flip_rate": 0.7}}},
    },
}


class NonePropagation(unittest.TestCase):
    """None is a value. It must survive every series untouched."""

    def test_missing_different_side_stays_none(self):
        rows = fragmentation_ceiling_series(LEVEL1, "nilechat")
        self.assertEqual(rows[0].values[1], None)

    def test_undefined_auc_stays_none(self):
        rows = separability_series(LEVEL1, "nilechat")
        self.assertIsNone(rows[0].values[0])

    def test_none_over_norm_reaches_the_scatter_as_none(self):
        points = {p.label: p for p in robustness_points(LEVEL3)}
        self.assertIsNone(points["model-y"].y)
        self.assertEqual(points["model-x"].y, 0.25)

    def test_an_arm_missing_entirely_is_skipped_not_zeroed(self):
        rows = fragmentation_ceiling_series(LEVEL1, "nilechat")
        self.assertEqual([r.label for r in rows], ["tok-a"])  # byt5 has no arm


class SmallN(unittest.TestCase):
    def test_alpha_bands_at_the_documented_boundaries(self):
        self.assertEqual(cell_alpha(4), 0.3)
        self.assertEqual(cell_alpha(5), 0.55)
        self.assertEqual(cell_alpha(19), 0.55)
        self.assertEqual(cell_alpha(20), 1.0)

    def test_annotation_appears_exactly_below_twenty(self):
        self.assertEqual(annotate_n(19), "n=19")
        self.assertEqual(annotate_n(20), "")

    def test_series_carry_their_n(self):
        _phen, _toks, _vals, ns = phenomenon_matrix(LEVEL1, "human")
        self.assertIn(3, [n for row in ns for n in row])


class Statistics(unittest.TestCase):
    def test_bootstrap_is_deterministic_under_the_project_seed(self):
        import statistics as st
        values = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2, 0.55]
        first = bootstrap_ci(values, st.fmean, boot=200)
        second = bootstrap_ci(values, st.fmean, boot=200)
        self.assertEqual(first, second)

    def test_bootstrap_brackets_the_point_estimate(self):
        import statistics as st
        values = [0.3, 0.4, 0.5, 0.6, 0.7]
        lo, hi = bootstrap_ci(values, st.fmean, boot=500)
        self.assertLess(lo, st.fmean(values))
        self.assertGreater(hi, st.fmean(values))

    def test_bootstrap_refuses_a_single_value(self):
        import statistics as st
        self.assertIsNone(bootstrap_ci([0.5], st.fmean))

    def test_spearman_on_monotone_and_reversed(self):
        self.assertEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_spearman_degenerate_input_is_none_not_a_crash(self):
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2], [1, 2]))


class Shapes(unittest.TestCase):
    def test_controls_sort_below_production(self):
        rows = fragmentation_ceiling_series(LEVEL1, "human")
        self.assertEqual(rows[-1].label, "byt5")

    def test_phenomenon_rows_sort_worst_first(self):
        phenomena, _toks, _vals, _ns = phenomenon_matrix(LEVEL1, "human")
        self.assertEqual(phenomena, ["DIGIT_SAFE", "FINAL_Y_I"])

    def test_ladder_inverts_non_answer_into_answered(self):
        rows = {r.label: r for r in agreement_ladder_series(LEVEL3)}
        self.assertEqual(rows["model-x"].values[3], 0.95)

    def test_capability_tiles_carry_the_verdict(self):
        tiles = dict((m, (a, ok)) for m, a, ok in capability_tiles(LEVEL3))
        self.assertEqual(tiles["model-x"], (0.8, True))
        self.assertEqual(tiles["model-y"], (0.3, False))

    def test_panels_share_one_class_order(self):
        panels = flip_by_phenomenon_panels(LEVEL3)
        orders = {tuple(r.label for r in rows) for rows in panels.values()}
        self.assertEqual(len(orders), 1)

    def test_frag_vs_flip_joins_on_phenomenon_constants(self):
        points, rho = frag_vs_flip_points(
            LEVEL1, LEVEL3, model_to_tokenizer={"model-x": "tok-a"},
            arm="nilechat")
        self.assertEqual([p.family for p in points], ["DIGIT_SAFE"])
        self.assertIsNone(rho)  # a single point cannot correlate

    def test_no_stdlib_layer_imports_matplotlib(self):
        """The layering rule that keeps 90% of this module venv-free."""
        base = pathlib.Path(__file__).resolve().parents[1] / "src" / "visualization"
        for module in ("theme.py", "loaders.py", "series.py", "__init__.py"):
            source = (base / module).read_text()
            for forbidden in ("import matplotlib", "from matplotlib"):
                self.assertNotIn(forbidden, source, module)


if __name__ == "__main__":
    unittest.main()
