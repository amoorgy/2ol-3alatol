import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval.phenomena import (
    CLIPPING,
    DIGIT_POSITION,
    DEGEMINATION,
    DIGIT_AMBIG,
    DIPHTHONG,
    TAA_MARBUTA,
    DIGIT_SAFE,
    EXCLUDED_DIGITS,
    _COMBOS,
    _STEPS,
    FINAL_Y_I,
    INITIAL_HAMZA,
    VOWEL_DROP,
    VOWEL_DROP_MED,
    VOWEL_LENGTH,
    UNRELATED,
    VOWEL_SWAP,
    BY_DIGIT,
    classify_edge,
    illegal_variants,
    is_licensed,
)


class NativeReviewedInventory(unittest.TestCase):
    """The digit inventory reflects the 2026-08-20 native review."""

    def test_excluded_digits_absent(self):
        for digit in EXCLUDED_DIGITS:
            self.assertNotIn(digit, BY_DIGIT, f"{digit} is not modern Egyptian usage")

    def test_digraph_digits_are_reversible(self):
        for digit in ("5", "8", "4"):
            self.assertTrue(BY_DIGIT[digit].reversible)

    def test_ambiguous_digits_are_not_reversible(self):
        for digit in ("7", "3", "2"):
            self.assertFalse(BY_DIGIT[digit].reversible)

    def test_four_is_flagged_rare(self):
        self.assertTrue(BY_DIGIT["4"].rare)
        self.assertFalse(BY_DIGIT["5"].rare)


class EdgeClassification(unittest.TestCase):
    """Every pair supplied by the native speaker must classify as expected."""

    # (a, b, expected classes as a set, expected n_edits)
    CASES = [
        # --- single-phenomenon, hand-authored -------------------------------
        ("walahy", "walahi", {FINAL_Y_I}, 1),
        ("benhawel", "benhawl", {VOWEL_DROP_MED}, 1),
        ("gedan", "gdn", {VOWEL_DROP}, 1),
        ("5asam", "khasam", {DIGIT_SAFE}, 1),
        ("akhsam", "a5sam", {DIGIT_SAFE}, 1),
        ("ghror", "8ror", {DIGIT_SAFE}, 1),
        ("a7san", "ahsan", {DIGIT_AMBIG}, 1),
        ("3ayez", "aayez", {DIGIT_AMBIG}, 1),
        ("aayez", "ayez", {VOWEL_LENGTH}, 1),
        ("sha2a", "4a2a", {DIGIT_SAFE}, 1),
        # --- compound -------------------------------------------------------
        ("akhsam", "2a5sam", {DIGIT_SAFE, INITIAL_HAMZA}, 2),
        ("3ayez", "ayez", {DIGIT_AMBIG, VOWEL_LENGTH}, 2),
        # --- mined from data/benchmark/variant_clusters.tsv ------------------
        ("7aga", "haga", {DIGIT_AMBIG}, 1),
        ("mesh", "msh", {VOWEL_DROP_MED}, 1),
        ("elly", "ely", {DEGEMINATION}, 1),
        ("elly", "elli", {FINAL_Y_I}, 1),
        ("ely", "elli", {FINAL_Y_I, DEGEMINATION}, 2),
        ("keda", "kida", {VOWEL_SWAP}, 1),
        # --- diphthong: one phonological process, not two vowel edits --------
        ("3ein", "3een", {DIPHTHONG}, 1),
        ("beit", "beet", {DIPHTHONG}, 1),
        ("3ein", "3en", {DIPHTHONG, VOWEL_LENGTH}, 2),
        # --- taa marbuta: ~20% of Egyptian word types end in -a/-ah ----------
        ("shanta", "shantah", {TAA_MARBUTA}, 1),
        ("7aga", "7agah", {TAA_MARBUTA}, 1),
    ]

    def test_expected_classes(self):
        for a, b, classes, n_edits in self.CASES:
            with self.subTest(pair=f"{a}/{b}"):
                edge = classify_edge(a, b)
                self.assertTrue(edge.converged, f"{a}/{b} failed to converge")
                self.assertEqual(set(edge.classes), classes)
                self.assertEqual(edge.n_edits, n_edits)

    def test_compound_is_harder_than_its_parts(self):
        """Difficulty must be monotone: adding a phenomenon cannot get easier."""
        simple = classify_edge("akhsam", "a5sam").n_edits
        compound = classify_edge("akhsam", "2a5sam").n_edits
        self.assertGreater(compound, simple)

    def test_identical_spellings_have_no_phenomena(self):
        edge = classify_edge("gedan", "gedan")
        self.assertEqual(edge.classes, ())
        self.assertFalse(edge.is_variant)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            set(classify_edge(" A5sam ", "AKHSAM").classes), {DIGIT_SAFE}
        )

    def test_ba3atelak_is_compound_digit_plus_vowel_loss(self):
        """The author's hardest example: a digit change and vowel loss at once."""
        edge = classify_edge("ba3atelak", "baatelk")
        self.assertTrue(edge.converged)
        self.assertIn(DIGIT_AMBIG, edge.classes)
        self.assertGreaterEqual(edge.n_edits, 2)

    def test_2ol_oul_involves_the_glottal_digit(self):
        edge = classify_edge("2ol", "oul")
        self.assertTrue(edge.converged)
        self.assertIn("2", edge.digits)


class Clipping(unittest.TestCase):
    """Glide elision is a real variant; non-glide drops must stay unresolved."""

    def test_tayeb_tab_converges_via_clipping(self):
        edge = classify_edge("tayeb", "tab")
        self.assertTrue(edge.converged, "tayeb/tab is a real Egyptian variant pair")
        self.assertIn(CLIPPING, edge.classes)

    def test_miner_error_still_refuses_to_converge(self):
        """2aham/7amam are أهم and حمام: different words, must stay UNRELATED."""
        edge = classify_edge("2aham", "7amam")
        self.assertFalse(edge.converged)
        self.assertIn(UNRELATED, edge.classes)

    def test_clipping_does_not_hijack_local_explanations(self):
        """walahy/walahi is a final-letter change, not a dropped glide."""
        self.assertEqual(set(classify_edge("walahy", "walahi").classes), {FINAL_Y_I})


class DigitPosition(unittest.TestCase):
    """Digits in different places is a phenomenon, not an absence of one."""

    def test_a3mel_vs_3amel(self):
        """أعمل (I do) and عامل (doing) both canonicalise to 'aamel'."""
        edge = classify_edge("a3mel", "3amel")
        self.assertEqual(set(edge.classes), {DIGIT_POSITION})

    def test_ma3na_vs_m3ana(self):
        edge = classify_edge("ma3na", "m3ana")
        self.assertEqual(set(edge.classes), {DIGIT_POSITION})

    def test_never_empty_for_distinct_strings(self):
        """No distinct pair may come back with no explanation at all."""
        for a, b in [("a3mel", "3amel"), ("ma3na", "m3ana"), ("3ayez", "ayez")]:
            self.assertTrue(classify_edge(a, b).classes, f"{a}/{b} unexplained")


class Licensing(unittest.TestCase):
    """Latin-to-digit rewriting is licensed only by the Arabic pre-image."""

    def test_h_to_7_licensed_when_arabic_has_hah(self):
        self.assertTrue(is_licensed("7", "أحسن"))

    def test_h_to_7_unlicensed_when_arabic_has_heh(self):
        self.assertFalse(is_licensed("7", "هنا"))

    def test_a_to_3_licensed_only_for_ayn(self):
        self.assertTrue(is_licensed("3", "عايز"))
        self.assertFalse(is_licensed("3", "هنا"))

    def test_qaf_licenses_the_glottal_digit(self):
        """Cairene qaf surfaces as a glottal stop, so 2 stands for it."""
        self.assertTrue(is_licensed("2", "قول"))

    def test_unknown_digit_is_never_licensed(self):
        for digit in EXCLUDED_DIGITS + ("x", ""):
            self.assertFalse(is_licensed(digit, "هنا"))


class IllegalVariants(unittest.TestCase):
    """The over-robustness probe needs forms a native speaker would reject."""

    def test_hena_yields_7ena(self):
        variants = dict(illegal_variants("hena", "هنا"))
        self.assertIn("7ena", variants)
        self.assertEqual(variants["7ena"], "7")

    def test_licensed_word_is_not_offered_as_illegal(self):
        """a7san is a real spelling, so it must never be generated as illegal."""
        variants = dict(illegal_variants("ahsan", "أحسن"))
        self.assertNotIn("a7san", variants)

    def test_reversible_digits_are_never_illegal(self):
        """kh<->5 is safe both ways, so no 5-form is ever illegal."""
        for variant, digit in illegal_variants("khasam", "خصم"):
            self.assertNotEqual(digit, "5")

    def test_illegal_variants_differ_from_the_input(self):
        for variant, _ in illegal_variants("hena", "هنا"):
            self.assertNotEqual(variant, "hena")


class Diphthong(unittest.TestCase):
    """The /aj/ and /aw/ nuclei are one Egyptian habit, not a pile of edits.

    Before DIPHTHONG existed, 3ein/3een read as VOWEL_SWAP, 3ein/3en as
    VOWEL_DROP_MED, and 3ayn/3een as CLIPPING+VOWEL_DROP_MED -- three
    explanations for one process, which made the per-phenomenon fragmentation
    table meaningless for exactly the class Egyptians vary most freely.
    """

    def test_monophthongisation_is_one_phenomenon_not_two(self):
        for a, b in (("3ein", "3een"), ("bait", "beet"), ("zay", "zey"),
                     ("lawn", "loon"), ("aywa", "aiwa")):
            with self.subTest(pair=f"{a}/{b}"):
                self.assertEqual(set(classify_edge(a, b).classes), {DIPHTHONG})

    def test_diphthong_beats_the_vowel_swap_explanation(self):
        self.assertEqual(classify_edge("beit", "beet").n_edits, 1)

    def test_diphthong_beats_the_vowel_strip_explanation(self):
        # cost 6 (CLIPPING + vowel strip) before; cost 1 now.
        self.assertEqual(classify_edge("3ayn", "3een").n_edits, 1)

    def test_onset_glide_is_not_a_diphthong(self):
        """The (?![aeiou]) lookahead. In tayeb the y opens the next syllable."""
        self.assertIn(CLIPPING, classify_edge("tayeb", "tab").classes)
        self.assertEqual(set(classify_edge("mayya", "maya").classes), {DEGEMINATION})
        self.assertEqual(set(classify_edge("howa", "hoa").classes), {CLIPPING})

    def test_diphthong_step_precedes_vowel_length(self):
        """Do not 'tidy' the _STEPS ladder: _apply runs steps in index order.

        With VOWEL_LENGTH first, 3een collapses to 3en before _monophthongize
        sees it and 3ein/3een stops converging. Cost decides which combination
        wins; index decides what each one computes.
        """
        labels = [label for label, _fn, _cost in _STEPS]
        self.assertLess(labels.index(DIPHTHONG), labels.index(VOWEL_LENGTH))

    def test_unrelated_pairs_stay_unrelated(self):
        for a, b in (("2aham", "7amam"), ("dahab", "zahab")):
            with self.subTest(pair=f"{a}/{b}"):
                self.assertFalse(classify_edge(a, b).converged)


class TaaMarbuta(unittest.TestCase):
    """taxonomy.py and normalize.py both knew about final -a/-ah; the ladder did not.

    Every taa-marbuta pair came back UNRELATED, which mattered because 98 of
    the 500 selected NileChat word types end in -a or -ah.
    """

    def test_final_ah_and_a_are_one_phenomenon(self):
        for a, b in (("shanta", "shantah"), ("7aga", "7agah"), ("sa3a", "sa3ah")):
            with self.subTest(pair=f"{a}/{b}"):
                self.assertEqual(set(classify_edge(a, b).classes), {TAA_MARBUTA})

    def test_it_is_symmetric(self):
        self.assertEqual(classify_edge("shantah", "shanta").classes,
                         classify_edge("shanta", "shantah").classes)

    def test_medial_ah_is_untouched(self):
        """Only the word-final -ah folds; ahsan must not become asan."""
        self.assertFalse(classify_edge("ahsan", "asan").converged)


class CombinationOrdering(unittest.TestCase):
    """classify_edge stops at the first converging subset, so order IS the answer."""

    def test_combinations_are_ordered_cheapest_first(self):
        costs = [cost for cost, _combo in _COMBOS]
        self.assertEqual(costs, sorted(costs))

    def test_every_subset_is_present_exactly_once(self):
        combos = [combo for _cost, combo in _COMBOS]
        self.assertEqual(len(combos), len(set(combos)))
        self.assertEqual(len(combos), 2 ** len(_STEPS) - 1)

    def test_declared_cost_matches_the_ladder(self):
        for cost, combo in _COMBOS:
            self.assertEqual(cost, sum(_STEPS[i][2] for i in combo))


if __name__ == "__main__":
    unittest.main()
