import csv
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.anchor_bank import (BAND_LABELS, BELOW_BANDS, HEADER,
                                     band_of, bank_rows, family_ranks,
                                     incidental_spans, incidental_words,
                                     order_bare, select_bare, to_tsv,
                                     trap_anchors)

BANK = ROOT / "data" / "stimulus" / "v2_anchor_bank.tsv"
SPLIT = ROOT / "data" / "stimulus" / "nilechat" / "word_split.tsv"
TRAP_VOTES = ROOT / "data" / "stimulus" / "nilechat" / "trap_votes.tsv"


def vote(word, trap, valid="true", aw="", at="", mw="", mt=""):
    return {"word_type": word, "trap": trap, "valid": valid,
            "arabic_word": aw, "arabic_trap": at,
            "meaning_en_word": mw, "meaning_en_trap": mt}


class Bands(unittest.TestCase):
    def test_edges_are_lower_inclusive(self):
        self.assertEqual(band_of(200), "200-500")
        self.assertEqual(band_of(499), "200-500")
        self.assertEqual(band_of(500), "500-1k")
        self.assertEqual(band_of(999), "500-1k")
        self.assertEqual(band_of(1000), "1k-2k")
        self.assertEqual(band_of(4999), "2k-5k")
        self.assertEqual(band_of(5000), "5k+")
        self.assertEqual(band_of(10 ** 7), "5k+")

    def test_below_the_pool_gate_is_its_own_band(self):
        self.assertEqual(band_of(199), BELOW_BANDS)
        self.assertNotIn(BELOW_BANDS, BAND_LABELS)

    def test_labels_run_high_to_low(self):
        self.assertEqual(BAND_LABELS[0], "5k+")
        self.assertEqual(BAND_LABELS[-1], "200-500")


class TrapAnchors(unittest.TestCase):
    """Must match build_dataset_b.anchors_from_traps, deduplicated."""

    ROWS = [
        vote("2abl", "bal", aw="قبل", at="بال", mw="before", mt="mind"),
        vote("men", "meen", valid="false"),
        vote("heldw", "heldt"),
        vote("kan", "bal", aw="كان", at="بال-2", mw="was", mt="second"),
    ]

    def test_both_sides_in_order_first_occurrence_wins(self):
        out = trap_anchors(self.ROWS, {"2abl", "kan"})
        self.assertEqual(out, [("2abl", "قبل", "before"), ("bal", "بال", "mind"),
                               ("kan", "كان", "was")])

    def test_invalid_rows_never_yield_anchors(self):
        words = [w for w, _a, _g in trap_anchors(self.ROWS, None)]
        self.assertNotIn("men", words)
        self.assertNotIn("meen", words)

    def test_split_filters_on_the_pair_word(self):
        words = [w for w, _a, _g in trap_anchors(self.ROWS, {"2abl", "kan"})]
        self.assertNotIn("heldw", words)
        self.assertIn("heldw", [w for w, _a, _g in trap_anchors(self.ROWS, None)])


class Incidental(unittest.TestCase):
    def test_counts_sentences_where_a_word_would_be_a_non_anchor_span(self):
        counts, n = incidental_spans([
            ("ya sa7by ana ta3ban awi", "ta3ban"),
            ("embare7 ana ro7t el beit", "beit"),
            ("Ana 3ayez anam", "anam"),          # offset 0: never a span
            ("howa ana w ana", "howa"),          # twice: never a span
        ])
        self.assertEqual(n, 4)
        self.assertEqual(counts["ana"], 2)
        self.assertEqual(counts["ta3ban"], 0)   # it was the anchor there
        self.assertEqual(counts["awi"], 1)

    def test_rate_threshold_is_inclusive(self):
        counts = {"ana": 10, "awi": 9}
        self.assertEqual(incidental_words(counts, 1000, max_rate=0.01), {"ana"})

    def test_no_sentences_means_nothing_is_incidental(self):
        self.assertEqual(incidental_words({"ana": 3}, 0), frozenset())


class Families(unittest.TestCase):
    def test_most_frequent_spelling_heads_and_variants_rank_by_count(self):
        ranks = family_ranks({"3ashan": 100, "3shan": 50, "3ashn": 10})
        self.assertEqual(ranks, {"3ashan": 0, "3shan": 1, "3ashn": 2})

    def test_prefix_relation_is_inflection_not_a_variant(self):
        # same skeleton "kn", but kana extends kan: mine.py's guard
        self.assertEqual(family_ranks({"kan": 100, "kana": 50}),
                         {"kan": 0, "kana": 0})

    def test_far_members_of_a_skeleton_are_their_own_heads(self):
        # same skeleton "ktb", edit distance 3
        ranks = family_ranks({"katab": 100, "kotoob": 50})
        self.assertEqual(ranks, {"katab": 0, "kotoob": 0})

    def test_one_letter_skeletons_never_cluster(self):
        # mine.py skips skeletons shorter than 2 (lel/2al/ela all -> "l")
        self.assertEqual(family_ranks({"lel": 100, "ela": 50}),
                         {"lel": 0, "ela": 0})

    def test_ties_break_alphabetically(self):
        self.assertEqual(family_ranks({"sa7by": 7, "sahby": 7}),
                         {"sa7by": 0, "sahby": 1})

    def test_a_second_head_collects_its_own_variants(self):
        # skeleton "mSkl" is headed by moshkela; mashakel is 3 edits away,
        # so it heads its own family and mashaakel (1 edit) joins it
        ranks = family_ranks({"moshkela": 100, "mashakel": 50, "mashaakel": 10})
        self.assertEqual(ranks, {"moshkela": 0, "mashakel": 0, "mashaakel": 1})

    def test_a_word_joins_the_most_frequent_head_it_matches(self):
        # kamya is guarded from kam (prefix), so it heads; kameya joins it
        ranks = family_ranks({"kam": 1000, "kamya": 50, "kameya": 40})
        self.assertEqual(ranks, {"kam": 0, "kamya": 0, "kameya": 1})


class SelectBare(unittest.TestCase):
    def test_first_matching_reason_is_recorded_and_the_rest_kept(self):
        kept, reasons = select_bare(
            ["a1", "b1", "c1", "d1"],
            [("held_out", {"a1"}), ("trap_word", {"a1", "b1"})])
        self.assertEqual(kept, ["c1", "d1"])
        self.assertEqual(reasons, {"held_out": 1, "trap_word": 1})

    def test_input_order_does_not_matter(self):
        rules = [("english", {"mile"})]
        self.assertEqual(select_bare(["zz", "mile", "aa"], rules)[0],
                         select_bare(["aa", "zz", "mile"], rules)[0])


def fake_pool(per_band: int) -> dict[str, int]:
    """per_band words in each of the five bands (ranks are passed apart)."""
    counts = (9000, 3000, 1500, 700, 300)
    return {f"b{band}w{i:03d}": count for band, count in enumerate(counts)
            for i in range(per_band)}


class OrderBare(unittest.TestCase):
    def test_a_prefix_draws_every_band_equally(self):
        counts = fake_pool(20)
        order = order_bare(counts, counts, {w: 0 for w in counts})
        prefix = order[:25]
        per_band = {b: sum(band_of(counts[w]) == b for w in prefix)
                    for b in BAND_LABELS}
        self.assertEqual(set(per_band.values()), {5})
        self.assertEqual([band_of(counts[w]) for w in order[:5]],
                         list(BAND_LABELS))

    def test_exhausted_bands_drop_out_of_the_rotation(self):
        counts = {"top1": 9000, "low1": 300, "low2": 301, "low3": 302}
        order = order_bare(counts, counts, {w: 0 for w in counts})
        self.assertEqual(order[0], "top1")
        self.assertEqual(sorted(order[1:]), ["low1", "low2", "low3"])

    def test_every_head_precedes_every_variant(self):
        counts = fake_pool(10)
        ranks = {w: (1 if i % 3 == 0 else 0) for i, w in enumerate(sorted(counts))}
        order = order_bare(counts, counts, ranks)
        last_head = max(i for i, w in enumerate(order) if ranks[w] == 0)
        first_variant = min(i for i, w in enumerate(order) if ranks[w] == 1)
        self.assertLess(last_head, first_variant)

    def test_within_a_band_order_is_hashed_not_by_frequency(self):
        counts = {f"w{i}": 300 + i for i in range(40)}
        order = order_bare(counts, counts, {w: 0 for w in counts})
        self.assertNotEqual(order, sorted(counts, key=lambda w: -counts[w]))
        other = order_bare(counts, counts, {w: 0 for w in counts}, salt="x")
        self.assertNotEqual(order, other)
        self.assertEqual(sorted(order), sorted(other))

    def test_deterministic_and_input_order_independent(self):
        counts = fake_pool(8)
        ranks = {w: 0 for w in counts}
        self.assertEqual(order_bare(list(counts), counts, ranks),
                         order_bare(list(reversed(list(counts))), counts, ranks))


class Rows(unittest.TestCase):
    FREQ = {"2abl": (48554, 30000), "bal": (99, 50), "ezay": (187389, 113364)}

    def test_trap_rows_first_then_bare_with_contiguous_priority(self):
        rows = bank_rows([("2abl", "قبل", "before"), ("bal", "بال", "mind")],
                         ["ezay"], self.FREQ)
        self.assertEqual([r["word"] for r in rows], ["2abl", "bal", "ezay"])
        self.assertEqual([r["priority"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["source"], "trap")
        self.assertEqual(rows[0]["context_type"], "trap_contrast")
        self.assertEqual(rows[1]["band"], BELOW_BANDS)
        self.assertEqual(rows[2]["source"], "bare")
        self.assertEqual(rows[2]["context_type"], "general")
        self.assertEqual((rows[2]["arabic"], rows[2]["gloss_en"]), ("", ""))
        self.assertEqual(rows[2]["band"], "5k+")

    def test_a_word_can_only_appear_once(self):
        with self.assertRaises(ValueError):
            bank_rows([("ezay", "إزاي", "how")], ["ezay"], self.FREQ)

    def test_tsv_header_is_exactly_the_contract(self):
        text = to_tsv(bank_rows([("2abl", "قبل", "before")], ["ezay"], self.FREQ))
        self.assertEqual(text.splitlines()[0].split("\t"), list(HEADER))
        self.assertEqual(HEADER, ("word", "arabic", "gloss_en", "context_type",
                                  "source", "count", "doc_freq", "band",
                                  "priority"))

    def test_a_tab_inside_a_field_is_refused(self):
        with self.assertRaises(ValueError):
            to_tsv(bank_rows([("2abl", "قبل", "before\tafter")], [], self.FREQ))


@unittest.skipUnless(BANK.exists(), "anchor bank not built yet")
class BankContract(unittest.TestCase):
    """The tracked file another agent codes against."""

    @classmethod
    def setUpClass(cls):
        with BANK.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            cls.fields = reader.fieldnames
            cls.rows = list(reader)

    def test_header(self):
        self.assertEqual(tuple(self.fields), HEADER)

    def test_priority_is_the_row_order(self):
        self.assertEqual([int(r["priority"]) for r in self.rows],
                         list(range(1, len(self.rows) + 1)))

    def test_trap_rows_form_the_head_of_the_file(self):
        sources = [r["source"] for r in self.rows]
        n_trap = sources.count("trap")
        self.assertEqual(sources[:n_trap], ["trap"] * n_trap)
        self.assertEqual(set(sources[n_trap:]), {"bare"})
        for row in self.rows[:n_trap]:
            self.assertEqual(row["context_type"], "trap_contrast")
            self.assertTrue(row["arabic"], row["word"])
        for row in self.rows[n_trap:]:
            self.assertEqual(row["context_type"], "general")
            self.assertEqual((row["arabic"], row["gloss_en"]), ("", ""))

    def test_no_word_twice(self):
        words = [r["word"] for r in self.rows]
        self.assertEqual(len(words), len(set(words)))

    def test_band_matches_count(self):
        for row in self.rows:
            self.assertEqual(row["band"], band_of(int(row["count"])), row["word"])

    @unittest.skipUnless(SPLIT.exists() and TRAP_VOTES.exists(), "split missing")
    def test_firewall_no_held_out_word_and_no_trap_word_as_bare(self):
        with SPLIT.open(encoding="utf-8") as handle:
            heldout = {r["word"] for r in csv.DictReader(handle, delimiter="\t")
                       if r["side"] == "heldout"}
        with TRAP_VOTES.open(encoding="utf-8") as handle:
            trap_words = {w for r in csv.DictReader(handle, delimiter="\t")
                          for w in (r["word_type"], r["trap"])}
        for row in self.rows:
            self.assertNotIn(row["word"], heldout)
            if row["source"] == "bare":
                self.assertNotIn(row["word"], trap_words)


if __name__ == "__main__":
    unittest.main()
