"""Scenario bank for Dataset B fusion (data/stimulus/scenarios_v2.tsv).

Two layers: the committed bank file itself (contract, pinned ids, content
rules, no near-duplicates) and the loader / review round-trip on temp files.
"""

import csv
import pathlib
import re
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval import scenario_bank
from franco_eval.fusion import SCENARIOS
from franco_eval.scenario_bank import (NEAR_DUP_JACCARD, PINNED_IDS, apply_review,
                                       jaccard, load_scenarios, near_duplicates,
                                       read_bank, verdict_status, write_review_sheet)

BANK = ROOT / "data" / "stimulus" / "scenarios_v2.tsv"
REVIEW = ROOT / "data" / "stimulus" / "scenarios_v2_review.csv"

MIN_WORDS, MAX_WORDS = 8, 18
# Capitalised words a scenario may carry: an acronym, holidays, weekdays,
# one river. Anything else capitalised mid-text is a named person or brand.
ALLOWED_CAPITALISED = frozenset(
    "TV ID Ramadan Eid Nile Mother's Day Friday Thursday Saturday".split())


def new_rows():
    return [row for row in read_bank(BANK) if row.scenario_id >= PINNED_IDS]


class BankContract(unittest.TestCase):
    """Another agent codes against this file; the header and ids are fixed."""

    def test_header_is_exactly_the_contract(self):
        first = BANK.read_text(encoding="utf-8").split("\n", 1)[0]
        self.assertEqual(first, "scenario_id\tdomain\ttext\tstatus")

    def test_ids_0_to_15_are_fusion_scenarios_verbatim_in_order(self):
        rows = read_bank(BANK)
        self.assertEqual(PINNED_IDS, len(SCENARIOS))
        for index, text in enumerate(SCENARIOS):
            with self.subTest(index=index):
                self.assertEqual(rows[index].scenario_id, index)
                self.assertEqual(rows[index].text, text)
                self.assertEqual(rows[index].status, "accepted")

    def test_ids_unique_and_contiguous(self):
        ids = [row.scenario_id for row in read_bank(BANK)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, list(range(len(ids))))

    def test_bank_is_a_few_hundred_new_scenarios(self):
        self.assertGreaterEqual(len(new_rows()), 250)

    def test_statuses_are_known(self):
        for row in read_bank(BANK):
            self.assertIn(row.status, scenario_bank.STATUSES)

    def test_domains_are_known_and_spread(self):
        counts = {}
        for row in read_bank(BANK):
            self.assertIn(row.domain, scenario_bank.DOMAINS, row.text)
            counts[row.domain] = counts.get(row.domain, 0) + 1
        self.assertGreaterEqual(len(counts), 18)
        for domain, count in counts.items():
            self.assertGreaterEqual(count, 10, domain)


class NewScenarioContent(unittest.TestCase):
    """The rules fusion.py learned the hard way, applied to ids 16+."""

    def test_length_bounds(self):
        for row in new_rows():
            with self.subTest(id=row.scenario_id):
                self.assertTrue(MIN_WORDS <= len(row.text.split()) <= MAX_WORDS,
                                f"{len(row.text.split())} words: {row.text}")

    def test_second_person(self):
        for row in new_rows():
            self.assertRegex(row.text, r"^you('re|'ve| )", row.text)

    def test_no_named_people_or_brands(self):
        for row in new_rows():
            capitalised = {w for w in row.text.split() if w[0].isupper()}
            self.assertLessEqual(capitalised, ALLOWED_CAPITALISED, row.text)

    def test_no_arabizi_no_quotes_no_trailing_punctuation(self):
        """Digits read as Arabizi letters (an example phrasing to echo);
        quotes dictate a word; the prompts append their own punctuation."""
        for row in new_rows():
            with self.subTest(id=row.scenario_id):
                self.assertTrue(row.text.isascii())
                self.assertIsNone(re.search(r"[0-9\"\t]", row.text))
                self.assertNotRegex(row.text, r"[.!?,;:]$")
                self.assertEqual(row.text, row.text.strip())

    def test_new_scenarios_start_pending_review(self):
        for row in new_rows():
            self.assertIn(row.status, ("pending_review", "accepted", "rejected"))


class NearDuplicates(unittest.TestCase):
    def test_no_pair_in_the_bank_reaches_the_threshold(self):
        texts = [row.text for row in read_bank(BANK)]
        hits = near_duplicates(texts)
        self.assertEqual(hits, [], "\n".join(
            f"{j:.2f}  {texts[a]}  ||  {texts[b]}" for a, b, j in hits[:10]))

    def test_threshold_catches_a_one_word_swap(self):
        """Positive control: swapping one participant is not a new scenario."""
        pair = ["you're asking a friend for help with something",
                "you're asking a neighbour for help with something"]
        self.assertGreaterEqual(jaccard(*pair), NEAR_DUP_JACCARD)
        self.assertEqual(len(near_duplicates(pair)), 1)

    def test_function_words_do_not_count_as_overlap(self):
        self.assertEqual(jaccard("you're at the market", "you're at the clinic"), 0.0)


def _write_bank(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("scenario_id\tdomain\ttext\tstatus\n")
        for row in rows:
            handle.write("\t".join(str(v) for v in row) + "\n")


def _mini_rows(extra):
    pinned = [(i, "daily_life", text, "accepted") for i, text in enumerate(SCENARIOS)]
    return pinned + extra


class LoadScenarios(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.path = self.dir / "bank.tsv"

    def test_accepted_only_by_default(self):
        _write_bank(self.path, _mini_rows([
            (16, "food_cooking", "you're cooking lunch for the whole family today", "accepted"),
            (17, "money", "you're asking to borrow money until the end of the month", "pending_review"),
            (18, "weather", "you're complaining about the heat all week long", "rejected"),
        ]))
        got = load_scenarios(self.path)
        self.assertEqual([i for i, _t in got], [*range(16), 16])
        self.assertEqual(got[0], (0, SCENARIOS[0]))
        self.assertIsInstance(got[16], tuple)

    def test_not_accepted_only_still_never_returns_rejected(self):
        _write_bank(self.path, _mini_rows([
            (16, "money", "you're asking to borrow money until the end of the month", "pending_review"),
            (17, "weather", "you're complaining about the heat all week long", "rejected"),
        ]))
        ids = [i for i, _t in load_scenarios(self.path, accepted_only=False)]
        self.assertEqual(ids, [*range(16), 16])

    def test_the_committed_bank_loads(self):
        self.assertEqual(load_scenarios(BANK)[:16], list(enumerate(SCENARIOS)))

    def test_bad_files_fail_loudly(self):
        cases = {
            "duplicate id": _mini_rows([(15, "money", "you're paying twice today", "accepted")]),
            "unknown status": _mini_rows([(16, "money", "you're paying the rent late", "maybe")]),
            "pinned not accepted": [(0, "daily_life", SCENARIOS[0], "rejected"),
                                    *_mini_rows([])[1:]],
            "pinned missing": _mini_rows([])[1:],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                _write_bank(self.path, rows)
                with self.assertRaises(ValueError):
                    load_scenarios(self.path)

    def test_wrong_header_fails(self):
        self.path.write_text("id\tdomain\ttext\tstatus\n0\td\tt\taccepted\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_scenarios(self.path)


class Review(unittest.TestCase):
    NEW = [
        (16, "food_cooking", "you're cooking lunch for the whole family today", "pending_review"),
        (17, "money", "you're asking to borrow money until the end of the month", "pending_review"),
        (18, "weather", "you're complaining about the heat all week long", "pending_review"),
        (19, "work", "you're asking your manager for a day off next week", "pending_review"),
        (20, "friends", "you're telling a friend you can't come out tonight", "pending_review"),
    ]

    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.tsv = self.dir / "bank.tsv"
        self.csv = self.dir / "review.csv"
        _write_bank(self.tsv, _mini_rows(self.NEW))
        write_review_sheet(self.tsv, self.csv)

    def _fill(self, verdicts, texts=None):
        with self.csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                sid = int(row["scenario_id"])
                writer.writerow({**row, "reviewer": verdicts.get(sid, ""),
                                 "text": (texts or {}).get(sid, row["text"])})

    def test_review_sheet_lists_only_pending_rows_with_empty_reviewer(self):
        with self.csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["scenario_id", "domain", "text", "reviewer"])
        self.assertEqual([int(r["scenario_id"]) for r in rows], [16, 17, 18, 19, 20])
        self.assertTrue(all(r["reviewer"] == "" for r in rows))

    def test_round_trip_sets_accepted_and_rejected(self):
        self._fill({16: "ok", 17: "Accept: natural", 18: "no", 19: "yes."})
        apply_review(self.tsv, self.csv)
        status = {row.scenario_id: row.status for row in read_bank(self.tsv)}
        self.assertEqual([status[i] for i in range(16, 21)],
                         ["accepted", "accepted", "rejected", "accepted", "pending_review"])
        self.assertEqual([i for i, _t in load_scenarios(self.tsv)], [*range(16), 16, 17, 19])

    def test_apply_is_idempotent(self):
        self._fill({16: "ok", 18: "no"})
        apply_review(self.tsv, self.csv)
        once = self.tsv.read_bytes()
        apply_review(self.tsv, self.csv)
        self.assertEqual(self.tsv.read_bytes(), once)

    def test_pinned_ids_stay_accepted_whatever_the_reviewer_says(self):
        with self.csv.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([3, "friends", SCENARIOS[3], "no"])
        apply_review(self.tsv, self.csv)
        self.assertEqual(read_bank(self.tsv)[3].status, "accepted")

    def test_edited_text_is_refused_not_silently_dropped(self):
        before = self.tsv.read_bytes()
        self._fill({16: "ok"}, texts={16: "you're cooking dinner instead"})
        with self.assertRaises(ValueError):
            apply_review(self.tsv, self.csv)
        self.assertEqual(self.tsv.read_bytes(), before)

    def test_unknown_id_is_refused(self):
        with self.csv.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([999, "money", "you're lost", "ok"])
        with self.assertRaises(ValueError):
            apply_review(self.tsv, self.csv)

    def test_review_sheet_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            write_review_sheet(self.tsv, self.csv)

    def test_spreadsheet_resave_with_bom_and_semicolons_still_applies(self):
        """A German-locale Excel re-save writes a BOM and ';' separators."""
        with self.csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with self.csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows({**r, "reviewer": "ok" if r["scenario_id"] == "17" else "no"}
                             for r in rows)
        apply_review(self.tsv, self.csv)
        self.assertEqual([i for i, _t in load_scenarios(self.tsv)], [*range(16), 17])

    def test_verdict_words(self):
        for verdict, expected in [("ok", "accepted"), ("OK, fine", "accepted"),
                                  ("accepted", "accepted"), ("yes", "accepted"),
                                  ("", "pending_review"), ("  ", "pending_review"),
                                  ("no", "rejected"), ("okay-ish?", "rejected"),
                                  ("reword: you're x", "rejected")]:
            with self.subTest(verdict=verdict):
                self.assertEqual(verdict_status(verdict), expected)


class CommittedReviewSheet(unittest.TestCase):
    def test_sheet_matches_the_bank(self):
        bank = {row.scenario_id: row for row in read_bank(BANK)}
        with REVIEW.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), ["scenario_id", "domain", "text", "reviewer"])
        self.assertEqual({int(r["scenario_id"]) for r in rows},
                         {i for i in bank if i >= PINNED_IDS})
        for row in rows:
            source = bank[int(row["scenario_id"])]
            self.assertEqual((row["domain"], row["text"]), (source.domain, source.text))


if __name__ == "__main__":
    unittest.main()
