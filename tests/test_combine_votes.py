import csv
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_votes",
    ROOT / "experiments" / "d5_cohere_track" / "training_data" / "council" / "combine_votes.py")
cv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cv)

HEADER = ["sentence_id", "rendering_id", "valid", "votes_valid", "votes_invalid",
          "n_judges", "bad_substitutions", "reason"]


def write_tsv(path: pathlib.Path, rows: list[list[str]]) -> pathlib.Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


class Combine(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.dir = pathlib.Path(handle.name)

    def row(self, sid, valid):
        return [sid, sid, valid, "5", "0", "5", "", ""]

    def test_rows_keep_their_verdict_and_gain_a_panel_column(self):
        a = write_tsv(self.dir / "a.tsv", [self.row("x->y", "true")])
        b = write_tsv(self.dir / "b.tsv", [self.row("p->q", "false")])
        out = self.dir / "out.tsv"
        n = cv.combine([("opus/xhigh>=4", a), ("sonnet/medium>=5", b)], out)
        self.assertEqual(n, 2)
        rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
        self.assertEqual([(r["sentence_id"], r["valid"], r["panel"]) for r in rows],
                         [("x->y", "true", "opus/xhigh>=4"), ("p->q", "false", "sonnet/medium>=5")])

    def test_a_pair_judged_by_two_panels_is_refused(self):
        a = write_tsv(self.dir / "a.tsv", [self.row("x->y", "true")])
        b = write_tsv(self.dir / "b.tsv", [self.row("x->y", "false")])
        with self.assertRaises(ValueError):
            cv.combine([("a", a), ("b", b)], self.dir / "out.tsv")

    def test_mismatched_headers_are_refused(self):
        a = write_tsv(self.dir / "a.tsv", [self.row("x->y", "true")])
        b = self.dir / "b.tsv"
        b.write_text("sentence_id\tvalid\nz\ttrue\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            cv.combine([("a", a), ("b", b)], self.dir / "out.tsv")


if __name__ == "__main__":
    unittest.main()
