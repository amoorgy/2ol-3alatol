"""merge_council.py generic --dir mode and prep_council_v2.py --fused/--chunk-dir.

Fixtures only: nothing under data/ is read or written.
"""

import contextlib
import importlib.util
import io
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACK = ROOT / "experiments" / "d5_cohere_track"


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge_council = load("merge_council", TRACK / "nilechat_wordset" / "merge_council.py")
prep_v2 = load("prep_council_v2", TRACK / "training_data" / "prep_council_v2.py")

LENSES = ("anchor", "coherence", "degeneracy", "fluency", "register")


def run_main(main, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def write_votes(directory: pathlib.Path, votes: dict[str, str],
                sids=("s1", "s2", "s3")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for lens, pattern in votes.items():
        rows = [{"sid": s, "rid": s, "v": p == "T", "why": "" if p == "T" else f"no {lens}"}
                for s, p in zip(sids, pattern)]
        (directory / f"{lens}_c0.json").write_text(json.dumps(rows), encoding="utf-8")


# merge_training's TSV for the fixture below, captured before the generic mode
# was added: pins the existing --training/--fusion-v2/--pairs-v2 output format.
PINNED_TSV = (
    "sentence_id\trendering_id\tvalid\tvotes_valid\tvotes_invalid\tn_judges\t"
    "bad_substitutions\treason\r\n"
    "s1\ts1\ttrue\t5\t0\t5\t\t\r\n"
    "s2\ts2\ttrue\t4\t1\t5\t\tno anchor\r\n"
    "s3\ts3\tfalse\t3\t2\t5\t\tno coherence || no register\r\n")
FIXTURE = {"anchor": "TFT", "coherence": "TTF", "degeneracy": "TTT",
           "fluency": "TTT", "register": "TTF"}


class MergeCouncilGenericDir(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        write_votes(self.tmp / "votes", FIXTURE)
        (self.tmp / "votes" / "panel.meta").write_text("{}", encoding="utf-8")

    def test_merge_training_output_is_pinned(self):
        result = merge_council.merge_training(self.tmp / "votes", self.tmp / "a.tsv", 4)
        self.assertEqual((self.tmp / "a.tsv").read_bytes().decode("utf-8"), PINNED_TSV)
        self.assertEqual(result["rejected"], 1)

    def test_generic_mode_writes_exactly_what_merge_training_writes(self):
        merge_council.merge_training(self.tmp / "votes", self.tmp / "a.tsv", 4)
        with mock.patch.object(merge_council, "OUT_DIR", self.tmp / "summary_dir"):
            code, out, _ = run_main(merge_council.main, [
                "--dir", str(self.tmp / "votes"), "--out", str(self.tmp / "b.tsv")])
        self.assertEqual(code, 0)
        self.assertEqual((self.tmp / "a.tsv").read_bytes(), (self.tmp / "b.tsv").read_bytes())
        self.assertIn("1 rejected", out)
        self.assertFalse((self.tmp / "summary_dir").exists(),
                         "generic mode must not touch council_summary.json")

    def test_generic_mode_refuses_items_missing_a_lens(self):
        rows = json.loads((self.tmp / "votes" / "register_c0.json").read_text())
        (self.tmp / "votes" / "register_c0.json").write_text(json.dumps(rows[:2]))
        code, _, err = run_main(merge_council.main, [
            "--dir", str(self.tmp / "votes"), "--out", str(self.tmp / "b.tsv")])
        self.assertEqual(code, 2)
        self.assertIn("1 items", err)
        self.assertFalse((self.tmp / "b.tsv").exists())
        code, _, _ = run_main(merge_council.main, [
            "--dir", str(self.tmp / "votes"), "--out", str(self.tmp / "b.tsv"),
            "--allow-incomplete"])
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "b.tsv").exists())

    def test_generic_mode_refuses_a_missing_lens(self):
        (self.tmp / "votes" / "register_c0.json").unlink()
        code, _, err = run_main(merge_council.main, [
            "--dir", str(self.tmp / "votes"), "--out", str(self.tmp / "b.tsv")])
        self.assertEqual(code, 2)
        self.assertIn("4 lenses", err)

    def test_generic_mode_refuses_non_boolean_votes(self):
        path = self.tmp / "votes" / "anchor_c0.json"
        rows = json.loads(path.read_text())
        rows[1]["v"] = "false"
        path.write_text(json.dumps(rows))
        code, _, err = run_main(merge_council.main, [
            "--dir", str(self.tmp / "votes"), "--out", str(self.tmp / "b.tsv")])
        self.assertEqual(code, 2)
        self.assertIn("non-boolean", err)

    def test_generic_mode_refuses_an_empty_dir(self):
        (self.tmp / "empty").mkdir()
        code, _, err = run_main(merge_council.main, [
            "--dir", str(self.tmp / "empty"), "--out", str(self.tmp / "b.tsv")])
        self.assertEqual(code, 2)
        self.assertIn("no verdict files", err)

    def test_dir_and_out_go_together(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                merge_council.main(["--dir", str(self.tmp / "votes")])


def fused_row(sid: str, word: str = "w") -> dict:
    return {"sentence_id": sid, "text": f"text {sid}", "doc_id": "fusion:x",
            "anchor": {"word": word, "arabic": "ا", "gloss_en": "g",
                       "context_type": "trap_contrast"}}


class PrepCouncilV2(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.a = self.tmp / "a.jsonl"
        self.b = self.tmp / "b.jsonl"
        self.a.write_text("".join(json.dumps(fused_row(s)) + "\n"
                                  for s in ("f1", "f2", "f3", "f1")), encoding="utf-8")
        self.b.write_text("".join(json.dumps(fused_row(s, "v")) + "\n"
                                  for s in ("f3", "f4")), encoding="utf-8")
        self.out = self.tmp / "chunks"

    def prep(self, *argv):
        return run_main(prep_v2.main, list(argv))

    def chunks(self) -> dict[str, list[dict]]:
        return {p.stem: json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(self.out.glob("c*.json"))}

    def test_defaults_are_unchanged(self):
        args = prep_v2.parse_args([])
        self.assertEqual(args.fused, [prep_v2.FUSED])
        self.assertEqual(args.chunk_dir, prep_v2.CHUNK_DIR)
        self.assertEqual(args.chunk_size, 115)
        self.assertEqual(prep_v2.FUSED, ROOT / "data" / "training" / "v2" / "fused_sentences.jsonl")
        self.assertEqual(prep_v2.CHUNK_DIR,
                         ROOT / "data" / "generated" / "v2" / "council" / "fused_chunks")

    def test_several_fused_files_first_id_wins(self):
        code, _, _ = self.prep("--fused", str(self.a), "--fused", str(self.b),
                               "--chunk-dir", str(self.out), "--chunk-size", "3")
        self.assertEqual(code, 0)
        chunks = self.chunks()
        self.assertEqual([[i["sid"] for i in c] for c in chunks.values()],
                         [["f1", "f2", "f3"], ["f4"]])
        self.assertEqual(chunks["c0"][2]["anchor"], "w", "f3 must come from the first file")
        self.assertEqual(chunks["c1"][0], {"sid": "f4", "rid": "f4", "anchor": "v",
                                          "arabic": "ا", "gloss": "g",
                                          "sentence": "text f4"})
        self.assertEqual((self.out / "c1.json").read_text(encoding="utf-8"),
                         json.dumps(chunks["c1"], ensure_ascii=False))

    def test_identical_rerun_and_appending_are_allowed(self):
        self.prep("--fused", str(self.a), "--chunk-dir", str(self.out), "--chunk-size", "2")
        code, _, _ = self.prep("--fused", str(self.a), "--chunk-dir", str(self.out),
                               "--chunk-size", "2")
        self.assertEqual(code, 0)
        code, _, _ = self.prep("--fused", str(self.a), "--fused", str(self.b),
                               "--chunk-dir", str(self.out), "--chunk-size", "2")
        self.assertEqual(code, 0)
        self.assertEqual([[i["sid"] for i in c] for c in self.chunks().values()],
                         [["f1", "f2"], ["f3", "f4"]])

    def test_rewriting_chunks_a_council_may_be_judging_is_refused(self):
        self.prep("--fused", str(self.a), "--chunk-dir", str(self.out), "--chunk-size", "2")
        before = {p.name: p.read_bytes() for p in self.out.glob("*.json")}
        code, _, err = self.prep("--fused", str(self.b), "--chunk-dir", str(self.out),
                                 "--chunk-size", "2")
        self.assertEqual(code, 2)
        self.assertIn("--replace", err)
        self.assertEqual({p.name: p.read_bytes() for p in self.out.glob("*.json")}, before)
        code, _, _ = self.prep("--fused", str(self.b), "--chunk-dir", str(self.out),
                               "--chunk-size", "2", "--replace")
        self.assertEqual(code, 0)
        self.assertEqual([[i["sid"] for i in c] for c in self.chunks().values()],
                         [["f3", "f4"]])

    def test_a_row_without_an_anchor_is_refused(self):
        self.b.write_text(json.dumps({"sentence_id": "f9", "text": "t"}) + "\n",
                          encoding="utf-8")
        code, _, err = self.prep("--fused", str(self.b), "--chunk-dir", str(self.out))
        self.assertEqual(code, 2)
        self.assertIn("f9", err)


if __name__ == "__main__":
    unittest.main()
