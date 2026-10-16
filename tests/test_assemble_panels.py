import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assemble_panels",
    ROOT / "experiments" / "d5_cohere_track" / "training_data" / "council" / "assemble_panels.py")
ap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ap)

LENSES = ("identity", "skeptic")


class ParseChunks(unittest.TestCase):
    def test_ranges_and_singles(self):
        self.assertEqual(ap.parse_chunks("c0-c2,c5"), ["c0", "c1", "c2", "c5"])

    def test_bad_spec_fails(self):
        with self.assertRaises(ValueError):
            ap.parse_chunks("c3-c1")


class Assemble(unittest.TestCase):
    def setUp(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        self.root = pathlib.Path(handle.name)
        self.a, self.b = self.root / "opus", self.root / "sonnet"
        for panel, chunks in ((self.a, ("c0", "c1")), (self.b, ("c1", "c2"))):
            panel.mkdir()
            for c in chunks:
                for lens in LENSES:
                    (panel / f"{lens}_{c}.json").write_text(json.dumps([{"panel": panel.name}]))

    def test_each_chunk_comes_from_exactly_one_panel(self):
        out = self.root / "final"
        ap.assemble([("opus", self.a, ["c0", "c1"]), ("sonnet", self.b, ["c2"])],
                    LENSES, out)
        self.assertEqual(json.loads((out / "identity_c1.json").read_text()), [{"panel": "opus"}])
        self.assertEqual(json.loads((out / "skeptic_c2.json").read_text()), [{"panel": "sonnet"}])
        self.assertEqual(json.loads((out / ap.PANEL_MAP).read_text()),
                         {"c0": "opus", "c1": "opus", "c2": "sonnet"})

    def test_a_chunk_claimed_twice_is_refused(self):
        with self.assertRaises(ValueError):
            ap.assemble([("opus", self.a, ["c1"]), ("sonnet", self.b, ["c1"])],
                        LENSES, self.root / "final")

    def test_a_missing_lens_file_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            ap.assemble([("opus", self.a, ["c2"])], LENSES, self.root / "final")


if __name__ == "__main__":
    unittest.main()
