import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "run_level1_tokenizer",
    ROOT / "experiments" / "d5_cohere_track" / "e1_variant_probe"
    / "run_level1_tokenizer.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
REGISTRY = _MODULE.REGISTRY

# The four Ollama tags confirmed for Level 3. E1<->E2 one-to-one is asserted,
# not assumed.
E2_MODELS = {"aya-expanse:8b", "gemma4:e2b", "qwen3.5:9b", "llama3.1:8b"}


class Registry(unittest.TestCase):
    def test_names_are_unique(self):
        names = [spec.name for spec in REGISTRY]
        self.assertEqual(len(names), len(set(names)))

    def test_every_mirror_has_a_pinned_signature_or_a_written_reason(self):
        """An untrusted mirror without a pin is an unverifiable stand-in."""
        for spec in REGISTRY:
            if spec.mirror_of:
                with self.subTest(name=spec.name):
                    self.assertTrue(spec.expect_signature or spec.note,
                                    "mirror without pin or explanation")

    def test_e2_models_map_one_to_one_onto_registry_rows(self):
        tags = {spec.ollama for spec in REGISTRY if spec.ollama}
        self.assertEqual(tags, E2_MODELS)

    def test_rows_carrying_an_ollama_tag_are_never_optional(self):
        """Losing the tokenizer of a model under test breaks the E1xE2 join."""
        for spec in REGISTRY:
            if spec.ollama:
                with self.subTest(name=spec.name):
                    self.assertFalse(spec.optional)

    def test_exactly_one_byte_and_one_unigram_control(self):
        families = [spec.family for spec in REGISTRY if spec.group == "control"]
        self.assertEqual(sorted(families), ["byte-level", "unigram-lm"])

    def test_optional_rows_document_why(self):
        for spec in REGISTRY:
            if spec.optional:
                with self.subTest(name=spec.name):
                    self.assertTrue(spec.note)


if __name__ == "__main__":
    unittest.main()
