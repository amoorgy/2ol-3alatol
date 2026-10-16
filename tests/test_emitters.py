import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.emitters import (MEANING_TEMPLATES, NORMALIZE_TEMPLATES, pair_status,
                                  anchor_form, meaning_row, normalize_row)
import difflib
import re
import string

from franco_eval.emitters import _pick
from franco_eval.script_id import arabic_ratio

PERMUTATION = {
    "sentence_id": "fs_meen_03", "rendering_id": "fs_meen_03_p1",
    "role": "permutation", "text": "msh 3arf miin elly etasal",
    "base_text": "mesh 3aref meen elly etasal",
    "substitutions": [
        {"word": "mesh", "rendering": "msh", "start": 0, "end": 3},
        {"word": "3aref", "rendering": "3arf", "start": 4, "end": 8},
        {"word": "meen", "rendering": "miin", "start": 9, "end": 13},
    ],
    "split": "train", "provenance": "fusion_synthetic", "source": "fusion",
    "anchor": {"word": "meen", "arabic": "مين", "gloss_en": "who",
               "context_type": "trap_contrast"},
}
BASE = {**PERMUTATION, "rendering_id": "fs_meen_03_p0", "role": "base",
        "text": PERMUTATION["base_text"], "substitutions": []}


class Normalize(unittest.TestCase):
    def test_permutation_maps_rendering_to_base(self):
        row = normalize_row(PERMUTATION)
        user, assistant = row["messages"]
        self.assertEqual(user["role"], "user")
        self.assertIn(PERMUTATION["text"], user["content"])
        self.assertEqual(assistant, {"role": "assistant",
                                     "content": PERMUTATION["base_text"]})
        self.assertEqual(row["meta"]["task"], "normalize")

    def test_base_row_yields_no_pair(self):
        self.assertIsNone(normalize_row(BASE))

    def test_template_choice_is_deterministic(self):
        self.assertEqual(normalize_row(PERMUTATION), normalize_row(PERMUTATION))


class Meaning(unittest.TestCase):
    def test_uses_the_anchor_as_rendered_in_this_row(self):
        self.assertEqual(anchor_form(PERMUTATION), "miin")
        self.assertEqual(anchor_form(BASE), "meen")

    def test_answer_is_the_anchor_arabic(self):
        row = meaning_row(PERMUTATION)
        self.assertIn('"miin"', row["messages"][0]["content"])
        self.assertEqual(row["messages"][1]["content"], "مين")
        self.assertEqual(row["meta"]["task"], "meaning")

    def test_no_row_without_a_confirmed_arabic(self):
        bare = {**PERMUTATION, "anchor": {**PERMUTATION["anchor"], "arabic": ""}}
        self.assertIsNone(meaning_row(bare))
        self.assertIsNone(meaning_row({**PERMUTATION, "anchor": None}))

    def test_no_row_when_the_confirmed_arabic_holds_two_answers(self):
        """A meaning row has exactly one right answer; 'شهر / شر' is two."""
        for arabic in ("شهر / شر", "مجموعة/مجموعت", "خدمة (خدمت)"):
            with self.subTest(arabic=arabic):
                bad = {**PERMUTATION, "anchor": {**PERMUTATION["anchor"], "arabic": arabic}}
                self.assertIsNone(meaning_row(bad))

    def test_diacritics_are_still_one_word(self):
        voweled = {**PERMUTATION, "anchor": {**PERMUTATION["anchor"], "arabic": "عَرَضَ"}}
        self.assertIsNotNone(meaning_row(voweled))

    def test_no_row_when_the_anchor_is_not_in_the_text(self):
        lost = {**BASE, "text": "mesh 3aref elly etasal"}
        self.assertIsNone(meaning_row(lost))


class PairGate(unittest.TestCase):
    """Rows carrying a judge-flagged pair enter only once that pair is
    accepted; unjudged is held back, never admitted."""

    FLAGGED = staticmethod(lambda sub: sub["word"] == "3aref")

    def test_row_without_flagged_pairs_passes(self):
        self.assertEqual(pair_status(PERMUTATION, {}, lambda sub: False), "ok")

    def test_unjudged_flagged_pair_holds_the_row_back(self):
        self.assertEqual(pair_status(PERMUTATION, {}, self.FLAGGED), "pair_unjudged")

    def test_accepted_pair_passes_and_rejected_pair_blocks(self):
        self.assertEqual(pair_status(PERMUTATION, {"3aref->3arf": True}, self.FLAGGED), "ok")
        self.assertEqual(pair_status(PERMUTATION, {"3aref->3arf": False}, self.FLAGGED),
                         "pair_rejected")


class NotTheProbeWording(unittest.TestCase):
    """The held-out eval uses the Level 3 probe's own prompts; training on
    them would let a gain come from template familiarity."""

    def test_no_training_template_repeats_a_probe_template(self):
        spec = importlib.util.spec_from_file_location(
            "level3", ROOT / "experiments" / "d5_cohere_track"
            / "e1_variant_probe" / "run_level3_meaning.py")
        level3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(level3)
        probe_lines = {line.strip() for t in (level3.PROMPT_EN, level3.PROMPT_ARZ)
                       for line in t.splitlines() if len(line.strip()) > 20}
        for template in (*MEANING_TEMPLATES, *NORMALIZE_TEMPLATES):
            for line in template.splitlines():
                self.assertNotIn(line.strip(), probe_lines)

    def test_no_template_line_is_a_near_copy_of_a_probe_line(self):
        """Exact-line inequality is too weak: a one-word edit of the probe
        prompt is still the probe prompt to a fine-tuned model."""
        spec = importlib.util.spec_from_file_location(
            "level3", ROOT / "experiments" / "d5_cohere_track"
            / "e1_variant_probe" / "run_level3_meaning.py")
        level3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(level3)
        probe_lines = [_shape(line) for t in (level3.PROMPT_EN, level3.PROMPT_ARZ)
                       for line in t.splitlines() if len(line.strip()) > 20]
        for template in (*MEANING_TEMPLATES, *NORMALIZE_TEMPLATES):
            for line in template.splitlines():
                for probe in probe_lines:
                    ratio = difflib.SequenceMatcher(None, _shape(line), probe).ratio()
                    self.assertLess(ratio, PROBE_NEAR_COPY, f"{line!r} ~ {probe!r}")


# The three templates every row used before the widening; they stay first.
ORIGINAL_NORMALIZE = (
    "Rewrite this Egyptian Franco-Arabic sentence with the usual spelling. "
    "Change only how words are spelled, never which words they are:\n{text}",
    "Fix the spelling of this Arabizi sentence and keep every word as it "
    "is meant:\n{text}",
    "اكتب الجملة دي بالفرانكو بالإملاء المعتاد، من غير ما تغير أي كلمة:\n{text}",
)
ORIGINAL_MEANING = (
    "Egyptian Arabizi: {text}\n"
    "Which Arabic word is \"{word}\" here? Reply with that one Arabic word.",
    "\"{text}\"\n"
    "Give the Arabic-script word that \"{word}\" stands for in this sentence, "
    "nothing else.",
    "في الجملة \"{text}\" -- كلمة \"{word}\" هي أنهي كلمة بالعربي؟ "
    "اكتبها بس.",
)
PROBE_NEAR_COPY = 0.65  # max observed 0.54 (original meaning template 3)
ARABIZI_TOKEN = re.compile(r"[a-z][2357]|[2357][a-z]")  # a digit used as a letter


def _shape(line: str) -> str:
    return re.sub(r"\{\w+\}", "{}", line.strip().lower())


def _instruction_language(template: str) -> str:
    bare = re.sub(r"\{\w+\}", " ", template)
    if arabic_ratio(bare) > 0.5:
        return "arabic"
    return "arabizi" if ARABIZI_TOKEN.search(bare.lower()) else "english"


class TemplateBank(unittest.TestCase):
    """Hundreds of thousands of rows over 3 phrasings per task would make
    the phrasing a fixed trigger; the bank is widened, mixed-script."""

    SENTENCE = "msh 3arf miin elly etasal"
    WORD = "miin"

    def test_original_three_stay_first_and_unchanged(self):
        self.assertEqual(NORMALIZE_TEMPLATES[:3], ORIGINAL_NORMALIZE)
        self.assertEqual(MEANING_TEMPLATES[:3], ORIGINAL_MEANING)

    def test_about_ten_distinct_templates_per_task(self):
        for templates in (NORMALIZE_TEMPLATES, MEANING_TEMPLATES):
            self.assertGreaterEqual(len(templates), 10)
            self.assertEqual(len(set(templates)), len(templates))

    def test_every_template_formats_and_contains_the_sentence(self):
        cases = ((NORMALIZE_TEMPLATES, {"text"}), (MEANING_TEMPLATES, {"text", "word"}))
        for templates, fields in cases:
            for template in templates:
                with self.subTest(template=template):
                    used = {f for _l, f, _s, _c in string.Formatter().parse(template) if f}
                    self.assertEqual(used, fields)
                    out = template.format(text=self.SENTENCE, word=self.WORD)
                    self.assertIn(self.SENTENCE, out)
                    self.assertNotIn("{", out)
                    if "word" in fields:
                        self.assertIn(f'"{self.WORD}"', out)

    def test_instruction_languages_are_mixed(self):
        for templates in (NORMALIZE_TEMPLATES, MEANING_TEMPLATES):
            langs = [_instruction_language(t) for t in templates]
            with self.subTest(langs=langs):
                self.assertGreaterEqual(langs.count("english"), 3)
                self.assertGreaterEqual(langs.count("arabic"), 3)
                self.assertGreaterEqual(langs.count("arabizi"), 2)

    def test_every_template_is_reachable(self):
        """_pick rotates on a hash of the rendering id; over many ids every
        template must actually get used."""
        for templates in (NORMALIZE_TEMPLATES, MEANING_TEMPLATES):
            picked = {_pick(templates, f"fs_x_{i:02d}_p{j}")
                      for i in range(60) for j in range(4)}
            self.assertEqual(picked, set(templates))


if __name__ == "__main__":
    unittest.main()
