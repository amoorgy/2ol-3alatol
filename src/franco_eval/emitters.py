"""v2 master records -> fine-tuning rows, one pure function per row type.

The master file (build_dataset_v2.py) holds every rendering once, with its
provenance and substitutions. Each training tactic is a function over those
records, so trying a different tactic later is a new function here, never a
regeneration of the data.

  normalize_row  mixed spelling -> base spelling. Teaches SAME; on fused
                 trap sentences the trap word maps to ITSELF, so the same
                 task teaches that a look-alike is not to be merged.
  meaning_row    word in context -> its Arabic, for anchors whose Arabic is
                 council-confirmed. Variants of one word get one answer
                 (SAME); the trap gets a different one (DIFFERENT).

Wording is deliberately NOT the Level 3 probe's (run_level3_meaning.PROMPT_EN
/ PROMPT_ARZ; tests enforce it): the held-out evaluation uses the probe's
templates, so a gain there cannot come from having memorised the prompt.
Templates rotate by a hash of the rendering id -- deterministic, and no one
phrasing becomes a fixed trigger.

Stdlib only.
"""

import hashlib
import re
from typing import Callable, Mapping, Sequence

from franco.corpus import tokens

# Arabic letters and diacritics only -- no space, slash or bracket
ONE_ARABIC_WORD = re.compile(r"[\u0621-\u064A\u064B-\u0652\u0670\u0671]+")

NORMALIZE_TEMPLATES = (
    "Rewrite this Egyptian Franco-Arabic sentence with the usual spelling. "
    "Change only how words are spelled, never which words they are:\n{text}",
    "Fix the spelling of this Arabizi sentence and keep every word as it "
    "is meant:\n{text}",
    "اكتب الجملة دي بالفرانكو بالإملاء المعتاد، من غير ما تغير أي كلمة:\n{text}",
    # widened 2026-09-25: English / Egyptian Arabic script / Arabizi, the
    # sentence both before and after the instruction. The three above stay
    # first and unchanged; _pick's modulus changes, so re-emitting reassigns
    # templates (still deterministic).
    "Normalize the spelling in this Egyptian Arabizi text. Same words, same "
    "order; only the spelling changes:\n{text}",
    "{text}\n\nWrite this Franco sentence again using the common spelling of "
    "each word. Do not add, drop or swap any word.",
    "Spelling cleanup only, Egyptian Franco-Arabic in and out:\n{text}",
    "صحح إملاء الجملة الفرانكو دي وسيب كل كلمة زي ما هي:\n{text}",
    "{text}\n\nاكتب الجملة دي تاني بالفرانكو بالطريقة اللي الناس بتكتب بيها "
    "عادة، نفس الكلام بالظبط.",
    "Salla7 ketabet el gomla di bel franco w seeb kol kelma zay ma heya:\n{text}",
    "{text}\n\nEkteb el gomla di tany bel tari2a el 3adeya, nafs el kalam bel zabt.",
)

MEANING_TEMPLATES = (
    "Egyptian Arabizi: {text}\n"
    "Which Arabic word is \"{word}\" here? Reply with that one Arabic word.",
    "\"{text}\"\n"
    "Give the Arabic-script word that \"{word}\" stands for in this sentence, "
    "nothing else.",
    "في الجملة \"{text}\" -- كلمة \"{word}\" هي أنهي كلمة بالعربي؟ "
    "اكتبها بس.",
    # widened 2026-09-25 (see NORMALIZE_TEMPLATES). Eleven, not ten: _pick
    # hashes the same rendering id for both tasks, so equal lengths would
    # pin a rendering's two rows to the same template slot, and thus the
    # same instruction language, every time.
    "In the Arabizi sentence below, which Arabic word is written as \"{word}\"? "
    "Give only that word, in Arabic script.\n{text}",
    "Sentence: {text}\nWrite \"{word}\" as the Arabic-script word it stands for "
    "here. One word, no extra text.",
    "Egyptian Franco-Arabic text: {text}\nTarget: \"{word}\"\n"
    "Return the target as a single word in Arabic letters.",
    "{text}\nكلمة \"{word}\" هنا تتكتب إزاي بالحروف العربي؟ اكتب الكلمة لوحدها.",
    "حوّل كلمة \"{word}\" اللي في الجملة دي للحروف العربي على حسب معناها فيها، "
    "كلمة واحدة:\n{text}",
    "{text}\n\"{word}\" fel gomla di tetketeb ezay bel 3arabi? Ektebha bas.",
    "{text}\nLaw katabt \"{word}\" bel 7orof el 3arabi hatib2a eh? Kelma wa7da bas.",
    "Gomla bel franco: {text}\nEkteb \"{word}\" bel 7orof el 3arabi, kelma wa7da "
    "men gher shar7.",
)


def _pick(templates: Sequence[str], key: str) -> str:
    index = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return templates[index % len(templates)]


def _meta(record: dict, task: str) -> dict:
    return {"task": task, "rendering_id": record["rendering_id"],
            "source": record.get("source", ""),
            "provenance": record["provenance"]}


def normalize_row(record: dict) -> dict | None:
    """A permutation's pair; a base row has nothing to normalise."""
    if record["role"] != "permutation":
        return None
    prompt = _pick(NORMALIZE_TEMPLATES, record["rendering_id"])
    return {"messages": [
        {"role": "user", "content": prompt.format(text=record["text"])},
        {"role": "assistant", "content": record["base_text"]},
    ], "meta": _meta(record, "normalize")}


def anchor_form(record: dict) -> str | None:
    """The anchor as spelled in THIS rendering, or None if it is absent."""
    anchor = record.get("anchor") or {}
    word = anchor.get("word")
    if not word:
        return None
    for sub in record["substitutions"]:
        if sub["word"] == word:
            return sub["rendering"]
    if any(w == word for w, _o, _c in tokens(record["text"])):
        return word
    return None


def pair_status(record: dict, pair_votes: Mapping[str, bool],
                is_flagged: Callable[[dict], bool]) -> str:
    """"ok" | "pair_unjudged" | "pair_rejected" for one rendering.

    A substitution flagged as needing a judge (substitution_policy
    .needs_judge) admits its row only once the council ACCEPTED that
    word->form pair; one rejected pair blocks the row outright.
    """
    status = "ok"
    for sub in record["substitutions"]:
        if not is_flagged(sub):
            continue
        verdict = pair_votes.get(f"{sub['word']}->{sub['rendering']}")
        if verdict is False:
            return "pair_rejected"
        if verdict is None:
            status = "pair_unjudged"
    return status


def meaning_row(record: dict) -> dict | None:
    """None unless the anchor's confirmed Arabic is ONE word: 10 of the 284
    v2 anchors carry two readings ('شهر / شر', 'مجموعة/مجموعت'), and a
    meaning row must have exactly one right answer."""
    anchor = record.get("anchor") or {}
    if not ONE_ARABIC_WORD.fullmatch(anchor.get("arabic", "")):
        return None
    form = anchor_form(record)
    if form is None:
        return None
    prompt = _pick(MEANING_TEMPLATES, record["rendering_id"])
    return {"messages": [
        {"role": "user", "content": prompt.format(text=record["text"], word=form)},
        {"role": "assistant", "content": anchor["arabic"]},
    ], "meta": {**_meta(record, "meaning"), "anchor": anchor["word"],
                "anchor_form": form}}
