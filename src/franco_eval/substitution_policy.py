"""Which substitutions the v2 training respeller may make: pair-level checks.

licensing.py decides which respellings are safe to GENERATE, and it is shared
with the benchmark stimuli (build_variants.py) -- changing a rule there would
silently change how the committed stimuli regenerate. This layer is
v2-training-only and sits after it.

Why pairs, not rows: a_v1 judged every row with a five-lens council, and the
reject reasons were almost always about ONE substitution (training_a_votes'
`bad_substitutions`), which recurs identically in every sentence it lands in.
At corpus scale no row is individually judged, so the a_v1 review's rulings --
applied there by hand, row by row -- live here as code, checked once per
(word, form, position) and applied everywhere.

`check()` returns "" when a substitution is allowed, else the reason; the
builder records reasons so the policy's effect is auditable, never silent.

Stdlib only.
"""

import csv
import pathlib
import re
from dataclasses import dataclass, field
from typing import Mapping

from franco_eval.licensing import BY_ID, VOWELS, Variant, apply_op

# Three identical vowels in a row: chat emphasis (okayyy). User policy
# 2026-08-24: legitimate sentence-FINALLY, invalid mid-sentence or initially.
ELONGATION = re.compile(r"([aeiou])\1\1")
# The review ruled fake ee-/ii- ARTICLE onsets (el -> eel) -- only e/i. An
# aa- onset is the real a/3 habit at word start (3and -> aand, native ruling
# 2026-09-25), so it must not be caught here.
LONG_ONSET = re.compile(r"^([ei])\1")

# English tokens the a_v1 review caught being Franco-respelled (joint pass
# 2026-08-24). Deliberately a short explicit list: the system dictionary
# (web2) also contains kalema, zaman, kitab, shahid -- real Arabizi -- and
# misses offline/doi, so no dictionary is a usable detector here.
REVIEW_ENGLISH = frozenset({"offline", "mile", "steel", "doi", "biology"})


@dataclass(frozen=True)
class Policy:
    trap_partners: Mapping[str, frozenset[str]] = field(default_factory=dict)
    bad_pairs: frozenset[tuple[str, str]] = frozenset()
    protected_forms: frozenset[str] = frozenset()
    english: frozenset[str] = REVIEW_ENGLISH


def op_ids(variant: Variant) -> tuple[str, ...]:
    return tuple(op for op, _at in variant.path)


def variant_from_rule_path(form: str, rule_path: str) -> Variant:
    """Rebuild a Variant from a record's "OP@pos|OP@pos" rule_path string."""
    steps = (step.partition("@") for step in rule_path.split("|") if step)
    return Variant(form=form, path=tuple((op, int(at)) for op, _, at in steps))


def check(policy: Policy, word: str, variant: Variant, *,
          sentence_final: bool) -> str:
    """"" if `word -> variant.form` may be applied at this position."""
    form = variant.form
    if form in policy.trap_partners.get(word, frozenset()):
        return "known trap: the form is a different word"
    if (word, form) in policy.bad_pairs:
        return "council-named bad substitution"
    if form in policy.protected_forms:
        return "would create a held-out stimulus form"
    if word in policy.english:
        return "English token"
    return (review_ruling(word, variant, sentence_final=sentence_final)
            or class_ruling(word, variant))


def review_ruling(word: str, variant: Variant, *, sentence_final: bool) -> str:
    """The a_v1 review's shape rulings (changes.md 2026-08-24 addenda 2-3)."""
    form = variant.form
    if (ELONGATION.search(form) and not ELONGATION.search(word)
            and not sentence_final):
        return "elongation (3+ vowels) is sentence-final only"
    if ("7_TO_H" in op_ids(variant) and "t7" in word
            and form.count("th") > word.count("th")):
        # mot7araka -> motharaka: t + h fabricates a th digraph (ث)
        return "7_TO_H after t fabricates a th digraph"
    onset = LONG_ONSET.match(form)
    if onset and not word.startswith(onset.group(0)):
        return "fabricated long-vowel onset (el -> eel)"
    return ""


def class_ruling(word: str, variant: Variant) -> str:
    """Native-speaker rulings on the two classes the a_v1 review rejected.

    The review rejected every FLAGGED row of ع-deletion (3_TO_A: n3m -> nam,
    3amr -> amr) and of medial vowel drop (DROP_MEDIAL_VOWEL: 3arefa ->
    3arfa), but council-accepted rows of both stayed in a_v1 (604 and 1,169
    kept substitutions). At corpus scale there is no council per row, so this
    function IS the ruling for every row: deny the whole class, guard it, or
    let it through.

    NATIVE RULING 2026-09-25:
      3_TO_A -- a/3 are interchangeable only at the START of a word. Nobody
        writes 3amr for the name; it is just amr, so 3amr -> amr is fine.
        Inside a word the 3 carries the word: n3m -> nam reads as نام, never
        as نعم. Word length is not the criterion (a_v1's 59%-on-short-words
        was the medial cases), position is. `el3ard` counts as medial: the
        string does not start with the 3.
      DROP_MEDIAL_VOWEL -- allowed (leha -> lha, 3alameya -> 3alamya).
    """
    for op, position in variant.path:
        if op == "3_TO_A" and position != 0:
            return "ع-deletion is word-initial only (n3m -> nam reads as نام)"
    return ""


def needs_judge(word: str, variant: Variant) -> str:
    """Why this ALLOWED substitution still needs a per-pair judge, or "".

    NATIVE RULING 2026-09-25: word-initial ع-deletion (3arf -> arf, 3nd ->
    and) and a vowel dropped beside another vowel (radio -> rado, 3amalia ->
    3amala) read correctly in context, so they are not blocked -- but no rule
    can prove a given pair, so each distinct (word, form) goes to the council
    once and its verdict applies to every row carrying it.

    Positions in `variant.path` index the word as it stood when that op
    fired, so the path is replayed. A step that cannot be replayed is
    flagged rather than guessed at.
    """
    current = word
    for op_id, position in variant.path:
        op = BY_ID.get(op_id)
        match = op.regex.search(current, position) if op else None
        if match is None or match.start() != position:
            return "unreplayable rule path: judge per pair"
        if op_id == "3_TO_A":
            return "word-initial ع-deletion: judge per pair"
        if op_id == "DROP_MEDIAL_VOWEL" and (
                current[position - 1:position] in VOWELS
                or current[position + 1:position + 2] in VOWELS):
            return "vowel dropped beside a vowel: judge per pair"
        current = apply_op(current, op, match)
    return ""


# --- loaders: committed verdict files -> Policy ------------------------------

def parse_bad_substitutions(cell: str) -> list[tuple[str, str, int]]:
    """"gedida->gidda:5 farz->faaarz:2" -> [(word, form, n_judges), ...]"""
    out = []
    for item in cell.split():
        pair, _, count = item.rpartition(":")
        word, _, form = pair.partition("->")
        if word and form:
            out.append((word, form, int(count or 1)))
    return out


def load_policy(*, split: pathlib.Path, stimuli: pathlib.Path,
                trap_votes: pathlib.Path,
                training_votes: pathlib.Path) -> Policy:
    heldout: set[str] = set()
    with split.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["side"] == "heldout":
                heldout.add(row["word"])

    # every form the probe shows for a held-out word, so no train-side
    # respelling can hand the model a held-out stimulus string
    protected = set(heldout)
    with stimuli.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["word_type"] in heldout:
                protected.add(row["rendering"])
                if row["trap_word"]:
                    protected.add(row["trap_word"])

    partners: dict[str, set[str]] = {}
    with trap_votes.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["valid"] == "true":
                partners.setdefault(row["word_type"], set()).add(row["trap"])
                partners.setdefault(row["trap"], set()).add(row["word_type"])

    # a pair is bad when >=2 judges named it, or when it was named in a row
    # the council rejected; single naming in an accepted row is kappa-0.39
    # noise, not a verdict
    bad: set[tuple[str, str]] = set()
    with training_votes.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            for word, form, n in parse_bad_substitutions(
                    row.get("bad_substitutions", "")):
                if n >= 2 or row["valid"] == "false":
                    bad.add((word, form))

    return Policy(
        trap_partners={w: frozenset(p) for w, p in partners.items()},
        bad_pairs=frozenset(bad),
        protected_forms=frozenset(protected),
    )
