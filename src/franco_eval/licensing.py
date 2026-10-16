"""Which respellings of an Arabizi word are safe to generate without gold Arabic.

Mining variants from a corpus does not work: 93% of mined co-skeleton pairs are
orthographically explainable, and that bucket mixes true variants with
different words indistinguishably (`men`~`meen`, من / مين, reads as a clean
VOWEL_LENGTH edit). Generating from a known base word instead makes same-word
identity true by construction -- provided each rewrite is *licensed*.

`phenomena.is_licensed` decides that from the Arabic gold form, which we do not
have for NileChat. Three substitutes, in descending strength:

  reversible      5<->kh, 8<->gh, 4<->sh are safe in BOTH directions with no
                  Arabic at all, because the Latin digraph has essentially one
                  Arabic pre-image. phenomena.DigitMapping already marks these
                  `reversible=True`.
  digit -> Latin  Always sound-preserving: the digit tells us which Arabic
                  letter is meant, so spelling it out loses nothing.
  self-licensed   Latin -> ambiguous digit is unsafe in general (`hena` must
                  never become `7ena`), but safe when the base word ALREADY
                  contains that digit -- its presence is the evidence
                  is_licensed would have read off the Arabic.

Everything else is denied and says why. The table was reviewed by a
five-lens council (data/stimulus/nilechat/rule_votes.tsv) on 2026-08-22; the
guards and weights below carry its verdicts. Collisions are recorded, never dropped:
a rule that is valid in general but wrong for one word is exactly where
normalisation pipelines break, so those cases are worth counting.

Stdlib only.
"""

import re
from dataclasses import dataclass, field
from typing import Sequence

from franco_eval.phenomena import (DEGEMINATION, DIGIT_AMBIG, DIGIT_SAFE,
                                   classify_edge,
                                   DIPHTHONG, FINAL_Y_I, INITIAL_HAMZA,
                                   TAA_MARBUTA, VOWEL_DROP_MED, VOWEL_LENGTH,
                                   VOWEL_SWAP)

ALWAYS = "always"
SELF_LICENSED = "self_licensed"

VOWELS = frozenset("aeiou")


@dataclass(frozen=True)
class Op:
    """One licensed rewrite. `weight` is carried over from taxonomy.RULES."""

    op_id: str
    pattern: str
    replacement: str
    phenomenon: str
    weight: float
    direction: str = ALWAYS
    requires_digit: str = ""   # SELF_LICENSED: base word must contain this
    requires_attestation: bool = False  # result must be a corpus word
    rare: bool = False
    note: str = ""

    @property
    def regex(self) -> re.Pattern[str]:
        return _COMPILED[self.op_id]


LICENSED_OPS: tuple[Op, ...] = (
    # --- reversible digits: safe both ways, no Arabic needed -----------------
    Op("KH_TO_5", r"kh", "5", DIGIT_SAFE, 0.7, note="خ has one Latin digraph"),
    Op("5_TO_KH", r"5", "kh", DIGIT_SAFE, 0.7),
    Op("GH_TO_8", r"gh", "8", DIGIT_SAFE, 0.35,
       note="council: minority habit; weight cut from 0.4"),
    Op("8_TO_GH", r"8", "gh", DIGIT_SAFE, 0.45),
    # --- digit -> Latin: always sound-preserving ----------------------------
    Op("7_TO_H", r"(?<![skgc])7", "h", DIGIT_AMBIG, 0.85,
       note="council: restored taxonomy's lookbehind -- a bare 7 after s/k/g "
            "manufactures sh/kh/gh digraphs (es7ab -> eshab reads as ش)"),
    Op("3_TO_A", r"3", "a", DIGIT_AMBIG, 0.3,
       note="council: guarded against intervocalic position -- sa3a -> saaa "
            "is a pile-up nobody types (see _guard_3_to_a)"),
    Op("INITIAL_2_DROP", r"^2", "", INITIAL_HAMZA, 0.5,
       note="word-initial only; medial 2 is a root consonant. Council: "
            "min-length guard, since on very short words the drop leaves "
            "too little word behind"),
    # --- vowel habits: swaps never fire inside vowel clusters ---------------
    Op("E_TO_I", r"(?<![aeiou])e(?![aeiou])", "i", VOWEL_SWAP, 0.6,
       note="council: bare e/i swap inside ee/ei clusters invents sequences "
            "like 'ie' that nobody types"),
    Op("I_TO_E", r"(?<![aeiou])i(?![aeiou])", "e", VOWEL_SWAP, 0.6),
    Op("O_TO_U", r"(?<![aeiouw])o(?![aeiouw])", "u", VOWEL_SWAP, 0.5,
       note="NATIVE RULING 2026-08-24: u is never written adjacent to w -- "
            "the w already carries the /u/ quality. howa (628,539x) vs huwa "
            "(125x); guwa is unattested entirely. Corpus-wide, ow:uw = 123:1. "
            "(The e/i-by-y analogue is only 22:1 -- iy IS a minority habit -- "
            "so no symmetric guard there without a further ruling.)"),
    Op("U_TO_O", r"(?<![aeiou])u(?![aeiou])", "o", VOWEL_SWAP, 0.5,
       note="u->o beside w moves TOWARD the attested form, so no w-guard here"),
    Op("SHORTEN", r"([aeiou])\1", r"\1", VOWEL_LENGTH, 0.55),
    Op("LENGTHEN", r"([aeiou])", r"\1\1", VOWEL_LENGTH, 0.15,
       note="council (0/5 keep): emphatic lengthening targets the FINAL "
            "vowel, never an arbitrary internal one (see _guard_lengthen)"),
    # --- diphthongs ----------------------------------------------------------
    Op("AY_TO_EI", r"ay(?![aeiou])", "ei", DIPHTHONG, 0.4),
    Op("AY_TO_EE", r"ay(?![aeiou])", "ee", DIPHTHONG, 0.5),
    Op("EI_TO_EE", r"ei(?![aeiou])", "ee", DIPHTHONG, 0.6),
    Op("EE_TO_EI", r"ee", "ei", DIPHTHONG, 0.25),
    Op("AW_TO_OO", r"aw(?=[b-df-hj-np-tv-z])", "oo", DIPHTHONG, 0.4,
       note="council: closed syllables only (lawn -> loon). Word-final aw "
            "must survive -- law (لو) is not loo"),
    Op("OU_TO_OO", r"ou(?![aeiou])", "oo", DIPHTHONG, 0.5),
    # --- word shape ----------------------------------------------------------
    Op("DEGEMINATE", r"([b-df-hj-np-tv-z])\1", r"\1", DEGEMINATION, 0.5),
    Op("FINAL_Y_TO_I", r"y\b", "i", FINAL_Y_I, 0.55),
    Op("FINAL_I_TO_Y", r"(?<=[a-z0-9])i\b", "y", FINAL_Y_I, 0.45),
    Op("AH_TO_A", r"ah\b", "a", TAA_MARBUTA, 0.3, requires_attestation=True,
       note="council (0/5 keep): a final -ah can be the 3ms clitic ـه rather "
            "than ة, and no string test separates them -- so the result must "
            "be corpus-attested to fire"),
    Op("A_TO_AH", r"(?<=[a-z0-9])a\b", "ah", TAA_MARBUTA, 0.12,
       requires_attestation=True,
       note="council: the old self-licence was INERT (requires_digit='' "
            "short-circuited the check, so it ran unconditionally -- ana -> "
            "anah). Attestation is the licence now"),
    # --- medial vowel deletion: bounded, see _medial_vowel_ok ---------------
    Op("DROP_MEDIAL_VOWEL", r"(?<=[a-z0-9])[ei](?=[a-z0-9])", "",
       VOWEL_DROP_MED, 0.35,
       note="council: the habit Egyptians have is skipping the epenthetic "
            "e/i (3arefa -> 3arfa), not deleting a/o/u, and never creating "
            "a 3-consonant pile-up; at most one per variant"),
)

# Rewrites deliberately excluded, with the reason, so the table is auditable.
DENIED: dict[str, str] = {
    "9_TO_S": "9 (ص) is in phenomena.EXCLUDED_DIGITS: not modern Egyptian usage",
    "6_TO_T": "6 (ط) is in phenomena.EXCLUDED_DIGITS",
    "Q_TO_2": "classify_edge cannot explain q/2, so generated pairs would fail "
              "the round-trip guard; and word-initially it would mislabel qaf "
              "as INITIAL_HAMZA. 8 of the 500 selected words contain q",
    "2_TO_Q": "2 may be ء rather than ق; not recoverable without the Arabic",
    "MEDIAL_2_TO_A": "a medial 2 is a root consonant (a2ollak), not a hamza",
    "SH_TO_CH": "francophone habit, not Cairene default",
    "SH_TO_4": "council 3/5 deny: 4=ش is a Gulf/Levantine chat convention; "
               "Egyptians write sh. Generating it manufactures a variant "
               "class no Egyptian corpus contains",
    "4_TO_SH": "symmetric to SH_TO_4, plus a bare 4 in Egyptian text is "
               "almost always the numeral",
    "H_TO_7": "council 0/5 keep: the self-licence was logically INVERTED -- "
              "in a word that already spells ح as 7, any REMAINING h is by "
              "construction ه, so rewriting it to 7 is precisely the merge "
              "the licence existed to prevent (7abaha: the -ha is a clitic)",
    "G_TO_J": "non-Cairene; g is gim in Egyptian",
    "A_TO_3": "Latin->3 is never self-evident: a realises ا/أ and short vowels",
}

_COMPILED: dict[str, re.Pattern[str]] = {op.op_id: re.compile(op.pattern)
                                         for op in LICENSED_OPS}
BY_ID: dict[str, Op] = {op.op_id: op for op in LICENSED_OPS}


@dataclass(frozen=True)
class Variant:
    """One generated respelling, with the full derivation that produced it."""

    form: str
    path: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    phenomena: tuple[str, ...] = field(default_factory=tuple)
    score: float = 1.0
    collision: str = ""        # a different corpus word this form equals
    collision_count: int = 0
    # What classify_edge makes of (base, form). This is the DERIVATION's
    # cheapest explanation, and it is frequently NOT the derivation itself:
    # 3ashan -> 3shan is generated by deleting a medial vowel, but after digit
    # canonicalisation it reads as VOWEL_LENGTH (aashan -> ashan), which is a
    # cheaper and equally valid account. Analyses key on this field, because it
    # is the same function E1 uses to label its edges; `phenomena` is kept as
    # the provenance of how the form was actually produced.
    classes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_edits(self) -> int:
        return len(self.path)


_VOWEL_SET = frozenset("aeiou")


def _guard_3_to_a(word: str, match: re.Match) -> bool:
    """3 -> a only when NOT between two vowels: sa3a -> saaa is junk."""
    before = word[match.start() - 1: match.start()]
    after = word[match.end(): match.end() + 1]
    return not (before in _VOWEL_SET and after in _VOWEL_SET)


def _guard_lengthen(word: str, match: re.Match) -> bool:
    """Emphatic lengthening hits the final vowel only (tab3an -> tab3aan)."""
    return not any(ch in _VOWEL_SET for ch in word[match.end():])


def _guard_initial_2(word: str, match: re.Match) -> bool:
    """Dropping the initial 2 of a very short word leaves too little word."""
    return len(word) >= 4


_POSITION_GUARDS = {
    "3_TO_A": _guard_3_to_a,
    "LENGTHEN": _guard_lengthen,
    "INITIAL_2_DROP": _guard_initial_2,
}


def _medial_vowel_ok(word: str, match: re.Match, path) -> bool:
    """One deletion per variant, never at index 0, never the last vowel.

    Unbounded, this op explains anything -- it is the class that swallows
    Egyptian morphology (e3mel/a3mel, katab/ketab all land in VOWEL_DROP_MED),
    so it has to be the most tightly held rule in the table.
    """
    if any(op_id == "DROP_MEDIAL_VOWEL" for op_id, _at in path):
        return False
    if match.start() == 0:
        return False
    if sum(1 for ch in word if ch in VOWELS) <= 1:
        return False
    # Council: never create a 3-consonant pile-up (fakerni -> fakrni is fine,
    # but a deletion that yields CCC is not an Egyptian habit).
    result = word[:match.start()] + word[match.end():]
    run = 0
    for ch in result:
        run = run + 1 if ch not in VOWELS and not ch.isdigit() else 0
        if run >= 3:
            return False
    return True


def licensed_ops(word: str, *, base: str = "", allow_rare: bool = False,
                 path: Sequence[tuple[str, int]] = (),
                 vocabulary: dict[str, int] | None = None,
                 attest_min: int = 20) -> list[tuple[Op, re.Match]]:
    """Every (op, match) legal on `word`, given the original `base` word.

    Self-licensing is judged against `base`, not against the partially rewritten
    form: 7aga -> haga -> h7ga would otherwise be reachable, inventing a ح that
    the intermediate string no longer shows.
    """
    base = base or word
    out: list[tuple[Op, re.Match]] = []
    for op in LICENSED_OPS:
        if op.rare and not allow_rare:
            continue
        if op.direction == SELF_LICENSED and op.requires_digit \
                and op.requires_digit not in base:
            continue
        for match in op.regex.finditer(word):
            if op.op_id == "DROP_MEDIAL_VOWEL" and not _medial_vowel_ok(word, match, path):
                continue
            guard = _POSITION_GUARDS.get(op.op_id)
            if guard is not None and not guard(word, match):
                continue
            if op.requires_attestation:
                # The result itself must be a corpus word: the only gold-free
                # licence for the taa-marbuta directions (council verdict).
                result = apply_op(word, op, match)
                if vocabulary is None or vocabulary.get(result, 0) < attest_min:
                    continue
            out.append((op, match))
    return out


def apply_op(word: str, op: Op, match: re.Match) -> str:
    return word[:match.start()] + match.expand(op.replacement) + word[match.end():]


def enumerate_variants(word: str, *, max_depth: int = 2, allow_rare: bool = False,
                       vocabulary: dict[str, int] | None = None,
                       collision_min: int = 50,
                       require_converged: bool = True) -> list[Variant]:
    """All distinct forms reachable in <= max_depth licensed ops.

    Deterministic and exhaustive: ops in table order, matches left to right, no
    RNG anywhere, so there is no seed to record or mis-set. Duplicates keep the
    highest-scoring (most plausible) derivation.

    A form that collides with a different corpus word is KEPT and labelled, not
    dropped -- `men` -> `meen` is a valid VOWEL_LENGTH rewrite that lands on
    مين, and those cases are the most diagnostic cell in the experiment.

    `require_converged` drops forms classify_edge cannot explain at all. Depth-2
    stacking can route through an intermediate neither endpoint shows: 3ala ->
    (A_TO_AH) 3alah -> (DROP_MEDIAL_VOWEL) 3alh, which ends in -lh and is not a
    shape anyone types. Roughly 0.8% of generated forms, and dropping them is
    what keeps the SAME arm defensible.
    """
    word = word.strip().lower()
    best: dict[str, Variant] = {word: Variant(word)}
    frontier = [best[word]]
    for _depth in range(max_depth):
        nxt: list[Variant] = []
        for current in frontier:
            for op, match in licensed_ops(current.form, base=word,
                                          allow_rare=allow_rare, path=current.path,
                                          vocabulary=vocabulary):
                form = apply_op(current.form, op, match)
                if form == current.form or form == word:
                    continue
                candidate = Variant(
                    form=form,
                    path=current.path + ((op.op_id, match.start()),),
                    phenomena=current.phenomena + (op.phenomenon,),
                    score=current.score * op.weight,
                )
                previous = best.get(form)
                if previous is None or candidate.score > previous.score:
                    best[form] = candidate
                    nxt.append(candidate)
        frontier = nxt

    del best[word]
    out: list[Variant] = []
    for form, variant in best.items():
        edge = classify_edge(word, form)
        if require_converged and not edge.is_variant:
            continue
        count = vocabulary.get(form, 0) if vocabulary else 0
        out.append(Variant(
            form=variant.form, path=variant.path, phenomena=variant.phenomena,
            score=variant.score,
            collision=form if count >= collision_min else "",
            collision_count=count if count >= collision_min else 0,
            classes=edge.classes,
        ))
    return sorted(out, key=lambda v: (-v.score, v.form))
