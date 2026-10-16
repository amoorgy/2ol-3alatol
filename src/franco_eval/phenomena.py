"""Phenomenon classes for Egyptian Arabizi spelling variation.

Two spellings of one word differ by a *set* of orthographic phenomena. This
module names those phenomena, decides which ones separate a given pair, and
enforces the licensing asymmetry that makes reverse substitution unsafe.

Stdlib only, so the test suite runs without the eval venv.

Scope boundary, important: `classify_edge` describes *how* two spellings
differ. It does NOT decide whether they are the same word. `kol` and `akel`
reduce to the same consonant skeleton but mean "all/eat" and "food"; no string
method separates them. Same-word judgements come from the Arabic gold form and
native review, never from this module.

=======================================================================
NATIVE-SPEAKER REVIEW (resolved 2026-08-20)
  - 6 (ط) and 9 (ص) excluded: not used in modern Egyptian Arabizi.
  - 4 (ش) retained but flagged rare; no class of its own.
  - kh is confirmed interchangeable with 5: the English digraph carries the
    same sound as خ, so the substitution is reversible in both directions.
Still open: per-cluster Arabic accept-sets (see probe set review file).
=======================================================================
"""

import itertools
import re
from dataclasses import dataclass, field

# --- phenomenon class labels ------------------------------------------------

DIGIT_SAFE = "DIGIT_SAFE"            # digit <-> Latin, reversible without Arabic
DIGIT_AMBIG = "DIGIT_AMBIG"          # digit -> Latin only; reverse needs Arabic
INITIAL_HAMZA = "INITIAL_HAMZA"      # word-initial hamza written as 2 or omitted
FINAL_Y_I = "FINAL_Y_I"              # word-final y vs i
TAA_MARBUTA = "TAA_MARBUTA"          # word-final -a vs -ah (ة)
DEGEMINATION = "DEGEMINATION"        # doubled consonant collapsed
VOWEL_LENGTH = "VOWEL_LENGTH"        # doubled vowel vs single
DIPHTHONG = "DIPHTHONG"              # /aj/ or /aw/ nucleus vs its monophthong
VOWEL_SWAP = "VOWEL_SWAP"            # e/i or o/u interchange
VOWEL_DROP_MED = "VOWEL_DROP_MED"    # exactly one vowel removed
VOWEL_DROP = "VOWEL_DROP"            # two or more vowels removed
CLIPPING = "CLIPPING"                # a glide and its vowel dropped (tayeb/tab)
DIGIT_POSITION = "DIGIT_POSITION"    # same digits, different position (a3mel/3amel)
UNRELATED = "UNRELATED"              # did not converge; not a variant pair

# --- digit inventory --------------------------------------------------------

@dataclass(frozen=True)
class DigitMapping:
    """One Arabizi digit and the Arabic letters it can stand for."""

    digit: str
    arabic: tuple[str, ...]   # Arabic pre-images this digit may represent
    latin: tuple[str, ...]    # Latin realisations of the same sound
    canonical: str            # the Latin form used when canonicalising
    reversible: bool          # is Latin -> digit safe without the Arabic form
    rare: bool = False        # attested in Egyptian usage but infrequent
    note: str = ""

    @property
    def phenomenon(self) -> str:
        return DIGIT_SAFE if self.reversible else DIGIT_AMBIG


DIGITS: tuple[DigitMapping, ...] = (
    DigitMapping(
        "5", ("خ",), ("kh",), "kh", reversible=True,
        note="kh has essentially one Arabic pre-image, so 5<->kh is safe both ways",
    ),
    DigitMapping(
        "8", ("غ",), ("gh",), "gh", reversible=True,
        note="gh has essentially one Arabic pre-image",
    ),
    DigitMapping(
        "4", ("ش",), ("sh",), "sh", reversible=True, rare=True,
        note="reversible like 5/8, but 4 is seldom typed in Egyptian Arabizi",
    ),
    DigitMapping(
        "7", ("ح",), ("h",), "h", reversible=False,
        note="h also realises ه, so h->7 needs the Arabic form",
    ),
    DigitMapping(
        "3", ("ع",), ("aa", "a", ""), "a", reversible=False,
        note="a also realises ا/أ and plain short vowels",
    ),
    DigitMapping(
        "2", ("ء", "أ", "إ", "آ", "ق"), ("", "a", "o", "u"),
        "", reversible=False,
        note="hamza and Cairene qaf both surface as a glottal stop",
    ),
)

BY_DIGIT: dict[str, DigitMapping] = {m.digit: m for m in DIGITS}

# Excluded after native review: 6 (ط), 9 (ص) — not modern Egyptian usage.
EXCLUDED_DIGITS = ("6", "9")


# --- canonicalisation steps -------------------------------------------------

def _canon_digits(word: str) -> str:
    """Rewrite every digit to its canonical Latin form (always licensed)."""
    for mapping in DIGITS:
        word = word.replace(mapping.digit, mapping.canonical)
    return word


def _strip_initial_hamza(word: str) -> str:
    return word[1:] if word[:1] == "2" else word


def _final_y_to_i(word: str) -> str:
    return word[:-1] + "i" if word[-1:] == "y" else word


def _canon_final_ah(word: str) -> str:
    """Fold a final -ah onto -a: shanta/shantah, 7aga/7agah (ة).

    taxonomy.py has carried both directions of this rule and normalize.py has
    folded it in the canon since the beginning, but classify_edge could not
    explain it, so every taa-marbuta pair came back UNRELATED. 98 of the 500
    selected word types end in -a or -ah, so this was the single largest hole
    in the ladder.
    """
    return word[:-2] + "a" if word.endswith("ah") else word


def _collapse_run(word: str, members: frozenset[str]) -> str:
    out: list[str] = []
    for ch in word:
        if out and out[-1] == ch and ch in members:
            continue
        out.append(ch)
    return "".join(out)


def _collapse_double_consonants(word: str) -> str:
    return _collapse_run(word, CONSONANTS)


def _collapse_double_vowels(word: str) -> str:
    return _collapse_run(word, VOWEL_SYMBOLS)


def _canon_vowel_classes(word: str) -> str:
    """Fold e/i to one symbol and o/u to another; a stays distinct."""
    table = {"e": "I", "i": "I", "o": "U", "u": "U"}
    return "".join(table.get(ch, ch) for ch in word)


def _drop_glides(word: str) -> str:
    """Drop the semivowels y and w, modelling syllable clipping.

    Deliberately narrow. `tayeb` and `tab` (طيب/طب) are the same word with a
    glide elided, but a general "one form is a subsequence of the other" rule
    would also merge `2aham` and `7amam` (أهم "most important" vs حمام
    "bathroom"), which the consonant-skeleton miner already wrongly clustered.
    Restricting clipping to glides keeps the true pair and rejects the false one.

    Known limitation: clipping that removes a non-glide consonant is not
    detected, and such pairs will surface as UNRELATED for the reviewer.
    """
    return "".join(ch for ch in word if ch not in ("y", "w"))


_DIPH_FRONT = re.compile(r"[ae][iy](?![aeiou])")   # ai ay ei ey
_DIPH_BACK = re.compile(r"[ao][uw](?![aeiou])")    # au aw ou ow


def _monophthongize(word: str) -> str:
    """Fold a diphthong nucleus onto its monophthong.

    3ein/3een/3en (عين) and beit/beet/bet (بيت) are one Egyptian phonological
    process, but without this step `beit`~`beet` reads as VOWEL_SWAP and
    `beit`~`bet` costs a full vowel strip -- four different explanations for
    one habit.

    The lookahead is load-bearing. In `tayeb` (طيب) the y is the onset of the
    next syllable, not a coda glide, so tayeb/tab must stay CLIPPING. Drop the
    lookahead and it becomes DIPHTHONG+VOWEL_DROP_MED at cost 5, which beats
    CLIPPING's 6 and breaks test_tayeb_tab_converges_via_clipping.
    """
    return _DIPH_BACK.sub("oo", _DIPH_FRONT.sub("ee", word))


def _strip_vowels(word: str) -> str:
    return "".join(ch for ch in word if ch not in VOWEL_SYMBOLS)


def _vowel_count(word: str) -> int:
    return sum(1 for ch in word if ch in VOWEL_SYMBOLS)


VOWEL_SYMBOLS = frozenset("aeiouIU")
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")

# Each step carries a destructiveness cost. Stripping every vowel converges
# almost any pair, so it must not be preferred over a local explanation:
# `aayez`/`ayez` is a vowel-length change, not wholesale vowel loss.
_VOWEL_DROP_MARKER = "__VOWEL_DROP__"
_STEPS: tuple[tuple[str, object, int], ...] = (
    (FINAL_Y_I, _final_y_to_i, 1),
    (TAA_MARBUTA, _canon_final_ah, 1),
    (DEGEMINATION, _collapse_double_consonants, 1),
    # Must precede VOWEL_LENGTH: _apply runs steps in index order, so
    # collapsing `3een`->`3en` first would stop `3ein` ever converging.
    (DIPHTHONG, _monophthongize, 1),
    (VOWEL_LENGTH, _collapse_double_vowels, 1),
    (VOWEL_SWAP, _canon_vowel_classes, 2),
    (CLIPPING, _drop_glides, 2),
    (_VOWEL_DROP_MARKER, _strip_vowels, 4),
)


# --- edge classification ----------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """How two spellings of the same word differ."""

    a: str
    b: str
    classes: tuple[str, ...] = field(default_factory=tuple)
    digits: tuple[str, ...] = field(default_factory=tuple)
    converged: bool = True

    @property
    def n_edits(self) -> int:
        """Difficulty: how many independent phenomena separate the pair."""
        return len(self.classes)

    @property
    def is_variant(self) -> bool:
        return self.converged and bool(self.classes)


def _apply(indices: tuple[int, ...], word: str) -> str:
    for i in indices:
        word = _STEPS[i][1](word)  # type: ignore[operator]
    return word


def _combinations_by_cost() -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Every step subset, cheapest first, so the search can stop at the first hit.

    Tie-break is (cost, size, lexicographic), which is what the previous
    exhaustive "keep the strictly cheaper" scan produced.
    """
    combos = [
        (sum(_STEPS[i][2] for i in combo), combo)
        for size in range(1, len(_STEPS) + 1)
        for combo in itertools.combinations(range(len(_STEPS)), size)
    ]
    combos.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    return tuple(combos)


_COMBOS = _combinations_by_cost()


def classify_edge(a: str, b: str) -> Edge:
    """Name the phenomena separating two spellings of the same word.

    Digit substitutions are read directly off the two strings. The remaining
    phenomena are chosen as the *cheapest* set of canonicalising steps that
    makes the two forms converge, so a pair is explained locally where a local
    explanation exists rather than by discarding all vowels.
    """
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return Edge(a, b, (), (), converged=True)

    fired: list[str] = []
    differing = sorted(
        {c for c in a if c in BY_DIGIT} ^ {c for c in b if c in BY_DIGIT}
    )
    for digit in differing:
        mapping = BY_DIGIT[digit]
        initial = digit == "2" and (a.startswith("2") or b.startswith("2"))
        label = INITIAL_HAMZA if initial else mapping.phenomenon
        if label not in fired:
            fired.append(label)

    x = _strip_initial_hamza(_canon_digits(a))
    y = _strip_initial_hamza(_canon_digits(b))
    if x == y:
        # Two distinct strings can canonicalise to one form when the digits are
        # the same but sit in different places: a3mel (أعمل) and 3amel (عامل)
        # both become "aamel". Position is then the only surviving signal, so
        # this is a phenomenon in its own right rather than an absence of one.
        if not fired:
            fired.append(DIGIT_POSITION)
        return Edge(a, b, tuple(fired), tuple(differing), converged=True)

    vowel_gap = abs(_vowel_count(x) - _vowel_count(y))
    best: tuple[int, tuple[int, ...]] | None = None
    for cost, combo in _COMBOS:
        if _apply(combo, x) == _apply(combo, y):
            best = (cost, combo)
            break
    if best is None:
        return Edge(a, b, (UNRELATED,), tuple(differing), converged=False)

    for i in best[1]:
        label = _STEPS[i][0]
        if label == _VOWEL_DROP_MARKER:
            label = VOWEL_DROP_MED if vowel_gap <= 1 else VOWEL_DROP
        if label not in fired:
            fired.append(label)
    return Edge(a, b, tuple(fired), tuple(differing), converged=True)


# --- licensing --------------------------------------------------------------

def is_licensed(digit: str, arabic: str) -> bool:
    """Is rewriting the Latin form *to* this digit licensed for this word?

    Licensed only when the Arabic form actually contains one of the letters the
    digit stands for. `ahsan` (أحسن) may become `a7san` because ح is present;
    `hena` (هنا) may not become `7ena`, because that h realises ه.
    """
    mapping = BY_DIGIT.get(digit)
    if mapping is None:
        return False
    return any(letter in arabic for letter in mapping.arabic)


def illegal_variants(word: str, arabic: str) -> tuple[tuple[str, str], ...]:
    """Spellings a native speaker would reject, for the over-robustness probe.

    Applies each *unlicensed* Latin-to-digit rewrite. A model that reads these
    as the original word has over-normalised and lost an Arabic distinction.
    Returns (variant, digit) pairs. Review these before use: an "illegal" form
    that people actually type invalidates the probe.
    """
    word = word.strip().lower()
    out: list[tuple[str, str]] = []
    for mapping in DIGITS:
        if mapping.reversible or is_licensed(mapping.digit, arabic):
            continue
        for latin in mapping.latin:
            if not latin or latin not in word:
                continue
            candidate = word.replace(latin, mapping.digit, 1)
            if candidate != word and (candidate, mapping.digit) not in out:
                out.append((candidate, mapping.digit))
            break
    return tuple(out)
