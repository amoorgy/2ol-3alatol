"""FusioN-style synthesis gates for Dataset B: fuse the prose, freeze the facts.

Design split (per arXiv:2510.00931 and its documented failure modes):
  - OPEN-ENDED work -- writing a natural Egyptian sentence -- goes to teachers
    (tau=0.3, N drafts) and a fusor. That is where fusion beats selection.
  - CLOSE-ENDED facts -- WHICH word, its Arabic identity, its gloss -- are
    anchors the models never author. The fusor writes prose AROUND them, and
    the gates below verify the anchors survived and the language did not
    drift (the paper measured a real leak toward the instruction language
    across its eleven languages -- 98.9% language-correctness, attributed to
    the fusion prompt being English-only; Arabic itself was not the weakest
    performer, so the gates below treat this as a general risk, not an
    Arabic-specific one).

Every rejected example carries its reason: an audited dataset is the
substitute for a regenerable one when a cloud fusor is involved.

Stdlib only.
"""

import difflib
from dataclasses import dataclass, field

from franco.corpus import is_clean, word_strings
from franco_eval.script_id import arabic_ratio


@dataclass(frozen=True)
class Anchor:
    """The fixed facts a fused example is built around. Never model-authored."""

    word: str                  # Arabizi base spelling; must appear verbatim
    arabic: str = ""           # from council votes where they exist
    gloss_en: str = ""
    context_type: str = "general"


@dataclass(frozen=True)
class GateReport:
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


ARABIZI_MARKERS = frozenset("23578")
NON_EGYPTIAN_DIGITS = frozenset("469")

# Tokens that betray the sentence is not Egyptian Arabizi at all. Small and
# high-precision on purpose: Franco borrows English nouns freely, but these
# function words only appear when a model drifted into English prose.
ENGLISH_DRIFT = frozenset(
    "the is are was and of this that with for from your please here word "
    "sentence list answer meaning translation".split())

# Scaffolding phrases from our own prompts: a fused sentence echoing them is
# the fusor leaking instructions into data (observed in the first smoke run).
SCAFFOLD_PHRASES = ("fixed fact", "do not change", "one sentence only",
                    "franco only", "8 to 20", "el 7a2ay2", "mamno3 tetghayar",
                    "the following situation")

# A bare anchor word gives a model nothing to write ABOUT -- its only real
# task becomes "find a grammatical slot for this word," and the cheapest
# slot is always the same shape (reported speech, "I want to X"), which is
# exactly the templated-sentence problem the first pilot measured. One
# scenario is drawn per anchor and given to every teacher AND the fusor for
# that anchor (never a different scenario per teacher -- that would give
# them nothing comparable left to fuse). Terse and non-literary on purpose:
# teacher_prompt()'s trailing note already documents models copying a prior
# style example's clauses verbatim, and flowery scenario text risks the
# same echo.
SCENARIOS = (
    "you just got home after a long day and are texting a friend about it",
    "you're planning an errand for tomorrow",
    "you're describing something that happened yesterday",
    "you're asking a friend for help with something",
    "you're complaining about traffic or the weather",
    "you're making weekend plans with family",
    "you're reacting to a piece of news you just heard",
    "you're explaining why you're running late",
    "you're talking about a meal you had recently",
    "you're catching up with someone you haven't seen in a while",
    "you're giving someone directions or advice",
    "you're talking about a TV show or match you watched",
    "you're describing your morning routine",
    "you're apologizing for forgetting something",
    "you're asking about the price of something at a shop",
    "you're telling someone about a problem at work or school",
)


def check_sentence(sentence: str, anchor: Anchor, *,
                   min_tokens: int = 6, max_tokens: int = 26) -> GateReport:
    """Mechanical gates on one fused sentence. All reasons reported, not
    just the first -- the audit trail is the point."""
    reasons: list[str] = []
    words = word_strings(sentence)

    if not (min_tokens <= len(words) <= max_tokens):
        reasons.append(f"length {len(words)} outside [{min_tokens},{max_tokens}]")
    if not is_clean(sentence):
        reasons.append("contains Arabic script, prime digraph, or URL")
    if not sentence.isascii():
        # Franco is ASCII by construction; one accented character means the
        # model drifted into French/another Latin language (observed live).
        reasons.append("non-ASCII characters (Latin-language drift)")
    # 6 (ط) / 9 (ص) are phenomena.EXCLUDED_DIGITS and 4 = ش is the
    # Gulf/Levantine chat habit licensing denies (SH_TO_4). The respeller
    # never produces them, but a fusor can write them into the base text
    # itself (glm-5.3: maba5las4, fe6or -- 9 of 1,192 v2 jobs). Numerals
    # (6, 24) are digits-only tokens and stay allowed.
    foreign = [w for w in words if NON_EGYPTIAN_DIGITS & set(w) and not w.isdigit()]
    if foreign:
        reasons.append(f"non-Egyptian digit letters: {' '.join(foreign[:3])}")
    english = [w for w in words if w in ENGLISH_DRIFT]
    if english:
        reasons.append(f"English drift: {' '.join(english[:4])}")
    lowered = sentence.lower()
    echoed = [phrase for phrase in SCAFFOLD_PHRASES if phrase in lowered]
    if echoed:
        reasons.append(f"prompt scaffolding echoed: {echoed[0]!r}")
    # Language-drift gate: the sentence must be Latin-script Arabizi. Any
    # Arabic-letter content at all means the fusor leaked toward Arabic --
    # the paper's documented failure -- and is caught by is_clean above; the
    # ratio check additionally catches heavy Latin-English drift by requiring
    # the anchor to sit among Arabizi-shaped tokens.
    if arabic_ratio(sentence) > 0.0:
        reasons.append("Arabic script in an Arabizi field")
    count = words.count(anchor.word)
    if count == 0:
        reasons.append(f"anchor word {anchor.word!r} missing (byte-identity)")
    elif count > 1:
        reasons.append(f"anchor word {anchor.word!r} appears {count}x, need 1")
    elif words[0] == anchor.word:
        # sentence-initial targets are excluded corpus-wide (casing confound,
        # and the respeller cannot span offset 0).
        #
        # Compared as a TOKEN, never as a string prefix. `3ayezha ... 3ayez
        # ...` starts with the characters of the anchor `3ayez` while the
        # anchor itself sits mid-sentence, and startswith() rejected it.
        # Egyptian clitics (-ha -ak -ni -lo, see mine_traps.CLITICS) make a
        # stem a prefix of its own inflected forms constantly, so a string
        # test here throws away valid sentences.
        reasons.append("anchor is sentence-initial")
    return GateReport(not reasons, tuple(reasons))


def contribution(drafts: list[str], fused: str) -> dict[str, float]:
    """How much of the fused text each draft supplied (paper's own audit).

    difflib matching-block coverage of the fused string per draft, plus the
    residue no draft covers ("novel"). Values are fractions of the fused
    text; overlaps mean the per-draft numbers can sum past 1.0, so `novel`
    is computed from the union, not the sum.
    """
    covered = [False] * len(fused)
    out: dict[str, float] = {}
    for index, draft in enumerate(drafts):
        matcher = difflib.SequenceMatcher(None, draft, fused, autojunk=False)
        draft_cover = 0
        for block in matcher.get_matching_blocks():
            if block.size < 4:
                continue  # ignore trivial shared fragments
            draft_cover += block.size
            for position in range(block.b, block.b + block.size):
                covered[position] = True
        out[f"draft_{index}"] = round(draft_cover / max(len(fused), 1), 4)
    out["novel"] = round(1.0 - sum(covered) / max(len(fused), 1), 4)
    return out


def _facts(anchor: Anchor) -> str:
    facts = f'"{anchor.word}"'
    if anchor.arabic:
        facts += f" = {anchor.arabic}"
        if anchor.gloss_en:
            facts += f" ({anchor.gloss_en})"
    return facts


def fusion_prompt(anchor: Anchor, drafts: list[str], scenario: str) -> str:
    """English instruction, Franco-only output, one scenario shared with
    the teachers so there is real material to fuse (see SCENARIOS above).

    First smoke run taught us the paper's language-leak cuts both ways: an
    all-Arabizi instruction is too low-resource for teachers/fusors to parse
    AS an instruction (one fusor answered in French, another echoed the
    scaffolding into the data). English instructions with a hard output
    constraint is the same trade the Level-3 probe settled on; the gates
    catch whatever still drifts.

    Also adopts the FusioN paper's own compare-then-fuse structure
    (arXiv:2510.00931, Table 6): weigh each draft's strengths AND
    weaknesses before writing, not just "combine the best parts" -- but
    silently, so the output contract stays "one sentence only" and
    clean_reply() doesn't need to parse a new format.
    """
    numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(drafts))
    return f"""You are given {len(drafts)} draft sentences in Egyptian Franco-Arabic (Arabizi: Egyptian Arabic written in Latin letters and digits like 2 3 5 7 8). All drafts are attempts at the following situation: {scenario}. All drafts use the Egyptian word {_facts(anchor)}.

Drafts:
{numbered}

First, silently compare the drafts: notice what each one gets right (natural phrasing, correct word sense, fits the situation) and what it gets wrong. Then fuse them into ONE new natural Egyptian Arabizi sentence for the same situation, keeping the strengths and avoiding the weaknesses, 8 to 20 words. Hard rules:
- the word {anchor.word} must appear EXACTLY ONCE, spelled exactly "{anchor.word}", and NOT as the first word
- Latin letters and digits only: no Arabic script, no English or French words, no accented characters
- output the sentence only, nothing else"""


def teacher_prompt(anchor: Anchor, scenario: str) -> str:
    """`scenario` is drawn once per anchor and given to every teacher AND
    the fusor for that anchor (see SCENARIOS above) -- a bare anchor word
    gives a model nothing to write about, so it falls back on whichever
    sentence shape most easily fits any word (reported speech, "I want to
    X"), which is exactly the templated-output problem measured in the
    first pilot's naturalness council. Giving every teacher the SAME
    scenario keeps their drafts comparable (still genuinely fusable)
    instead of five unrelated attempts."""
    return f"""In the following situation: {scenario} -- write ONE natural sentence in Egyptian Franco-Arabic (Arabizi: Egyptian Arabic written in Latin letters and digits like 2 3 5 7 8) using the Egyptian word {_facts(anchor)}.

Hard rules:
- the word {anchor.word} must appear EXACTLY ONCE, spelled exactly "{anchor.word}", and NOT as the first word
- 8 to 20 words, everyday Egyptian register
- Latin letters and digits only: no Arabic script, no English words
- output the sentence only, nothing else"""
# NOTE: no style example on purpose -- the second smoke run showed teachers
# copying the example's clauses verbatim into their drafts (example-echo is
# scaffolding leak, same family as instruction-echo, and ungateable since a
# "good" example is by construction valid Arabizi).
