"""The anchor bank for scaling Dataset B fusion: which words, in which order.

A fusion job writes ONE new sentence around a fixed anchor word. The bank is
the priority-ordered list of those words (data/stimulus/v2_anchor_bank.tsv,
built by experiments/d5_cohere_track/training_data/build_anchor_bank.py), so
"run the first N rows" is always a sensible run:

  1. trap anchors (source=trap): both sides of every council-confirmed
     train-side trap pair, in trap_votes order, first occurrence kept --
     exactly what build_dataset_b.anchors_from_traps plans, deduplicated.
     They carry Arabic + gloss, so every new scenario is another MEANING row.
  2. bare anchors (source=bare): train-side respell-pool words with no
     confirmed Arabic. Their sentences feed NORMALIZATION rows only.

Selection of bare anchors (evidence measured 2026-09-25 on the 54,604-word
v2 respell pool and the 1,135 tracked v2 fused sentences):

  Hard exclusions -- held-out words, held-out stimulus forms (policy
  `protected_forms`), every word of trap_votes.tsv (valid or not: a trap
  word is either already a trap anchor or a council-judged look-alike),
  REVIEW_ENGLISH and the benchmark English stoplist, and any word the
  respeller cannot respell mid-sentence (no variant that passes
  substitution_policy.check AND needs no per-pair judge: an anchor whose
  only variants wait on a pair council yields no normalization row).

  Function words are NOT excluded for being function words. Measured, the
  sentences built around them are natural -- the v2 trap anchors include
  ana / 3ashan / 3an / law / ezay / feeh / 2al / dayman, and their fused
  sentences read fine ("El zahma el naharda keteer awi w ana mesh 2ader
  awsal el beet") -- and the exactly-once gate does not fight them: it
  failed 9 of 1,192 v2 jobs, and the 3 failures among those eight anchors
  were script drift (2) and a sentence-initial anchor (1). What
  disqualifies a word is REDUNDANCY: the fused register writes some words
  whatever the anchor is, and build_dataset_v2 respells every pool word of
  a fused sentence, not only its anchor. As a non-anchor respelling span
  (once, not sentence-initial) awi occurs in 16% of the v2 fused
  sentences, kan 13%, yom 9%, naharda 8%, bas 8%, momken 6%, lel 6%: a
  2,500-job bare run respells `kan` ~330 times without ever anchoring it,
  so an anchor slot spent on it buys nothing. Rule: exclude a word whose
  incidental-span rate is >= INCIDENTAL_MAX_RATE (1%: at that rate a
  2,500-job run gives the word >= 25 free contexts against the 1 its own
  job adds). Measured on sentences rather than a word class, the rule
  also removes the scenario vocabulary the fusors reach for (shoghl,
  beit, zahma, bokra) and keeps function words the chat register rarely
  writes (heya 0.1%, laken 0.0%, aktar 0.4%), where an anchor does add
  contexts. 168 words cross 1%; 63 pool words are excluded by it alone.

Order of bare anchors -- every prefix is diverse on two axes:

  frequency  five count bands on a 1-2-5 scale (BAND_LABELS), interleaved
             round-robin high -> low; within a band, a salted sha256 order
             (a uniform sample of the band, not its top edge). The pool is
             heavily skewed (52% of it sits in 200-500), so a straight
             frequency order would be all function words and a hashed
             global order would be half rare tokens (ttw23, t2a, yuta);
             equal band shares let a pilot measure quality PER BAND with
             equal sample sizes, which is where the risk is (the low band
             holds the junk, the news-register MSA and the typos).
  spelling   tier = rank in the word's variant family (mine.py's link
             test: same skeleton, edit distance <= 2 to the family head,
             no prefix/suffix relation -- sa7afy / sahafy / sahafi / s7afy
             are one family -- with leader clustering, see family_ranks).
             Every family head (tier 0) precedes every tier-1 word, and so
             on. Families are computed over the whole gated vocabulary
             INCLUDING excluded words, so a variant of an excluded head is
             deferred too: 3shan (head 3ashan, a trap anchor), ally (head
             elly, held out). Measured on the bank: the first 2,500 bare
             anchors held 270 confirmed variant pairs with mine.py's single
             star per skeleton, and hold 0 (0 in the first 5,000) with
             leaders. The link test over-merges some different words (akl
             under kol, 3elm under 3alam) -- tiers only DEFER, never drop.

Stdlib only.
"""

import collections
import hashlib
from typing import Collection, Iterable, Mapping, Sequence

from franco.corpus import tokens
from franco.mine import edit_distance
from franco.skeleton import skeleton

HEADER = ("word", "arabic", "gloss_en", "context_type", "source", "count",
          "doc_freq", "band", "priority")

# (label, lower bound inclusive), high -> low; the pool gate is count >= 200
BANDS: tuple[tuple[str, int], ...] = (
    ("5k+", 5000), ("2k-5k", 2000), ("1k-2k", 1000), ("500-1k", 500),
    ("200-500", 200))
BAND_LABELS = tuple(label for label, _lo in BANDS)
BELOW_BANDS = "<200"

INCIDENTAL_MAX_RATE = 0.01
ORDER_SALT = "v2-anchor-bank"

TRAP, BARE = "trap", "bare"
TRAP_CONTEXT, BARE_CONTEXT = "trap_contrast", "general"


def band_of(count: int) -> str:
    for label, low in BANDS:
        if count >= low:
            return label
    return BELOW_BANDS


def trap_anchors(votes: Iterable[Mapping[str, str]],
                 train_side: Collection[str] | None
                 ) -> list[tuple[str, str, str]]:
    """[(word, arabic, gloss_en)] -- anchors_from_traps, first occurrence kept.

    A valid pair counts when its word_type is on the train side (a pair
    never straddles the split); both sides are emitted, word then trap.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in votes:
        if row["valid"] != "true":
            continue
        if train_side is not None and row["word_type"] not in train_side:
            continue
        for word, arabic, gloss in (
                (row["word_type"], row["arabic_word"], row.get("meaning_en_word", "")),
                (row["trap"], row["arabic_trap"], row.get("meaning_en_trap", ""))):
            if word not in seen:
                seen.add(word)
                out.append((word, arabic, gloss))
    return out


def incidental_spans(sentences: Iterable[tuple[str, str]]
                     ) -> tuple[collections.Counter, int]:
    """(word -> sentences where it is a NON-anchor respelling span, n).

    Same span test as build_dataset_v2.fused_spans: the word occurs exactly
    once and not at offset 0 (the respeller never spans a sentence start).
    """
    counts: collections.Counter = collections.Counter()
    n = 0
    for text, anchor in sentences:
        n += 1
        toks = tokens(text)
        occurrences = collections.Counter(w for w, _o, _c in toks)
        counts.update({w for w, offset, _c in toks
                       if w != anchor and offset != 0 and occurrences[w] == 1})
    return counts, n


def incidental_words(counts: Mapping[str, int], n: int, *,
                     max_rate: float = INCIDENTAL_MAX_RATE) -> frozenset[str]:
    """Words the fused register already respells in >= max_rate of sentences."""
    if n <= 0:
        return frozenset()
    return frozenset(w for w, k in counts.items() if k / n >= max_rate)


def _is_variant(word: str, head: str) -> bool:
    """franco.mine.mine_variants' confirmation test against the family head."""
    return (edit_distance(word, head) <= 2
            and not word.startswith(head) and not head.startswith(word)
            and not word.endswith(head) and not head.endswith(word))


def family_ranks(counts: Mapping[str, int]) -> dict[str, int]:
    """word -> rank in its variant family (0 = family head).

    mine_variants' link test, with leader clustering instead of one star per
    skeleton: members of a skeleton group are visited by count (ties
    alphabetical); each joins the most frequent existing head it is a
    confirmed variant of, else becomes a head itself. Skeletons shorter than
    2 never cluster (mine.py skips them too). The star version left
    kameya / kamya / kamiya / keemeya as four heads, because the skeleton
    "km" is headed by `kam`, which the prefix guard keeps apart.
    """
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for word in counts:
        groups[skeleton(word)].append(word)
    ranks: dict[str, int] = {}
    for key, members in groups.items():
        members.sort(key=lambda w: (-counts[w], w))
        heads: list[str] = []
        size: dict[str, int] = {}
        for word in members:
            head = (next((h for h in heads if _is_variant(word, h)), None)
                    if len(key) >= 2 else None)
            if head is None:
                heads.append(word)
                size[word] = 0
                ranks[word] = 0
            else:
                size[head] += 1
                ranks[word] = size[head]
    return ranks


def select_bare(pool: Iterable[str],
                exclusions: Sequence[tuple[str, Collection[str]]]
                ) -> tuple[list[str], collections.Counter]:
    """(kept words sorted, reason -> count); the first matching reason wins."""
    kept: list[str] = []
    reasons: collections.Counter = collections.Counter()
    for word in sorted(set(pool)):
        reason = next((name for name, words in exclusions if word in words), "")
        if reason:
            reasons[reason] += 1
        else:
            kept.append(word)
    return kept, reasons


def _hash_key(salt: str, word: str) -> str:
    return hashlib.sha256(f"{salt}:{word}".encode()).hexdigest()


def _round_robin(queues: Sequence[Sequence[str]]) -> list[str]:
    out: list[str] = []
    for index in range(max((len(q) for q in queues), default=0)):
        out.extend(q[index] for q in queues if index < len(q))
    return out


def order_bare(words: Iterable[str], counts: Mapping[str, int],
               ranks: Mapping[str, int], *, salt: str = ORDER_SALT) -> list[str]:
    """Tier-major (family rank); within a tier, bands round-robin high -> low,
    each band in salted-hash order. Deterministic, input-order independent."""
    by_tier: dict[int, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for word in set(words):
        by_tier[ranks.get(word, 0)][band_of(counts[word])].append(word)
    order: list[str] = []
    for tier in sorted(by_tier):
        bands = by_tier[tier]
        queues = [sorted(bands[label], key=lambda w: _hash_key(salt, w))
                  for label in (*BAND_LABELS, BELOW_BANDS)]
        order.extend(_round_robin(queues))
    return order


def bank_rows(traps: Sequence[tuple[str, str, str]], bare: Sequence[str],
              freq: Mapping[str, tuple[int, int]]) -> list[dict]:
    """Contract rows (HEADER keys), trap rows first, priority 1..N."""
    rows: list[dict] = []
    for word, arabic, gloss in traps:
        rows.append({"word": word, "arabic": arabic, "gloss_en": gloss,
                     "context_type": TRAP_CONTEXT, "source": TRAP})
    for word in bare:
        rows.append({"word": word, "arabic": "", "gloss_en": "",
                     "context_type": BARE_CONTEXT, "source": BARE})
    seen: set[str] = set()
    for priority, row in enumerate(rows, start=1):
        if row["word"] in seen:
            raise ValueError(f"{row['word']!r} would appear twice in the bank")
        seen.add(row["word"])
        count, doc_freq = freq.get(row["word"], (0, 0))
        row.update(count=count, doc_freq=doc_freq, band=band_of(count),
                   priority=priority)
    return rows


def to_tsv(rows: Sequence[Mapping]) -> str:
    lines = ["\t".join(HEADER)]
    for row in rows:
        cells = [str(row[field]) for field in HEADER]
        if any(ch in cell for cell in cells for ch in "\t\r\n"):
            raise ValueError(f"tab or newline inside a field of {row['word']!r}")
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"
