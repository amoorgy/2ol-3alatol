# Council pass 1 — open items (2026-08-22)

Verdicts applied to `src/franco_eval/licensing.py` are in `rule_votes.tsv`.
Two things the council raised were NOT applied, with reasons; both need a
native-speaker call.

## 1. Q_TO_2 denial — the council disagrees 4/5

The generation table denies `q -> 2` because `classify_edge` cannot explain a
q/2 pair (the round-trip guard would reject every generated variant), and
word-initially it would mislabel qaf as INITIAL_HAMZA. Four of five judges
argue the underlying habit is real and central: Cairene qaf IS a glottal stop,
and `2alb/alb/qalb` (قلب) are one word. The correct fix is teaching
`phenomena.py` a QAF class rather than licensing the op blind. Deferred:
changing `classify_edge` mid-experiment invalidates the frozen taxonomy.
8 of the 500 probe words contain q (MSA loans like `taqreer`), so exposure is
small.

## 2. Habits the table does not cover (5 judges, 25 suggestions, deduplicated)

- **Apostrophe digraphs** `7'`=خ, `3'`=غ (older Franco; 79% of NileChat docs
  use them). Deliberately excluded: the boundary-safe tokenizer drops these
  tokens because Command R+ applied them inconsistently.
- **3ms clitic spelling** `-o / -oh / -ou` (kitabo/kitaboh/kitabou) — the
  masculine counterpart of TAA_MARBUTA. Real gap, needs a new phenomenon class.
- **Definite article** `el- / il- / al-` and clitic attachment (`w ana`/`wana`,
  `el bet`/`elbet`) — token-boundary variance, out of scope for a word-level
  probe but real for sentences.
- **Final `-a` ~ `-e`** (keda/kede, 7aga/7age) — distinct from medial e/i swap.
- **`ey` as diphthong spelling** (zey, ezzey, 3eyn) — the commonest Egyptian
  glide spelling; the table only reaches ei/ee/ay.
- **`oo -> ou/ow`** (yoom/youm/yowm) — the inverse of OU_TO_OO is missing, so
  the و nucleus is only ever monophthongised, never diphthongised.
- **Expressive elongation 3+ chars** (la2222, tamaaam) — LENGTHEN only doubles.
- **Sun-letter assimilation** (el shams / esh shams).

None of these invalidates the current stimuli; they bound what the benchmark
can claim to cover, and belong in a v2 of the taxonomy after native review.
