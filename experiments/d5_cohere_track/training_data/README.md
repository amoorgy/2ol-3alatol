# Training data — Dataset A (respell) and Dataset B (synthesize)

> **Current dataset: v2** (299,210 rows, held-out word split, council-verified).
> Card, counts, sized limitations and regeneration commands:
> [`data/training/v2/DATASHEET.md`](../../../data/training/v2/DATASHEET.md). The
> Dataset B fusor pilot this page calls "a separate decision" has since run:
> [`results/fusion_pilot_overview.md`](../results/fusion_pilot_overview.md). The rest
> of this page documents the v1 builders (Dataset A = `data/training/a_v1/`).

Manufactures fine-tuning data from the measured findings of the benchmark.
One principle separates the two datasets: **prose is generated, orthography is
ruled** — no model ever authors a spelling.

## Dataset A — respell real sentences (no model anywhere)

```bash
.venv/bin/python experiments/d5_cohere_track/training_data/collect_training_sentences.py
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_a.py
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_a.py --prep-council
# ... 5-lens Opus council runs over /tmp/council_ta_chunks (workflow) ...
python3 experiments/d5_cohere_track/nilechat_wordset/merge_council.py --training
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_a.py --emit
```

- 3,000 real NileChat sentences (hash-strided across the corpus, ≤3 per doc,
  ≤8 per word) × 1 base + 3 seeded permutations ≈ 12k rows.
- Every substitution is a licensed variant (`franco_eval.licensing`) applied by
  the multi-span respeller (`franco_eval.respell`) — seeded (13), regenerable
  byte-identically. Split 90/10 by SENTENCE at seed 42 (old trainer contract).
- **Contamination firewall**: the 500 benchmark words and human-arm words are
  never respelled; benchmark carrier sentences are excluded outright. Eval
  words may still appear in base spelling (they include function words);
  that residual exposure is documented, not hidden.
- Verification: generation → 5-judge council (≥4/5 to accept; κ reported) →
  `data/training/a_v1/rejected_for_review.csv` for the human pass (fill the
  `reviewer` column with `ok`/`accept` to override) → `--emit`.
- Output: `data/training/a_v1/{train,val}.jsonl` as `{"text": ...}` or
  `--format pairs` chat-JSONL (user: mixed rendering → assistant: base).

## Dataset B — synthesize new sentences, then respell them

```bash
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_b.py \
    --smoke 3 --fusor "qwen3.8:27b-mlx" --teachers "aya-expanse:8b,gemma4:e2b,llama3.1:8b"
```

FusioN-style (arXiv:2510.00931 synthetic-data track): N teacher drafts at
τ=0.3, one fusor merges. The anchors (which word, its Arabic, its gloss —
from council trap votes) are never model-authored; mechanical gates
(`franco_eval.fusion`) enforce anchor byte-identity, ASCII-only Franco,
English-drift and prompt-echo rejection, and a difflib contribution audit is
written per example (`data/generated/training_b_audit.jsonl`).

Accepted sentences land in Dataset A's sentence schema, so the same respeller
and council pipeline apply downstream.

**Status:** plumbing built and smoke-tested. The 3-arm fusor pilot
(glm-5.2:cloud / Falcon-H1-Arabic-34B / qwen3.8:27b-mlx; 50 examples each,
council-audited) is a separate decision: the cloud arm requires amending the
repo's no-network rule, and local-arm smoke output suggests local Egyptian
generation quality is the binding constraint.

Two leak channels the smoke runs caught, now gated or removed: fusors echoing
instruction scaffolding into data, and teachers copying the prompt's style
example verbatim (examples removed — a "good" example is ungateable).
