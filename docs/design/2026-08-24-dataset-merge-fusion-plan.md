# Continuation plan: rendering diversity, fusion merging, and taxonomy v2

Status: written 2026-08-24 for the NEXT working session; Dataset A v1 is
finished (7,438 kept / 963 adjudicated / 3,524 unjudged; see
data/training/a_v1/stats.json and changes.md addenda of 2026-08-24).
This chat continues with the SHORTEN_ATTESTED rule only; everything below is
deferred work in priority order.

## 1. Rendering diversity / anti-overfitting merge (user's 8→2 idea)

Problem: K permutations of one sentence can be near-identical (differ in one
word), teaching the model to parrot the carrier sentence. The user wants few,
maximally-DIFFERENT renderings per sentence instead of many near-duplicates.

Design (deterministic, no model):
- In build_dataset_a.generate(), after sampling candidate renderings,
  select the subset of size K that maximises the minimum pairwise
  edit-distance between renderings (greedy farthest-point over the ~3K
  sampled candidates; seeded, cheap at these sizes).
- Distance = sum over spans of (rendering_a != rendering_b), i.e. count of
  words where the two permutations disagree — NOT char edit distance, so a
  long word swap counts the same as a short one.
- Also dedup across the base: a permutation differing from base in <2 words
  is dropped (it is a near-copy).
- Emits the same fat schema; council verdicts for already-judged rows are
  keyed by rendering_id = sentence + choice hash, so CHANGING the sampler
  invalidates old ids -> regenerate as a_v2 and re-run a (sonnet) council,
  OR apply only to the unjudged 3,524 + future batches. Decide then.

### 1b. Measured verdict on the near-duplicate worry (2026-08-24)

v1 numbers: 22% of kept permutations differ from base by <=1 word; mean 2.5
renderings/sentence after council attrition; 18 exact duplicates found and
purged (empty-substitution permutations; emit guard added). Key finding:
carrier repetition, not rendering similarity, drives memorization risk --
even maximally-different permutations share ~85% of tokens. Recommended v2
shape: INVERTED ALLOCATION -- ~6,000 sentences x 2 renderings (same row count,
half the repetition, double the coverage), with the farthest-point diversity
criterion from section 1 choosing WHICH 2. 1-word-diff renderings are clean
signal in pairs format, waste in text format: decide by fine-tune format.
User verdict on SHORTEN_ATTESTED: skipped for now (value/complexity call;
the three-layer evidence stands in this spec's git history if revisited).

## 2. Fusion phase (Dataset B) — the 3-arm pilot

Already built and smoke-tested (training_data/build_dataset_b.py, gates in
src/franco_eval/fusion.py; two leak channels fixed). Next steps:
- Pull/import models: jais-2-8b GGUF, falcon-h1-arabic-34b GGUF,
  nemotron-3-nano:30b. Cloud arm (glm-5.2:cloud) requires the user to amend
  the no-network rule in CLAUDE.md (third amendment) — get explicit go-ahead.
- Run 50 examples per fusor arm (glm-5.2:cloud / falcon-h1-arabic-34b /
  qwen3.8:27b-mlx) from confirmed-trap anchors; identical gates; ONE sonnet
  council audits all three arms blind (arm hidden from judges).
- Pick the winning fusor by audit pass-rate + naturalness; local smoke says
  generation quality (not gating) is the binding constraint.
- Accepted B sentences flow through the A respeller (they already emit in
  the training_sentences schema) and through the same council pipeline.

## 3. "Merging using the fusion model" (user's phrase)

Two interpretations to confirm with the user before building:
 (a) fusing near-duplicate A renderings into fewer, richer examples — likely
     WRONG tool: section 1's deterministic diversity selection achieves the
     goal without touching orthography; and
 (b) fusing teacher drafts into novel sentences (= Dataset B, section 2).
Recommend (b); revisit (a) only if diversity selection proves insufficient.

## 4. Taxonomy/licensing v2 backlog (from measured failures)

1. `t` missing from 7_TO_H lookbehind (mot7araka→motharaka th-digraph).
2. English-token blocklist for the respeller pool (offline, mile, doi,
   biology got respelled).
3. Length/position guard on 3→a (ع-deletion collisions; the largest reject
   class, 215+38 rows).
4. SHORTEN_ATTESTED (this chat) — once user-verified: implement in
   licensing.py as a vocabulary+context-licensed op family; needs the PPMI
   model artifact cached to data/generated/ so licensing stays fast.
5. Elongation policy as a generation guard: LENGTHEN only word-finally at
   sentence end (currently a post-hoc review rule; should never generate).
6. Council-overlap calibration: seed ~10% of chunks to BOTH opus and sonnet
   so cross-model leniency (17.6% vs 2.5%) is measured on identical rows.

## 5. Then: fine-tune gate

Dataset A v1 (+B pilot output if accepted) → user amends hard rule 4 →
LoRA fine-tune (old Unsloth contract: data/processed format, or
data/training/a_v1 as-is) → re-run Level 1/Level 3 benchmarks on the tuned
model. The benchmark is the before/after instrument; eval words are firewalled
from training so the comparison is clean.
