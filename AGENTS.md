<!-- Codex mirror of CLAUDE.md: regenerate from CLAUDE.md, don't edit by hand. -->
# 2ol 3alatol (قول ع طول) — Egyptian Arabizi Robustness Research

Research project: the first orthographic-robustness benchmark for Egyptian
Franco-Arabic (Arabizi), plus a matched comparison of input-handling pipelines.
Rebuilt from the ground up on 2026-07-27 (previous fine-tuning-only project is
preserved in git history before that date).

**Motto: great change only needs a few steps.** Small stdlib-only core, five
specialized agents, one reproducible showcase.

## The one-paragraph pitch

Franco/Arabizi has no standard spelling ("a7a"/"aha" are the same word). Models
demonstrably behave differently across spellings of identical content. Nobody
has an instrument to measure that for Egyptian Arabizi — this repo builds it:
same content rendered in multiple attested spelling variants, prediction-
consistency metrics (C@k), and a bake-off of raw-subword vs normalization vs
transliteration vs byte-level pipelines under matched conditions. Master plan:
`docs/FRANCO_RESEARCH_PLAN.md`. Plain-language primer: `docs/BACKGROUND.md`.

## Repository structure

```
├── src/franco/               # stdlib-only core, one module per pipeline step
│   ├── skeleton.py           #   phonemic fingerprint (variant collision key)
│   ├── taxonomy.py           #   31 weighted spelling-habit rules (NATIVE REVIEW block)
│   ├── corpus.py             #   boundary-safe tokenizer for dirty machine text
│   ├── perturb.py            #   seeded compound variant generator (D1 legacy)
│   ├── normalize.py          #   "Franco canon v0" (this repo DEFINES the canon)
│   ├── mine.py               #   attested variant-cluster mining
│   ├── retrieval.py          #   char-trigram TF-IDF baseline
│   ├── metrics.py            #   recall@1, C@k, pairwise agreement
│   └── io_utils.py           #   JSONL + seed-corpus helpers
├── src/franco_eval/          # eval track; MAY use third-party deps (.venv)
│   ├── phenomena.py          #   phenomenon classes (incl. DIPHTHONG, TAA_MARBUTA)
│   ├── tokenizer_metrics.py  #   fragmentation, fertility, separability AUC
│   ├── licensing.py          #   licensed variant generation (council-audited)
│   ├── context.py            #   PPMI context model (trap ranking, no gold)
│   ├── arabic_norm.py        #   Arabic meaning normalization
│   ├── script_id.py          #   Arabic-vs-Latin answer gate
│   ├── meaning_metrics.py    #   under-robust vs over-norm scoring (never combined)
│   ├── respell.py            #   multi-span sentence respeller (Dataset A core)
│   ├── diversity.py          #   farthest-point rendering selection (no near-copies)
│   ├── fusion.py             #   FusioN gates: fuse the prose, freeze the facts
│   ├── fusion_jobs.py        #   frozen plans, checkpoints, quota-pause RunGuard
│   ├── word_split.py         #   v2 held-out split (trap pairs never straddle)
│   ├── substitution_policy.py#   v2 pair-level rulings (native rulings as code)
│   ├── emitters.py           #   v2 row types: normalize / meaning (+ pair gate)
│   ├── anchor_bank.py        #   expansion anchors (trap first, then bare)
│   ├── scenario_bank.py      #   expansion scenarios (ids 0-15 = SCENARIOS)
│   ├── model_clients.py      #   Ollama chat + retries + prompt-sha JSONL cache
│   ├── zai_client.py         #   Z.AI GLM Coding Plan client (reads .env API=)
│   ├── deepseek_client.py    #   DeepSeek client (reads .env DEEPSEEK=)
│   ├── openrouter_client.py  #   OpenRouter client (reads .env OPENROUTER=)
│   └── xcomet_metric.py      #   XCOMET-XL meaning-fidelity; needs .venv-comet, NOT .venv
├── src/visualization/        # figures; theme/loaders/series are stdlib-only
│                             #   (enforced by test), matplotlib only in figures_*
├── experiments/d5_cohere_track/ # the active track
│   ├── e1_variant_probe/     #   run_level{1,3}_*.py + frozen run_level2
│   ├── nilechat_wordset/     #   stimulus pipeline: build_wordset, build_variants,
│   │                         #   mine_traps, merge_council (see its README)
│   ├── training_data/        #   Dataset A (respell) + Dataset B (FusioN) builders
│   │                         #   + prep_council_fusion.py (blind N-arm audit prep)
│   │                         #   + run_arena_judge.py (FusioN-vs-BoN win rate)
│   │                         #   (see its README; council-verified, firewalled)
│   └── results/              #   committed tables + figures + manifest.json
├── experiments/d1_benchmark/ # Direction-1 baseline: run_showcase.py (restored
│                             #   2026-10-17; writes results/showcase.* +
│                             #   data/benchmark/{variant_clusters.tsv,retrieval_v0.jsonl})
├── data/
│   ├── benchmark/            # tracked: pairs.csv + its 3 inputs, retrieval_v0.jsonl
│   ├── stimulus/             # tracked: NileChat probe stimuli (machine-derived,
│   │                         #   provenance on every row) + committed council votes
│   ├── training/             # tracked: fine-tune datasets (a_v1/train,val,stats,
│   │                         #   rejected_for_review.csv for the manual pass)
│   ├── seed/                 # tracked: human-authored Franco conversations (51KB)
│   │                         #   the ONLY corpus; everything is mined from here
│   ├── interim/              # empty; intake point for a real collected corpus
│   ├── external/             # gitignored: third-party corpora pulled from HF
│   │                         #   nilechat_arabizi_egy/ = 890MB machine-derived
│   │                         #   (measurement corpus only, never a benchmark artifact)
│   └── generated/            # gitignored: runtime caches, created on demand
├── results/                  # D1 showcase tables (JSON + MD)
├── tests/                    # unittest suite (stdlib, no pytest dependency)
├── docs/
│   ├── FRANCO_RESEARCH_PLAN.md  # master research plan (10 sections)
│   ├── BACKGROUND.md            # plain-language concept primer
│   ├── LEARNING_PATH.md         # verified videos/papers study plan
│   ├── AGENTS.md                # multi-agent architecture
│   ├── design/                  # dated design specs (incl. the pre-registration)
│   └── directions/01..04*.md    # one doc per research direction
├── .github/workflows/tests.yml  # CI: stdlib suite on Python 3.14, no deps installed
├── LICENSE                   # MIT (code); data/LICENSE.md = CC BY-NC 4.0 (data)
├── CITATION.cff
└── .codex/agents/           # five specialized agents (TOML, xhigh)
```

## The four research directions

| # | Direction | Status | Doc |
|---|---|---|---|
| 1 | Robustness benchmark + pipeline bake-off | **ACTIVE** (showcase has real results) | `docs/directions/01_benchmark.md` |
| 2 | Free Arabizi→Arabic transliterator | planned (arm of D1 first) | `docs/directions/02_transliteration.md` |
| 3 | Variant-invariant embeddings | planned (follow-up, needs D1) | `docs/directions/03_invariant_embeddings.md` |
| 4 | Tokenizer instability study | planned (analysis chapter of D1) | `docs/directions/04_tokenizer_study.md` |

## Agents

Five specialists in `.codex/agents/` (TOML mirrors of `.claude/agents/`),
each `model_reasoning_effort = "xhigh"`
(user preference — never spawn project subagents on other models):
`franco-miner`, `franco-perturber`, `franco-benchmark-builder`,
`franco-evaluator`, `franco-scribe`. Pipeline and rules: `docs/AGENTS.md`.

## Key commands

```bash
# Test suite (588 tests, stdlib unittest — pytest is NOT installed here; in a
# fresh clone 12 dataset-contract tests skip until data/generated/ is built)
python3 -m unittest discover -s tests

# v2 fine-tuning dataset (2026-09-25): held-out split -> whole-corpus respell +
# fusion -> councils -> emit. Evaluate ONLY on word_split.tsv's held-out side.
python3 experiments/d5_cohere_track/training_data/split_words.py
.venv/bin/python experiments/d5_cohere_track/training_data/collect_training_sentences.py \
    --split data/stimulus/nilechat/word_split.tsv --doc-stride 1 --n-sentences 0 \
    --row-group-order hashed --id-prefix n2_ --out data/generated/v2/sentences_nilechat.jsonl
.venv/bin/python experiments/d5_cohere_track/training_data/audit_policy.py      # policy error estimate
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_v2.py  # master records
.venv/bin/python experiments/d5_cohere_track/training_data/prep_council_v2.py   # fused council chunks
.venv/bin/python experiments/d5_cohere_track/training_data/prep_council_pairs_v2.py
python3 experiments/d5_cohere_track/training_data/council/council_jobs.py --council pairs \
    --chunk-dir data/generated/v2/council/pair_chunks --out-dir data/generated/v2/council/pair_votes
#   -> Workflow args for training_data/council/{fused,pair}_council.js (resumable by listing)
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/merge_council.py --fusion-v2
# pair council ran on TWO panels (Opus c0-c17 >=4/5, Sonnet c18-c22 5/5): assemble
# each panel's chunks, merge each at its own bar, combine with a `panel` column
python3 experiments/d5_cohere_track/training_data/council/assemble_panels.py --out-dir <dir> \
    --lenses coherence,identity,plausibility,register,skeptic --panel opus/xhigh=<votes>:c0-c17
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/merge_council.py --dir <dir> --out <tsv> --min-agree 4
python3 experiments/d5_cohere_track/training_data/council/combine_votes.py --out data/stimulus/pairs_v2_votes.tsv \
    --part "opus/xhigh>=4=<tsv>" --part "sonnet/medium>=5=<tsv>"
python3 experiments/d5_cohere_track/training_data/council/compare_councils.py ...  # Sonnet-vs-Opus calibration
.venv/bin/python experiments/d5_cohere_track/training_data/emit_v2.py
# outputs: data/generated/v2/final/{normalize,meaning,mixed}_{train,val}.jsonl
#          (gitignored); tracked: data/training/v2/{DATASHEET.md,stats.json,sample.jsonl,fused_*.jsonl}
# expansion (checkpointed; pauses with exit 75 on a spent quota, resumes):
python3 experiments/d5_cohere_track/training_data/plan_fusion_v2.py --anchors N \
    --scenarios-per-anchor K --exclude-audit data/training/v2/fused_audit.jsonl --out <plan>
experiments/d5_cohere_track/training_data/run_fusion_v2.sh --plan <plan> --out <out> --audit <audit> ...

# D5 Cohere track — see experiments/d5_cohere_track/README.md and results/README.md
python3 experiments/d5_cohere_track/e1_variant_probe/build_pairs.py   # human arm
# NileChat stimulus pipeline (machine-derived arm; deterministic, no seeds)
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_wordset.py
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/mine_traps.py
.venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_variants.py
# runners
.venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level1_tokenizer.py
.venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level3_meaning.py \
    --models "llama3.1:8b,aya-expanse:8b,gemma4:e2b,qwen3.5:9b" --limit 150
# (--dry-run / --cache-stats work with Ollama stopped; level2 is frozen)
# Training data (Dataset A: no model in generation; council-verified)
.venv/bin/python experiments/d5_cohere_track/training_data/collect_training_sentences.py
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_a.py   # then --prep-council / --emit
# Dataset B: FusioN N-arm pilot (see results/fusion_pilot.md for the 2-arm run)
.venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_b.py \
    --smoke 20 --fusor "<model>" --teachers "aya-expanse:8b,gemma4:e2b,llama3.1:8b,jwnder/jais-adaptive:7b,nemotron-3-nano:30b" \
    --out data/generated/fusion_pilot/arm_X.jsonl --audit data/generated/fusion_pilot/arm_X_audit.jsonl
.venv/bin/python experiments/d5_cohere_track/training_data/prep_council_fusion.py \
    --arm data/generated/fusion_pilot/arm_X.jsonl --arm data/generated/fusion_pilot/arm_Y.jsonl
# Arena benchmark: FusioN vs Best-of-N, per arm (pool multiple --audit/--arm pairs in one call)
.venv/bin/python experiments/d5_cohere_track/training_data/run_arena_judge.py \
    --audit data/generated/fusion_pilot/arm_X_audit.jsonl --arm <model_X> \
    --audit data/generated/fusion_pilot/arm_Y_audit.jsonl --arm <model_Y>
# XCOMET-XL meaning-fidelity metric: needs .venv-comet (Python 3.12), NOT .venv -- see hard rule 5
.venv-comet/bin/python -c "
from franco_eval.xcomet_metric import score_batch
print(score_batch([('قائمة (a list)', 'Ana 3ayez 2a2ema el-mowaa3ed')]))"
# figures + reproducibility record
PYTHONPATH=src .venv/bin/python -m visualization.render_all \
    --results-dir experiments/d5_cohere_track/results
.venv/bin/python experiments/d5_cohere_track/write_manifest.py
# outputs: experiments/d5_cohere_track/results/{level*,figures/,manifest.json}
```

## Hard project rules

1. **Benchmark text is human-authored or freshly-collected only.** Machine-
   generated text (the old fusion data, LLM output, nilechat-arabizi-egy) never
   enters `data/benchmark/` as a benchmark artifact. **Amended 2026-08-20:** the
   `d5_cohere_track` may use machine-derived text as a *measurement corpus* or
   *probe stimulus*, provided every table labels provenance and any headline
   claim is also shown on the human-authored arm.
2. **Everything reproducible**: fixed seeds, committed scripts; a number that
   can't be regenerated doesn't get reported.
3. **Taxonomy weights need native-speaker sign-off** (marked block in
   `src/franco/taxonomy.py`).
4. **No pretraining, no RL** — measurement first (see plan §9 scope rules).
   **Amended 2026-08-20:** training small BPE tokenizers as *measurement
   instruments* is allowed (E1 universal-tokenizer arm); training a tokenizer
   for a model we then pretrain is still out of scope.
5. Python 3.14, stdlib only in `src/franco/` (and `src/franco_eval/phenomena.py`,
   so the suite runs without the venv). **Amended 2026-08-20:** heavy deps live
   in `.venv` (gitignored, see `requirements-eval.txt`) and may be imported from
   `src/franco_eval/`; Colab is no longer required — local M4 inference via
   Ollama and HF transformers is the default route. **Amended 2026-08-24:**
   `src/franco_eval/xcomet_metric.py` needs its own `.venv-comet` on
   **Python 3.12**, not 3.14 or `.venv` — `unbabel-comet` pins
   `huggingface-hub<1.0` (conflicts with `.venv`'s `huggingface_hub>=1.28`)
   and its pinned `torchmetrics==0.10.3` imports private `functools`
   internals removed in 3.14. See `requirements-comet.txt`. Run scripts
   using this module with `.venv-comet/bin/python`, never `.venv/bin/python`.

## Environment notes

- **Terminology (set 2026-08-22).** The one Arabic word a model returns in
  the Level 3 probe is a **meaning**, never a "gloss". Data columns follow
  `meaning_en` / `meaning_ar_accept` / `meaning_en_word`. The probe is the
  *in-context word-meaning probe* — not "sentence gloss", which wrongly
  suggested glossing the sentence rather than one word inside it.
- **No credentials, no network, no env vars.** The scrape-and-fuse collection
  era was retired 2026-08-20: `.env`, `.env.example`, and every collector key
  (Gemini / Z.AI / OpenRouter / X) are deleted. Nothing in `src/franco/` or
  `experiments/` reads an environment variable. **Amended 2026-08-24 (third
  amendment):** the Dataset B 3-arm fusion pilot
  (`experiments/d5_cohere_track/training_data/build_dataset_b.py`) may use
  `glm-5.2:cloud` as one fusor arm, which is cloud-routed through Ollama and
  therefore needs network at inference time. This carve-out is scoped to
  that one model, invoked only from that one script — nothing else in
  `src/franco/` or `experiments/` gains network access, and the general
  no-env-var rule is unchanged elsewhere. A `.env` file reappeared in the
  working tree on 2026-08-24 (a Z.AI key, added by the user as a fallback for
  when `glm-5.2:cloud` needs an Ollama subscription this account doesn't
  have); the user explicitly authorized wiring it up once that happened.
  `src/franco_eval/zai_client.py` reads `.env`'s `API=` line — invoked only
  from `build_dataset_b.py` when `--fusor` is prefixed `zai:` (e.g.
  `zai:glm-5.3`), calling the GLM Coding Plan's DEDICATED endpoint
  (`api.z.ai/api/coding/paas/v4`), not the generic pay-as-you-go one
  (`api.z.ai/api/paas/v4`) — the coding plan's quota lives behind that
  specific path; the generic endpoint returns an "insufficient balance"
  error even with a valid, funded coding-plan key. **Amended 2026-08-24
  (fourth amendment):** a second key, `DEEPSEEK=`, was added to the same
  `.env` for the arena-benchmark's external frontier reference (FusioN's
  fused output vs. a single strong model's direct attempt, run once per
  anchor, no fusion). `src/franco_eval/deepseek_client.py` is the only place
  that reads it, calling `api.deepseek.com/chat/completions` with
  `reasoning_effort: "high"` (DeepSeek's V4 models, like glm-5.3, spend a
  token budget on reasoning before answering unless capped — `"low"` was
  tried first and caused a real regression: it made the model drop the
  Latin-script-only constraint under this pipeline's multi-constraint
  prompt, discovered and corrected 2026-08-25). Invoked from
  `run_arena_judge.py` (as one of possibly several `--reference` specs,
  e.g. `deepseek:deepseek-v4-pro`) and from `build_dataset_b.py` (a
  `"deepseek:"` teacher/fusor prefix, same convention as `"zai:"`).
  `.env` is gitignored and untracked; nothing else in `src/franco/` or
  `experiments/` reads it or any other env var, except the amendment
  below. **Amended 2026-08-25 (fifth amendment):** a third key,
  `OPENROUTER=`, was added for teacher-pool diversity after finding
  (`results/fusion_pilot.md`'s "teacher ceiling" section) that a
  frontier-scale fusor still loses to `deepseek-v4-pro` almost every time
  when built from a teacher pool capped at 30B — fusion can only recombine
  what its drafts contain, so closing the gap needs genuinely large
  teachers, not just a large fusor. `src/franco_eval/openrouter_client.py`
  reads it, calling `openrouter.ai/api/v1/chat/completions` (OpenAI-
  compatible). Invoked from `build_dataset_b.py` (an `"openrouter:"`
  prefix, same convention) and `run_arena_judge.py` (same `--reference`
  mechanism). Model strings are OpenRouter's own catalog names passed
  through verbatim (e.g. `nvidia/nemotron-3.5-lightning:free`,
  `minimax/minimax-m3:free`). **`z-ai/glm-5.3-flash` and
  `qwen/qwen3-235b-a22b` were originally planned as OpenRouter models but
  are NOT routed there**: the OpenRouter key hit its own spending limit
  (`"Key limit exceeded (total limit)"`, a cap on the key itself, not a
  code issue — needs the user to raise it or fund the account at
  OpenRouter before those two can be used via that route). Worked around
  rather than blocked on it: `glm-5.3-flash` turned out to already be
  reachable through the existing `zai:` client (untested until tried live
  — the Z.AI docs page didn't list it, but the coding-plan endpoint
  accepts it), so it's dispatched as `zai:glm-5.3-flash`, no OpenRouter
  credential needed. `qwen/qwen3-235b-a22b` was replaced with
  `openrouter:minimax/minimax-m3:free` (MiniMax, genuinely free, a
  different family from everything else in the pool) since it was a paid
  model blocked by the same key limit. Separately, `nvidia/nemotron-3.5-
  lightning:free` has shown three different failure modes across
  attempts (404, timeout, HTTP 200 with a blank/non-JSON body) —
  OpenRouter's free-tier routing for this specific model looks broken on
  their end right now, not a bug in our client; flagged, not yet swapped.
  Also found and fixed while wiring up the
  `glm-4.6` reference: the same mandatory-reasoning-eats-the-budget
  failure `glm-5.3` and `deepseek-v4-pro` already hit, but with a
  different resolution — `glm-4.6`'s actual reasoning need was tiny
  (~35 tokens for a test prompt) and a too-tight `max_tokens` cap (300)
  caused degenerate all-budget-on-reasoning behavior, not a genuine need
  for a large budget; raising `zai_client.py`'s token-budget floor to
  2500 (from 300) fixed it for both `glm-4.6` and `glm-5.3` (costs
  nothing extra for calls that don't need the headroom).
- **Council verdicts are committed files** (`data/stimulus/nilechat/*_votes.tsv`):
  every number regenerates from the TSVs, never from a live model. Invalid rows
  are flagged `council_valid=false`, never deleted. Native-speaker sign-off is
  still pending on all of them.
- **Gated HF repos are replaced by pinned mirrors** in the L1 registry; a
  mirror must reproduce the committed token signature or the run aborts.
- **There is no freshly-collected corpus in this repo.** Every benchmark text
  traces back to `data/seed/*.jsonl` (human-authored, tracked).
  `data/interim/clean_franco_records.jsonl` was a leftover re-export of those
  same seed files — all 492 rows carried `source: local_seed` and the provenance
  filter discarded every one. **Deleted 2026-08-21** after verifying that
  `variant_clusters.tsv`, `retrieval_v0.jsonl` and `results/showcase.*` all
  regenerate byte-identically without it.
- The `INTERIM` branch in `run_showcase.py` is kept deliberately: it is the
  intake point for a real collected corpus if one is ever added, and its
  `source` / `provenance` filters are the live guard enforcing hard rule 1.

## History

- **Public release 2026-10-17.** The public repo starts from one squashed
  commit. The full pre-release history (22 commits, 2026-05-05 → 2026-09-28)
  is not on any GitHub branch; it is kept privately by the author (local
  branches `archive/main` and `archive/v2-dataset`). Every
  commit SHA cited in `changes.md` or `experiments/d5_cohere_track/results/manifest.json`
  refers to it.
  A full-history secret scan (positive-controlled, incl. unreachable objects)
  found no credentials before release.
- Pre-2026-07-27 state (Gemma fine-tuning project, collection/fusion pipeline)
  lives in the archive at commit "chore: checkpoint before ground-up
  restructure to benchmark-first layout".
- `changes.md` is append-only; every session that changes code adds an entry.
- Secrets live only in the gitignored `.env`; `.claude/settings.local.json` and
  `.kilo/` are gitignored in-repo too (not just by a global ignore).
