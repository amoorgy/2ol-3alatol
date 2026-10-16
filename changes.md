# Changes

## 2026-05-04: Initial Project Setup

### What was done
Set up the complete project structure for fine-tuning Gemma 4 E2B-it on Egyptian Franco Arabic.

### Files created
- `CLAUDE.md` - Project documentation and context
- `notebooks/finetune_gemma4_franco_colab.ipynb` - Main Colab notebook (8-step workflow)
- `data/seed/egyptian_franco_conversations.jsonl` - 134 single-turn Egyptian Franco conversations
- `data/seed/egyptian_franco_multiturn.jsonl` - 14 multi-turn Egyptian Franco conversations
- `data/processed/train.jsonl` - Training split (133 examples)
- `data/processed/val.jsonl` - Validation split (15 examples)
- `data/processed/stats.json` - Dataset statistics
- `scripts/prepare_dataset.py` - Dataset validation and train/val splitting

### Key decisions
- Using `FastVisionModel` (Gemma 4 is multimodal) with vision layers frozen for text-only task
- LoRA r=16, alpha=16 for efficiency on free T4 GPU
- 4-bit quantization to fit in 16GB VRAM
- Egyptian-specific Franco data covering: daily chat, food, culture, sports, tech, humor

---

## 2026-05-05: Fix Colab loader crash on `unsloth/gemma-4-E2B-it`

### What was fixed

Resolved the `NotImplementedError: "normal_kernel_cuda" not implemented for 'Byte'` raised at model-load time when the notebook ran on Colab. The crash was triggered by missing tied weights (`k_proj`, `v_proj`, `k_norm`, `lm_head`) in the auto-redirected pre-quantized `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` checkpoint, which `transformers` then tried to randomly initialize via `tensor.normal_()` on `uint8` storage.

### Files modified

- `notebooks/finetune_gemma4_franco_colab.ipynb`
  - **Step 1 (install)**: added `-U` to the unsloth/unsloth_zoo install so the loader fix lands; removed the `transformers==5.5.0` hard pin so unsloth can pull a compatible version.
  - **Step 3 (load model)**: fixed a corrupted first line that had glued `MODEL_NAME = "unsloth/gemma-2-2b-it"` onto `from unsloth import FastLanguageModel` (this was both a SyntaxError and was silently overriding the Step 2 config to Gemma 2). Removed the stale override entirely. Added `use_exact_model_name = True` to `FastLanguageModel.from_pretrained` so the loader skips the broken pre-quantized variant and quantizes the FP16 checkpoint on-the-fly.
  - Cleared stale outputs/error traceback from the load cell.

### Why this works

- `use_exact_model_name=True` prevents Unsloth from rewriting `unsloth/gemma-4-E2B-it` to its `-unsloth-bnb-4bit` sibling. The full-precision repo ships every weight, so there are no missing tensors to randomly init, and `load_in_4bit=True` still gets a 4-bit model via on-the-fly bitsandbytes quantization.
- Upgrading unsloth/unsloth_zoo gets the loader version that re-ties Gemma's weight aliases (`lm_head` ↔ `embed_tokens`) instead of treating them as missing.

---

## 2026-05-05: Migrate to Gemma 4 with FastVisionModel

### What was changed

Updated the notebook to properly use Gemma 4 E2B-it, which requires `FastVisionModel` (not `FastLanguageModel`) since Gemma 4 is a multimodal model. Also pinned `transformers==5.5.0` and added `torchcodec` dependency.

### Files modified

- `notebooks/finetune_gemma4_franco_colab.ipynb`
  - **Step 1 (install)**: pinned `transformers==5.5.0`, added `torchcodec`
  - **Step 2 (config)**: changed model to `unsloth/gemma-4-E2B-it`, added `FINETUNE_ATTENTION` and `FINETUNE_MLP` flags
  - **Step 3 (load)**: switched from `FastLanguageModel` to `FastVisionModel`, returns `processor` instead of `tokenizer`
  - **Step 4 (LoRA)**: uses vision-aware `get_peft_model` with `finetune_vision_layers`, `finetune_language_layers`, etc.
  - **Step 5 (data)**: uses `get_chat_template(processor, "gemma-4")`, messages use `[{"type": "text", "text": ...}]` content format
  - **Step 6 (train)**: uses `UnslothVisionDataCollator` instead of `DataCollatorForSeq2Seq`
  - **Steps 7-8 (test/save)**: all references updated from `tokenizer` to `processor`/`processor.tokenizer`
- `CLAUDE.md` - Updated training config table with new parameters

---

## 2026-05-05: Force transformers 5.5.0 install for Colab

### What was changed

Colab's pre-installed `transformers==5.0.0` was not being replaced by `pip install --no-deps transformers==5.5.0`. Added `--force-reinstall` and a verification cell to catch this.

### Files modified

- `notebooks/finetune_gemma4_franco_colab.ipynb`
  - Changed `!pip install --no-deps transformers==5.5.0` to `!pip install --no-deps --force-reinstall transformers==5.5.0`
  - Added verification cell after install that asserts `transformers >= 5.5`

---

## 2026-05-05: Expanded Dataset with Slang
- **Changes**: Generated and appended ~60 new highly slangy Franco-Arabic conversations (single and multi-turn).
- **Why**: User requested to review training data, add more Egyptian youth slang (e.g., 'yasta', 'gamed zang', 'eshta', 'fokak'), and increase the dataset size.
- **Files modified**: 
  - `data/seed/egyptian_franco_conversations.jsonl`
  - `data/seed/egyptian_franco_multiturn.jsonl`
  - `data/processed/train.jsonl`
  - `data/processed/val.jsonl`
  - `data/processed/stats.json`

---

## 2026-05-17: Franco Data Collection + Fusion Pipeline

### What was added

Implemented a staged, free-first data pipeline for expanding Egyptian Franco/Arabizi chat data.

### Files added

- `scripts/collect_sources.py` - local/free/licensed source collection with Hugging Face hooks
- `scripts/collect_x.py` - official X API recent-search collector with strict cost caps and dry-run mode
- `scripts/clean_franco.py` - text normalization, PII stripping, Arabizi/slang scoring, reject logging
- `scripts/build_conversations.py` - one-turn and multiturn draft builder
- `scripts/fusion_generate.py` - multi-provider teacher generation and heuristic Fusion-of-N style selection
- `scripts/franco_pipeline/` - shared pipeline helpers
- `tests/test_franco_pipeline.py` - unit tests for filtering, PII handling, cost caps, provider fallback, JSONL
- `docs/data_pipeline.md` - command flow and provider environment variables

### Verification

- Ran `python3 -m unittest discover -s tests`
- Ran X collector dry-run with a $0.02 cap
- Ran end-to-end dry run: collect 50 local seed turns, clean, build 20 prompts, fuse 20 mock examples
- Rebuilt processed train/validation splits successfully: 229 total examples, 0 validation errors

---

## 2026-05-18: Scale Fusion Seed and Add Local Env

### What was changed

- Added `.env.example` and local `.env` support for Gemini, Z.AI, OpenRouter, and X API keys.
- Added `scripts/franco_pipeline/env.py` and automatic `.env` loading in X collection and fusion generation.
- Set the default OpenRouter model to `nvidia/nemotron-3-super-120b-a12b:free`.
- Added timestamped logging to collection, cleaning, conversation building, X dry runs, and fusion generation.
- Switched the default Z.AI provider endpoint to the GLM Coding Plan endpoint `https://api.z.ai/api/coding/paas/v4`, with `zai-general` available for normal API credits.
- Added provider error samples to fusion logs so API key/endpoint failures are visible during generation.
- Added transient provider retries, automatic provider disabling after auth/repeated failures, optional `--sleep-between`, and partial-save behavior on interruption.
- Expanded `scripts/build_conversations.py` so it can synthesize thousands of varied Franco prompts from the seed style.
- Improved the mock teacher fallback so offline generation is more varied and prompt-aware.
- Regenerated `data/seed/generated_franco_fusion.jsonl` with 10,000 fused examples.
- Rebuilt processed splits: 10,209 total conversations, 9,188 train, 1,021 validation, 0 errors.

### Verification

- Ran `python3 -m unittest discover -s tests`
- Ran `python3 -m py_compile ...` across all pipeline scripts and tests

---

## 2026-07-23: Research Plan for Egyptian Arabizi Orthographic Robustness

### What was done

Research/scoping session (no code changes). Produced a full internship-oriented research plan for pivoting the project toward an **orthographic-robustness benchmark for Egyptian Arabizi** ("2ol-tany": same content in multiple spelling variants + consistency metrics) with a matched comparison of raw-subword vs. normalization vs. transliteration vs. byte-level pipelines. Two Opus research subagents verified (mid-2026 web state) that no such benchmark or pipeline comparison exists, mapped all available Arabizi datasets (Egyptian gold is LDC-locked; free path = own seed data + reverse-romanized ETC/ASTD + synthetic UBC-NLP set), and compiled an annotated bibliography.

### Files added

- `docs/FRANCO_RESEARCH_PLAN.md` — full 10-section plan: framing, 4 candidate directions, ranking, 8-week execution plan, data/benchmark strategy, modeling arms, evaluation design, risks, and a research-lead pitch.

### Key decisions

- Framing corrected: robustness **measurement** problem first; normalization/transliteration/byte-level are interventions to be compared, not assumed.
- No tokenizer training initially; tokenizer *analysis* (segmentation instability across variants) included as a chapter.
- Benchmark text must come from human-authored/freshly-collected data only — the 10k mock-fusion set is excluded from any benchmark use.
- Egyptian-first, dialect-agnostic schema (NArabizi/TArC extension later).

---

## 2026-07-27: Ground-up restructure — benchmark-first, multi-agent, working showcase

### What was done

Rebuilt the repository from scratch around Direction 1 of the research plan
(orthographic-robustness benchmark for Egyptian Arabizi). The previous
fine-tuning/collection project was checkpoint-committed first
("chore: checkpoint before ground-up restructure to benchmark-first layout")
and then removed from the working tree; human-authored seed data was kept.

### Removed (recoverable from checkpoint commit)

`scripts/` (entire collection/fusion pipeline), `notebooks/finetune_gemma4_franco_colab.ipynb`,
`tests/test_franco_pipeline.py`, `docs/data_pipeline.md`, `data/processed/*`,
`data/seed/generated_franco_fusion.jsonl`. The NileChat eval notebook moved to
`experiments/d1_benchmark/notebooks/`.

### Added

- `src/franco/` — stdlib-only core: `skeleton.py` (phonemic variant
  fingerprint), `taxonomy.py` (~25 weighted spelling-habit rules, native-review
  block), `perturb.py` (seeded compound variant generator), `normalize.py`
  ("Franco canon v0"), `mine.py` (attested variant-cluster mining, affix-guarded),
  `retrieval.py` (char-trigram TF-IDF), `metrics.py` (recall@1, C@k), `io_utils.py`.
- `experiments/d1_benchmark/run_showcase.py` — end-to-end experiment: 5-seed
  mean, fixed query set across intensities, clean control row, leakage
  diagnostic, attested-variant (real noise) evaluation.
- `tests/` — 27 unittest tests (stdlib; pytest not installed on this machine).
- `.claude/agents/` — five specialized agents (franco-miner, franco-perturber,
  franco-benchmark-builder, franco-evaluator, franco-scribe), all
  `model: opus` + `effort: xhigh` per user preference.
- `docs/` — BACKGROUND.md (plain-language primer), LEARNING_PATH.md (verified
  videos/papers, all URLs machine-checked), AGENTS.md (pipeline architecture),
  `directions/01..04` (one doc per research direction).
- `README.md` with reproducible results tables; rewrote `CLAUDE.md`.

### Results (mean over seeds 13/17/29/41/53, 120 fragment queries)

- Attested (real) variation: raw recall@1 0.850 vs normalized 0.871.
- Synthetic sweep: normalization wins every cell (C@3 +5 to +13 points), with
  a reported leakage diagnostic (40%→17% of variants normalize to the exact
  normalized source) because normalize.py partially inverts taxonomy.py.

### Verification

- `python3 -m unittest discover -s tests` → 27 tests OK.
- `python3 experiments/d1_benchmark/run_showcase.py` → tables above.
- Adversarial review workflow (29 Opus agents: 3 reviewers → 26 findings →
  per-finding verification): 7 findings refuted; all confirmed findings fixed,
  including 2 critical ones — (1) normalization-evaluation circularity, fixed
  by adding the attested-variant evaluation + leakage diagnostic; (2) mining
  corpus double-counting seed re-exports (and 8 synthetic fusion rows), fixed
  by provenance filtering + deduplication (true cluster count 60, not 252).
  Also fixed: skeleton merging inflectional paradigms (enta/enty) and
  stripping word-medial glottals; unguarded 7→h rule manufacturing spurious
  digraphs; missing y↔i rules; case-sensitivity hole in perturbation;
  C@k scoring total retrieval failure as "consistent"; silent benchmark row
  drops; unfitted-index crash; vacuous test assertion.

---

## 2026-08-20: Retire the Scrape-and-Fuse Collection Technique

Dead-code / dead-infrastructure cleanup after dropping the agent-driven
scraping approach to data acquisition. The five specialized agents stay; only
the collection plumbing and its residue were removed.

### Deleted

- `scripts/` + `scripts/franco_pipeline/` — empty directory skeletons left
  behind when the collection/fusion scripts were removed on 2026-07-27.
- `.env.example` (tracked) and `.env` (local, held live Gemini / Z.AI /
  OpenRouter keys). All ten variables in them served only the deleted
  collectors and the fusion generator; nothing in `src/franco/` or
  `experiments/` reads an environment variable.
- 8.4 MB of scrape/fusion-era working data, all gitignored and referenced by
  no live code: `data/raw/source_records.jsonl` (530 rows),
  `data/interim/conversation_drafts.jsonl` (10k rows, 5.4 MB),
  `data/interim/generation_prompts.jsonl` (10k rows, 2.4 MB),
  `data/interim/rejected_franco_records.jsonl` (38 rows),
  `data/generated/fusion_outputs.jsonl` (202 machine-generated rows, already
  banned from benchmarks by hard rule 1).
- Dead `.gitignore` entry `{filepath}.tmp` — a literal, un-interpolated
  f-string placeholder that could only ever match a file of that exact name.
  No live code writes a `.tmp` file.

### Corrected documentation

- `CLAUDE.md` "Environment notes" was factually wrong and is rewritten. It
  claimed `data/interim/clean_franco_records.jsonl` "is real collected Franco
  text — the miner uses it when present; don't delete it". Measured: all 492
  rows carry `source: local_seed` and are byte-identical re-exports of
  `data/seed/*.jsonl`; `run_showcase.py`'s provenance filter discards every one
  ("interim: kept 0 of 492"). The file contributes nothing to any result. **No
  freshly-collected external Franco text has ever existed in this repo** — every
  reported number traces to `data/seed/*.jsonl` alone. The file itself was kept.
- `docs/FRANCO_RESEARCH_PLAN.md` gained a dated amendment banner below the
  header: the collection pipeline is retired, `collect_x.py`,
  `collect_sources.py`, `clean_franco.py`, and `fusion_generate.py` live only
  in git history before `ab700b1`, and §4 / §5 / §8 passages that prescribe
  them are stale as written. Annotated rather than rewritten — replacing the
  data-sourcing strategy is a research decision, not a cleanup.
- `.claude/agents/franco-miner.md` — remit trimmed from "collecting/cleaning
  Franco text" to corpus curation + mining; the `data/interim/` "interim
  collected records" source was replaced with an explicit "there is no
  scraping pipeline — do not build one".
- `docs/AGENTS.md` — example invocation referencing "the new collected data"
  replaced with a real one.

### Deliberately NOT changed

- `run_showcase.py`'s `INTERIM` branch and its `"fusion" not in provenance`
  filter look like scraper residue but are a **live contamination guard** —
  `clean_franco_records.jsonl` really does contain 8 fusion-provenance rows,
  and the filter is what enforces hard rule 1. Deleting it would silently let
  machine-generated text into the benchmark.
- `.gitignore` fine-tuning-era entries (`*.gguf`, `*.safetensors`, `wandb/`,
  `outputs/`, `2ol-3alatol-lora/`, `2ol-3alatol-gguf/`) kept as cheap
  insurance against an accidental multi-GB commit.
- Two apparent duplicates were found and rejected as consolidation targets:
  (1) `skeleton._collapse` and `normalize.py`'s `(.)\1+` step are provably
  equivalent (property-tested, 20k random strings, 0 mismatches), but sharing
  a helper would force a cross-import between two modules the architecture
  keeps decoupled 1:1 with pipeline steps — 6 lines is not worth breaking that;
  (2) `perturb._TOKEN_RE` (`[a-zA-Z0-9]+`, case-preserving) vs
  `skeleton._WORD_RE` (`[a-z0-9]+`, post-lowercase) encode different
  requirements, not duplication.
- No dead code in `src/franco/`: all 17 public symbols have live consumers.

### Verification

- `python3 -m unittest discover -s tests` → 27 tests OK, after every step.
- `python3 experiments/d1_benchmark/run_showcase.py` (default args) regenerates
  `results/showcase.{json,md}` and `data/benchmark/*` **byte-for-byte
  identically** — a clean `git status` on those paths was the pass/fail signal
  after each deletion. All results unchanged: corpus still 489 unique
  human-authored messages, 57 attested variant clusters.
- Note for future sessions: `run_showcase.py` writes tracked artifacts, so
  smoke-running it with non-default flags (e.g. `--queries 20`) dirties
  committed results. Use default args, or `git checkout -- results/
  data/benchmark/` afterwards.

---

## 2026-08-20: Cohere-track experiment design + E1 phenomenon instrument

### What was done

Opened a new experiment track (`d5_cohere_track`) of mini-experiments anchored
to published Cohere Labs techniques, for an external deliverable (email to
Marzieh Fadaee, Head of Cohere Labs, target 2026-08-24). Design brainstormed
and specced; first module implemented and tested.

### Added

- `docs/superpowers/specs/2026-08-20-cohere-track-design.md` — 401-line design
  spec covering four experiments: E1 variant probe (One Tokenizer To Rule Them
  All, arXiv:2506.10766), E2 EgyMMLU with a script axis added to Global MMLU's
  culture decomposition, E3 prompt-transformation ladder (The Art of Asking,
  arXiv:2510.19806), E4 correlation with the null pre-registered.
- `src/franco_eval/phenomena.py` — phenomenon classes for Egyptian Arabizi
  spelling variation, plus the **rule-licensing** model: digit→Latin is
  sound-preserving and always licensed; Latin→digit is licensed only when the
  Arabic pre-image contains the digit's letter (`ahsan`→`a7san` yes, because أحسن
  has ح; `hena`→`7ena` no, because هنا has ه). `classify_edge` explains a pair by
  the *cost-minimal* set of canonicalising steps, so `aayez`/`ayez` reads as a
  vowel-length change rather than wholesale vowel loss.
- `tests/test_phenomena.py` — 19 tests; every hand-authored pair from the design
  conversation plus licensing and illegal-variant behaviour.
- `data/benchmark/probe_hand.json` — 11 human-authored clusters, 26 variants.
- `experiments/d5_cohere_track/e1_variant_probe/build_review_file.py` — merges
  hand and mined clusters, classifies all pairwise edges, flags likely
  miner errors, reports phenomenon coverage.
- `data/benchmark/probe_review.tsv` — 68 clusters awaiting native-speaker
  `decision` (KEEP / SPLIT / DROP). This is a hard gate.
- `.venv` + `requirements-eval.txt` for the eval track.

### Native-speaker decisions recorded

Digits `6` (ط) and `9` (ص) excluded — not modern Egyptian Arabizi usage. `4` (ش)
retained but flagged rare, no class of its own. `kh` confirmed interchangeable
with `5`, so `5`↔`kh` is reversible in both directions.

### Findings from the first run

- `DIGIT_SAFE` (5↔kh, 8↔gh) has **7 edges, all hand-authored, zero mined** — the
  digit/digraph phenomenon is invisible in the 330KB attested corpus. Needs either
  the 572k-row NileChat corpus or more hand authoring. Same for `INITIAL_HAMZA`.
- 8 mined edges fail to converge (`UNRELATED`), and 9 clusters are flagged —
  consistent with inspection showing the consonant-skeleton miner merges
  different words (`kol`/`akel` = all/food, `sa7`/`se7a` = correct/health).
  These false clusters are useful as hard negatives once relabelled.
- `nilechat-arabizi-egy` is `gated: auto`, so it needs an HF token plus one
  terms acceptance. `nilechat-arabizi-egy` was transliterated by Cohere's own
  `c4ai-command-r-plus-08-2024`, which the spec turns into a provenance finding.
- EgyMMLU is public: 44 subject configs, dev+test, schema
  `question/context/choices/subject/egy_subject/answer/split/source`.

### CLAUDE.md rule amendments

Rules #1, #4 and #5 amended (measurement corpus vs benchmark; BPE tokenizers as
instruments; `src/franco_eval/` may use third-party deps in `.venv`).

### Verification

`python3 -m unittest discover -s tests` → 46 tests OK.
`build_review_file.py` → 68 clusters, coverage table as above.

### E1 Layer A landed (same session)

- `src/franco_eval/tokenizer_metrics.py` — fragmentation (`1 - jaccard` over token
  ids), fertility, and per-phenomenon aggregation. Takes an `encode` callable so
  the metrics are stdlib-only and testable against a fake tokenizer.
- `tests/test_tokenizer_metrics.py` — 11 tests. Suite now 60, green.
- `experiments/d5_cohere_track/e1_variant_probe/run_fragmentation.py` — loads each
  tokenizer, scores all classified edges, writes `results/cohere_track/e1_fragmentation.{json,md}`.
  Gated repos are skipped with a note instead of crashing.
- Installed `transformers 5.15.1`, `tokenizers 0.22.2` (Python 3.14 wheels exist).

First results, 68 clusters / 150 variants / 97 classified edges:

| tokenizer | tokens/variant | mean frag |
|---|---|---|
| byt5 (byte-level control) | 5.01 | 0.234 |
| mbert | 2.46 | 0.745 |
| xlm-roberta | 2.48 | 0.745 |
| qwen2.5 | 2.51 | 0.710 |

- Byte-level fragments roughly a third as much as subword tokenizers, which is the
  mechanism the One Tokenizer paper's claim rests on.
- For byt5 the spec's prediction holds: `DIGIT_SAFE` (5↔kh, 8↔gh) is the *worst*
  class at 0.539, so the substitution easiest for a human to reverse is hardest for
  a byte-level tokenizer to unify. For subword tokenizers the ordering differs —
  `VOWEL_DROP` is worst — so the inversion is byte-level-specific, not general.
- `same_seq_rate` is 0.000 everywhere and is **near-tautological** (distinct strings
  almost never tokenize identically absent a normalizer). Reported for completeness;
  fragmentation is the informative metric.
- Blocked: `aya-expanse`, `command-r7b-arabic`, `NileChat-3B`, `gemma3`, `llama3.1`
  tokenizers are all gated and need `huggingface-cli login` plus terms acceptance.
- Several classes have small n (FINAL_Y_I 3, CLIPPING 4, INITIAL_HAMZA 5,
  DIGIT_SAFE 7); per-class numbers are indicative only until hand-authoring expands them.

### E1 Layer A completed on production tokenizers (same session)

HF gating cleared by the author. `run_fragmentation.py` gained a tokenizer
*identity fingerprint*: each tokenizer is hashed by how it splits the whole probe
set, identical tokenizers are grouped and scored once, and worked token-by-token
examples are emitted. Twelve registry entries collapse to six unique tokenizers.

Two findings, both verified by fingerprint rather than inferred from equal means:

1. **Cohere ships one tokenizer across its whole family.** `aya-expanse-8b`,
   `aya-expanse-32b`, `c4ai-command-r7b-arabic-02-2025` and `c4ai-command-a-03-2025`
   share fingerprint `413c12494d54` (vocab 255,000, `CohereTokenizer`). The
   Arabic-*specialised* model uses the same tokenizer as the general ones, so its
   specialisation is in the weights. This is the practice their own One Tokenizer
   paper advocates, observable in shipped artifacts.
2. **NileChat-3B did not adapt its tokenizer at all.** Fingerprint `a84440d294b4`
   is identical to `Qwen/Qwen3-8B` and `Qwen/Qwen2.5-7B` (vocab 151,643,
   `Qwen2Tokenizer`). A model built for Egyptian Arabic and Arabizi, pretrained on
   572k rows of Arabizi, inherits an unmodified general-purpose tokenizer, and its
   fragmentation (0.710) is indistinguishable from stock Qwen.

Overall mean fragmentation: aya-101 0.757, gpt-oss-20b 0.732, aya-expanse 0.722,
nilechat-3b / qwen3 0.710, mistral-7b 0.707, byt5 (byte-level control) 0.234.

Mechanism, visible in the worked examples: `akhsam` tokenizes as `ak|hs|am` while
`a5sam` becomes `a|5|sam` — zero shared tokens, and the `kh` digraph spelling خ is
split across two tokens. By contrast `3ayez` (`3|ay|ez`) and `ayez` (`ay|ez`) do
share tokens, which is why DIGIT_AMBIG fragments *less* (0.575) than DIGIT_SAFE
(0.861): dropping `3` leaves the stem intact, whereas `kh`→`5` restructures the word.

Still gated: `gemma-3-4b-it` and `llama-3.1-8b` (`gated=manual`, awaiting approval).

### E1 restructured as a two-level same/different test (same session)

Reframed at the author's direction: one question — *are these two the same word?* —
asked at the tokenizer level and at the inference level, over one shared test set
with two sections (pairs that really are the same, and pairs that only look it).

**Level 1 — the tokenizer cannot tell.** `separability AUC` (probability a
DIFFERENT pair fragments more than a SAME pair) is 0.555–0.624 for every tokenizer
tested, against 0.5 for no information. Fragmentation on real variants is 0.69–0.75
and on different words 0.76–0.80: `mesh`/`msh` is shredded about as much as
`meen`/`men` (مين *who* vs من *from*). Byte-level ByT5 has three times lower
absolute fragmentation (0.219/0.278) and the *same* AUC, so byte-level buys surface
overlap and nothing for this distinction. Structural reason: `vowel_length`,
`vowel_drop_med`, `vowel_swap` and `vowel_drop` all occur on both sides of the set,
so no string-level measure can separate them.

**Level 2 — most models cannot either, and fail in one direction.** Base rate of
SAME in the set is 0.657; every model answers SAME more often than that.

| model | acc. SAME | acc. DIFFERENT | balanced | says-SAME | order-flip |
|---|---|---|---|---|---|
| qwen3.5:0.8b | 1.000 | 0.000 | 0.500 | 1.000 | 0.000 |
| qwen3.8:27b-mlx | 1.000 | 0.014 | 0.507 | 0.995 | 0.010 |
| gemma4:e2b | 0.836 | 0.229 | 0.532 | 0.814 | 0.275 |
| gemma4-26b-a4b-it | 0.978 | 0.529 | **0.753** | 0.804 | 0.078 |

- **Scale does not fix it.** qwen3.8 at 27B behaves like qwen3.5 at 0.8B: both are
  degenerate always-SAME answerers at chance-level balanced accuracy. Family and
  training matter more than size here.
- The 27B Qwen got 1 of 70 DIFFERENT prompts right, and answered SAME to
  `2aham`/`7amam` — أهم *most important* and حمام *bathroom*.
- Only gemma4-26b discriminates, and it still misses 47% of traps: worst on
  number (0.167) and gender (0.333), best on person (0.688) and lexical (0.548).
  Models treat inflectional contrasts as spelling noise.
- **Order-flip needs care:** a low rate for an always-SAME model is a symptom of
  degeneracy, not stability. gemma4:e2b flips on 27.5% of pairs when the two words
  are swapped, which makes its other numbers weak.

Fixed while running: a single transient Ollama failure abandoned a whole model
(this cost the first qwen3.8 run); now retried three times with backoff, giving up
only after 20 failures. Subcategory rows now carry `n`.

Not yet done: no Cohere model at Level 2 (`aya-expanse:8b` not pulled locally), the
DIFFERENT labels are unconfirmed by the native speaker, and gender/number rest on
3 pairs each.

### Cleanup: one path through the code (same session)

Removed dead and contradictory material so the track has a single path:

- **Deleted** `mine_nilechat.py` — over-merged badly (two "clusters" of 169 and 464
  words), was abandoned mid-rewrite, and nothing in any result depended on it.
- **Deleted** `build_review_file.py` and `data/benchmark/probe_review.tsv` —
  superseded by `build_pairs.py` and `data/benchmark/pairs.csv`.
- **Deleted** `results/cohere_track/e1_fragmentation.{json,md}` — stale output from
  the pre-two-section framing whose numbers contradicted `level1_tokenizer.md`.
- **Removed** the legacy multilingual encoders (mbert, xlm-r) and the
  `--include-legacy` flag from the Level 1 registry.
- **Added** `experiments/d5_cohere_track/README.md`: the two levels, how to run
  them, where the test set comes from, what the metrics mean, and what is still open.

Also fixed a CLAUDE.md edit from earlier in the session that silently did nothing
because it was applied without an assertion and its target string did not match.
Test count corrected 46 → 68 and the D5 commands added.

**Author correction applied:** `probe_hand.json` cluster `khasam`/`5asam` corrected
to `khasm`/`5asm` (خصم is one syllable). This propagated through `build_pairs.py`
into Level 1, moving aya-101 separability AUC 0.555 → 0.561 and nilechat 0.609 →
0.606 — confirmation that a data correction reaches the results, and that the
earlier drift was signal rather than nondeterminism.

Verified after cleanup: 68 tests OK; `build_pairs.py` → 102 pairs (67/35);
Level 1 → 6 unique tokenizers, AUC 0.561–0.623; Level 2 → 4 models, all cached.

### data/ narrowed — and a correction

The author judged `raw`/`interim`/`generated`/`seed` to be clutter. Checking each:

- `data/raw/` — held only `.gitkeep`, nothing referenced it. **Deleted.**
- `data/interim/clean_franco_records.jsonl` — 323KB, 492 rows, **every row
  `source='local_seed'`**, i.e. a re-export of the seed files that the showcase's
  provenance filter discards in full. **Deleted**, after verifying empirically
  that `variant_clusters.tsv` (57 clusters), `retrieval_v0.jsonl` (2400 rows) and
  `results/showcase.{json,md}` all regenerate byte-identically without it. The
  directory and the `INTERIM` code path remain as the intake point for a real
  corpus, per the existing note that those provenance filters are the live guard
  on hard rule #1.
- `data/seed/` — **kept.** 51KB, tracked, and the only human-authored Franco text
  in the project; the showcase reports "corpus: 489 unique human-authored
  messages" drawn entirely from it, so everything mined traces back here.
- `data/generated/` — **kept**, gitignored runtime cache created on demand.

**Correction:** an earlier statement in this session claimed `variant_clusters.tsv`
was mined from `data/interim/` and that deleting it would make 57 of the 68 test
clusters unreproducible. That was wrong — it is mined from `data/seed/` only, as
CLAUDE.md already recorded. Verified by regeneration, not by re-reading the doc.

Also fixed: CLAUDE.md advertised `--seed 13` for `run_showcase.py`, which the
script does not accept (its seed list is fixed at 13/17/29/41/53).

---

## 2026-08-21 — NileChat Arabizi-Egypt corpus imported into `data/external/`

### What was actually there before

The question "did we import the NileChat dataset?" had a split answer worth
recording, because the download and the *use* of it had diverged:

- **Raw shard: yes, partially.** `~/.cache/huggingface/hub/datasets--UBC-NLP--
  nilechat-arabizi-egy/` held `train-00000-of-00003.parquet` only (284MiB), pulled
  2026-08-21 00:37 right after the HF login. Shards 1 and 2 were never fetched.
- **Repo artifact: no.** `.gitignore:18-19` reserves
  `data/generated/nilechat_wordfreq.tsv` and `nilechat_candidates.json`; neither
  file has ever existed. The only consumer, `mine_nilechat.py`, was deleted in
  `304ca83` for over-merging.
- **The NileChat *tokenizer* was already imported** and is in the Level 1 results
  (`run_level1_tokenizer.py:47`). That is the 3B model, not the corpus — the two
  were easy to confuse when grepping for "nilechat".

### Imported

Full dataset, all three shards, pinned to revision
`6d32974953e55eacc3f010b19533f01b9aa0ce50`:

```
data/external/nilechat_arabizi_egy/data/train-0000{0,1,2}-of-00003.parquet
```

853MiB on disk, schema `id: string, text: string`. Verified by streaming every
row group of all three files (not just reading the footers): **571,524 docs,
1,320.1M chars**, which matches the 571,524 `num_examples` on the dataset card
and the "572k-row" figure quoted earlier in this file.

`data/external/` is gitignored (`.gitignore:20`, confirmed via `git check-ignore`)
— the corpus bytes are never tracked, so the pinned revision SHA above is the
only reproducibility handle for it.

### Provenance (hard rule 1)

Per the 2026-08-20 amendment this corpus is a **measurement corpus / probe
stimulus only, never a benchmark artifact**:

- **License `cc-by-nc-4.0`** (non-commercial), language tag `arz`.
- Machine-derived twice over: Egyptian Arabic in *Arabic script* sampled from
  `UBC-NLP/fineweb-edu-Egypt` and `UBC-NLP/LHV-Egypt`, then converted to Arabizi.
  Earlier notes in this file record the converter as Cohere's own
  `c4ai-command-r-plus-08-2024`.
- Consequence: the documents are transliterated **educational/web prose**, not
  natively-written social Arabizi. Spelling variation in it reflects the
  transliterating model being inconsistent across documents, not Egyptian
  spelling habits. Usable for *attestation at scale*; not evidence about how
  people spell.

### Measured while scoping (shard 0, exploratory — no artifact committed)

Recorded because it settles a question `changes.md` left open, and because no
mining script was written, so these numbers live only here:

- 2,799,410 distinct word types; 71.3M word tokens; 870,311 types at freq>=3.
- The `DIGIT_SAFE` gap **is** covered at scale: `5` in 128,186 types /1.28M
  tokens, `kh` in 86,143 /1.34M, `8` in 14,854, `gh` in 55,436 — against the 7
  hand-authored, zero-mined edges the 330KB seed corpus could supply.
- Probing co-attested variant pairs by substitution splits the rules exactly along
  the existing `reversible` flag in `phenomena.py:59`:
  - **Phoneme-preserving / `DIGIT_SAFE`** give clean pairs — `kh`↔`5` 7,084 pairs
    (`khatar` 9,890 ~ `5atar` 9,429; `shakhs` ~ `sha5s`), `gh`↔`8` 210,
    `7`↔`h` 10,340 (`e7na` ~ `ehna`).
  - **Phoneme-collapsing / `DIGIT_AMBIG`** yield homographs, not variants —
    `m3` (مع) ~ `ma` (ما), `n2s` (نقص) ~ `nas` (ناس), `ra9m` (رقم) ~ `rasm` (رسم),
    `ma9al` (مقال) ~ `masal` (مثل).
  This is the concrete mechanism behind the `mine_nilechat.py` over-merge: mining
  all substitutions together guarantees it. Any future miner should be restricted
  to the reversible rules.

### Scope

Import only, at the author's request ("clean import into `data/`, I'll clean it up
later"). **No** word-frequency table, **no** candidate clusters, **no** miner —
`.gitignore:18-19` remain unfilled slots.

### Files changed

- `.gitignore` — added `data/external/`
- `CLAUDE.md` — data-dir map now lists `data/external/` with its provenance status

### Note

An empty `{filepath}.tmp` appeared in the repo root at 05:10 during this session —
a hook or proxy writing to an unexpanded `{filepath}` template. 0 bytes, untracked,
left in place for the author to remove.

---

## 2026-08-22 — NileChat-scale variant probe: 500-word stimuli, licensed generation, council, gloss probe, visualization

**Why.** The d5 track's 102-pair probe was too small to defend per-phenomenon
claims. Scaled both experiments onto a 500-word, 2,817-row stimulus set derived
from nilechat-arabizi-egy, upgraded E2 from same/different judgement to
sentence-level gloss consistency, and made the whole track reproducible from a
fresh clone.

**Measured findings that shaped the design (before any code):**
- Orthographic explainability ≠ word identity: 93% of mined co-skeleton pairs
  converge under `classify_edge`, mixing true variants with different words
  (`men`~`meen`). So variants are GENERATED (same-word by construction) and
  traps are MINED from exactly those converged look-alikes.
- 79% of NileChat docs use apostrophe prime-digraphs (`7'` 83k); `skeleton.words`
  splits them into junk types. New boundary-safe tokenizer drops 6.6% of tokens.
- `CohereLabs/aya-expanse-8b` and `UBC-NLP/NileChat-3B` are gated (401): the
  committed level1 results only reproduced from a warm local cache. Fixed with
  ungated mirrors pinned to the committed token signatures; non-optional load
  failures now exit 2 instead of writing a complete-looking file.

**Core library:**
- `src/franco_eval/phenomena.py`: new `DIPHTHONG` class (coda-restricted, cost 1,
  before VOWEL_LENGTH — order is load-bearing) and new `TAA_MARBUTA` class
  (98/500 words end -a/-ah; pairs previously read UNRELATED). `classify_edge`
  search rewritten cheapest-first: 6x faster despite an 8-step ladder.
- `src/franco/taxonomy.py`: 6 diphthong rules; fixed two pre-existing unanchored
  rules (`y→i`, `i→y` fired mid-word: ya3ny→ia3ny). Consequence: perturb output
  changes, so results/showcase.* would not regenerate identically — moot in
  practice because run_showcase.py was already deleted from the working tree.
- New stdlib modules: `franco/corpus.py` (boundary-safe tokenizer, sentence
  splitter, feature detector), `franco_eval/licensing.py` (licensed variant
  ops), `franco_eval/context.py` (PPMI context model; SAME/DIFFERENT AUC 0.873
  calibrated on the human arm), `franco_eval/{arabic_norm,script_id,
  gloss_metrics,model_clients}.py`.

**Council (ultracode workflows, Opus x-high, verdicts committed):**
- Pass 1 (5 lenses × 28 ops) found 3 real bugs: H_TO_7's self-licence was
  logically INVERTED (denied), 7_TO_H dropped taxonomy's anti-digraph
  lookbehind, A_TO_AH's guard was inert. Restricts: no 3→a between vowels,
  LENGTHEN final-vowel-only, vowel swaps never inside clusters, aw→oo closed
  syllables only, medial deletion e/i-only + no CCC. 4=ش denied (Gulf, not
  Egyptian). Verdicts: data/stimulus/nilechat/rule_votes.tsv; open items
  (Q_TO_2 disagreement 4/5, missing habits) in council_pass1_open_items.md.
- Pass 2 (3 lenses × 1,899 variants): 107 flagged invalid (κ=0.29) — real
  collisions like 3alam→alam (علم→ألم). Flagged, never deleted.
- Pass 3 (5 lenses × 418 traps): 314 confirmed at ≥4/5 (κ=0.875) with Arabic
  + gloss both sides. merge_council.py regenerates the TSVs deterministically.

**Experiments:**
- `nilechat_wordset/{build_wordset,build_variants,mine_traps,merge_council}.py`:
  stratified 500-word selection (quotas per phenomenon; top-500-by-frequency
  gives too few gem/diph words), 3 carrier sentences per word, licensed
  variants with attestation tiers, trap mining inverted to CONVERGED pairs.
  Committed stimuli: data/stimulus/nilechat/stimuli_v1.csv (2,817 rows).
- `run_level1_tokenizer.py` v2: two arms (human/nilechat) in every run,
  star edges, mirrors + pinned signatures, --offline, bootstrap CIs,
  aggregate_by_class finally reachable. mt5-base replaces aya-101 as the
  unigram control; sentencepiece+protobuf now declared.
- `run_level3_gloss.py` (new): sentence gloss probe, Arabic-script gate +
  normalization, same-rendering ceiling, 30-word capability floor,
  prompt-sha'd cache, nested hash-stratified pilot. level2 kept frozen; its
  --limit slicing fixed to be order-independent.
- `src/visualization/`: theme/loaders/series stdlib-only (enforced by test),
  9 figures, light+dark, None never plotted as 0, small-n fading. matplotlib
  3.11.1 added to requirements-eval.txt.

**Results (pilot, 150 items × 4 models; experiments/d5_cohere_track/results/):**
- L1: separability AUC 0.56–0.63 on BOTH arms; byt5 frag 0.21 vs ~0.67
  production; DIGIT_SAFE fragments worst (pre-registered prediction held).
- L3: qwen most robust (agreement .51) AND most trap-fooled (.14); aya the
  reverse (.075/.26); llama FAILS the Arabic capability floor (.23) — its
  numbers are incapacity, not orthography. Ceilings ≈1.0.
- Joint (pre-registered): Spearman ρ=0.20 — fragmentation does not explain
  comprehension failure; reported as the near-null it is, no further slicing.
- Results moved to experiments/d5_cohere_track/results/ (git mv); top-level
  results/ keeps D1's showcase.

**Tests: 68 → 221**, all stdlib, no venv needed.

**Open:** native-speaker review of council verdicts/weights/traps; full-500
scale-up on qwen+gemma; --prompt-lang arz pilot arm; Q_TO_2/QAF class.

---

## 2026-08-22 — Terminology: "gloss" → "meaning" across the whole D5 track

`gloss` is linguistically correct (a word-level meaning annotation) but forced
every reader of a metrics module to hold a linguistics term. Worse, it named
two different things: the model's answer in the Level 3 probe, and plain
English/Arabic translation columns in the stimulus data. Replaced throughout,
with the two senses now visibly separated.

**Naming decided:**
- Sense A — the model's answer → **meaning**. Matches the prompt text, which
  already asks "what does the word ... mean?".
- Sense B — data columns → `meaning_en` (English), `meaning_ar_accept`
  (accepted Arabic answers), `meaning_en_word` / `meaning_en_trap`.
- The probe itself is now "the in-context word-meaning probe", not
  "sentence gloss" — the old name wrongly suggested glossing the sentence.

**Renamed files:**
- `src/franco_eval/gloss_metrics.py` → `meaning_metrics.py`
- `tests/test_gloss_metrics.py` → `test_meaning_metrics.py`
- `experiments/d5_cohere_track/e1_variant_probe/run_level3_gloss.py`
  → `run_level3_meaning.py`
- `results/level3_gloss.{json,md}` → `level3_meaning.{json,md}`
- `data/generated/level3_gloss_cache.jsonl` → `level3_meaning_cache.jsonl`

**Renamed identifiers:** `ItemGlosses`→`ItemMeanings`, `modal_gloss`→
`modal_meaning`, `modal_gloss_share`→`modal_meaning_share`, `gloss_accuracy`→
`meaning_accuracy`, plus docstrings, figure labels and README prose.

**Renamed data columns:** `pairs.csv` (`gloss_en`), `probe_hand.json`
(`gloss_en`, `gloss_ar_accept`), `trap_votes.tsv` (`gloss_word`,`gloss_trap`).

**Deliberately NOT renamed:**
- `gl_w` / `gl_t` in `merge_council.py` — keys of council judge JSON files that
  no longer exist on disk; renaming would silently break re-reading archived
  judge output. They are an input contract, not our vocabulary.
- `glossary` in `docs/BACKGROUND.md` §11 and in the NileChat corpus wordlist —
  a different word. (A `Gloss`→`Meaning` rule did briefly corrupt the
  BACKGROUND heading to "Meaningary"; caught and reverted.)
- `changes.md` history above, which records the old names as they were.

**Verification:** Level 3 regenerated fully from the prompt-sha cache
(3,800/3,800 calls cached, zero model calls — the rename never touched prompt
text). Regenerated JSON is byte-identical to the pre-rename file modulo the two
renamed metric keys; figures and `manifest.json` rebuilt. 221/221 tests pass.

**Noted, not acted on:** re-running `build_pairs.py` now produces a
*substantively* different `pairs.csv` than the committed one (new DIPHTHONG
rows; `2ol/oul` reclassified), because the working tree's `taxonomy.py` /
`phenomena.py` changes postdate that commit. That drift is unrelated to this
rename, so `pairs.csv` was restored and only its header column renamed. It
needs a deliberate regeneration commit of its own.

---

## 2026-08-22 (addendum) — project-wide gloss→meaning rename; filler-stripper fix

- User renamed "gloss" to "meaning" project-wide (meaning_metrics.py,
  run_level3_meaning.py, level3_meaning.{json,md}, cache file, tests).
  Verified coherent: loader paths updated, cache not orphaned (its key hashes
  prompt text, not names), 221 tests green.
- Bug found while explaining non-answers: `first_arabic_word` stripped filler
  openers even when the filler WAS the whole answer, so a model answering
  exactly "هو" (legitimate for a word like howa) scored as a non-answer. Fixed:
  fillers strip only when tokens remain. New test. Re-scored from the hot
  cache: aya non_answer 0.010→0.002, qwen →0.0, no ranking changed. Figures
  and manifest refreshed. Tests 221→222.

---

## 2026-08-23 — training-data pipelines: Dataset A (respell) built+run, Dataset B (FusioN) built+smoked

**Why.** Next phase after the benchmark: manufacture fine-tuning data that
teaches the spelling system in sentence context. Design principle: prose is
generated, orthography is ruled — no model ever authors a spelling.

**Dataset A (no model in generation):**
- `src/franco_eval/respell.py` — multi-span respeller (right-to-left span
  application; seeded sampling; blocked-word firewall).
- `training_data/collect_training_sentences.py` — 3,000 corpus sentences with
  >=2 respellable pool words; hash-strided across docs (1,051 distinct, <=3
  per doc), <=8 sentences/word; benchmark words + carrier sentences firewalled.
- `training_data/build_dataset_a.py` — generate (11,925 fat rows, seed 13,
  sentence-level 90/10 split seed 42 per the pre-restructure trainer
  contract), --prep-council (30 chunks), --emit (train/val JSONL as text or
  user→assistant pairs + stats + tracked rejected_for_review.csv with a
  reviewer override column).
- Verification council (150 Opus x-high calls, 5 lenses × 30 chunks) launched;
  merge via new `merge_council.py --training` (>=4/5 to accept).

**Dataset B (FusioN, arXiv:2510.00931):**
- `src/franco_eval/fusion.py` — anchors never model-authored; gates: anchor
  byte-identity + exactly-once + non-initial, ASCII-only (catches a LIVE
  French-drift case), English function-word drift, prompt-scaffold echo;
  difflib contribution audit per example.
- `training_data/build_dataset_b.py` — teachers τ=0.3 → fusor → gates →
  Dataset-A sentence schema; fusor is config (3-arm pilot pending: cloud
  amendment + model pulls). Smoke found and fixed two real leak channels:
  instruction-echo into data, and teachers copying the prompt's style example
  (examples removed as ungateable). All-Arabizi instructions failed live
  (teachers can't parse them); switched to English instructions + hard output
  constraints, same trade as the Level-3 probe.

Tests 222 → 260. CLAUDE.md map/commands updated. Fine-tuning itself remains
out of scope (hard rule 4) until the user amends it.

---

## 2026-08-24 (addendum) — Dataset A council completed at user's stop-at-100; calibration finding

- Council survived two session-limit deaths via recover-and-resume (judge
  files excavated from workflow transcripts once, from /tmp thereafter).
  User switched bulk judging to Sonnet mid-run and capped at 100 judge files:
  final panel = opus x-high chunks 0-13, sonnet-high chunks 14-17 (c14 mixed),
  chunk 18 partial, 19-29 unjudged. Mixed panel + per-cohort reject rates
  recorded in council_summary.json. kappa=0.39 overall.
- **Calibration finding: opus rejected 17.6% of its rows, sonnet 2.5%.** On
  the identical known-collision class (short-word 3-deletion, n3m->nam
  نعم->نام), opus rejected 50%, sonnet 12% — sonnet accepted e.g. 3amr->amr
  (عمر->أمر) and t3rf->tarf (تعرف->طرف). Judge model choice is itself a
  measured quantity now; bulk-vs-critical model policy saved to memory.
- Mechanism added: build_dataset_a.py --force-review FILE pulls listed
  rendering_ids from the kept set into the review CSV; 43 sonnet-accepted
  suspects (data/training/a_v1/calibration_suspects.txt) routed there.
- emit() gained an "unjudged" category: rows without a full panel are
  excluded from train/val AND from the review CSV (top-up eligible), never
  conflated with genuinely disputed rows.
- **Emitted data/training/a_v1/: 7,587 kept (6,840 train / 747 val), 814 in
  rejected_for_review.csv (reviewer column empty, awaiting the user's manual
  pass), 3,524 unjudged.**

---

## 2026-08-24 (addendum 2) — elongation policy auto-review

User policy: 3+-vowel elongation (okayyy-style emphasis) is legitimate
sentence-FINALLY, invalid mid-sentence/initially. Applied mechanically to both
sides of the council verdicts: 22 word-final elongation rejects rescued into
train; 242 mid-sentence elongation rejects confirmed; 176 kept rows violating
the rule pulled via force_review.txt (the lenient sonnet cohort had accepted
them). load_overrides now accepts annotated verdicts ("ok (auto: ...)").
Final a_v1: 7,438 kept (6,699/739), 963 in review CSV of which 545 await the
user's manual pass, 3,524 unjudged.

---

## 2026-08-24 (addendum 3) — review complete: Dataset A v1 final

User adjudicated all 963 review rows via pattern rulings: ع-deletion class
rejected (215+38 calibration suspects), vowel-length class rejected
(collisions + fake ee-/ii- article onsets), medial-vowel-drop class rejected,
judge-named collisions in small clusters rejected by shape, 22 joint-pass
leftovers rejected (English-token respells, fabricated th digraphs, mixed
elongations, residual collisions). 22 word-final elongations rescued earlier
stand. Final v1: 7,438 kept (6,699 train / 739 val), 963 rejected (all
annotated), 3,524 unjudged (top-up eligible).

Two rule gaps discovered by the review, filed for taxonomy v2:
1. licensing 7_TO_H needs `t` in its lookbehind (t+h fabricates a th digraph:
   mot7araka→motharaka) — taxonomy's reverse rule already guards t.
2. the respeller pool needs an English-token blocklist (offline, mile, doi,
   biology passed the gates and got Franco-respelled).
Plus the standing candidate: length/position guard on 3→a (ع-deletion).

---

## 2026-08-24 (addendum 4) — near-duplicate audit; SHORTEN_ATTESTED skipped

- Measured the user's overfitting worry: 22% of kept permutations differ from
  base by <=1 word; 18 exact duplicates (empty-substitution permutations)
  found and purged via a new emit guard. Final v1: 7,420 kept (6,682/738).
- Finding recorded in the continuation plan: carrier repetition (not
  rendering similarity) drives memorization; recommended v2 shape is
  inverted allocation (~6k sentences x 2 diversity-selected renderings).
- SHORTEN_ATTESTED (attestation+context-licensed vowel shortening) derived
  and evidence-tested (3-layer stack, zero trap false-accepts in sample) but
  SKIPPED at the user's call — value/complexity; evidence preserved in
  docs/superpowers/specs/2026-08-24-dataset-merge-fusion-plan.md history.
- Level-3 split by licence tier: models flip 42-67% even on corpus-attested
  variants — distributional exposure has not taught them the equivalences,
  so training pairs are not redundant where cosine is high.

---

## 2026-08-24 (addendum 5) — first native-speaker ruling: u never adjacent to w

User (native speaker) ruled: howa must never become huwa. Corpus confirms at
scale: howa 628,539 vs huwa 125; guwa unattested; ow:uw = 123:1 corpus-wide.
- licensing O_TO_U gains a w-adjacency guard: (?<![aeiouw])o(?![aeiouw]).
  U_TO_O left unguarded (moves toward the attested form). The e/i-by-y
  analogue measured at only 22:1 (iy is a minority real habit) — no symmetric
  guard without a further ruling; noted in the op's docstring.
- 49 affected kept rows purged to review CSV with the ruling as reason.
  Final v1: 7,371 kept (train+val), 1,012 adjudicated rejects, tests 260→262.
- This is the FIRST genuine native-speaker sign-off in the taxonomy's history
  (everything else is council/author judgement pending review).

---

## 2026-08-24 (addendum 6) — rendering diversity shipped; Dataset B 3-arm pilot run (2/3 arms)

**Part 1 — Dataset A rendering diversity (section 1 of the continuation plan),
built and applied to production data:**
- New `src/franco_eval/diversity.py`: `pairwise_distance()` (word-level
  choices-dict comparison) + `select_diverse()` (greedy farthest-point/maximin
  selection over one or more anchors, with a hard near-copy exclusion —
  candidates within `min_distance_from_base` words of every anchor are never
  selected, even to fill a short quota). 8 unit tests, all passing.
- Wired into `build_dataset_a.py`'s `generate()`, which now also PRESERVES
  every already-council-voted permutation row byte-identical (rendering_id is
  positional, not content-hashed, despite the plan doc's assumption otherwise
  — overwriting it under a new sampler would have silently orphaned real
  votes). Only unjudged rows get replaced by the new diverse sampler.
- Regenerated the real fat file: 11,916 rows (was 11,925). Verified all 5,700
  previously-voted rendering_ids unchanged byte-for-byte; the 3,233 freshly
  generated rows have a 0.0% near-copy rate (vs 19.1% in the untouched
  old-sampler rows) — the near-duplicate problem from addendum 4 is now
  structurally impossible for new generations, not just measured-and-tolerated.
- Fixed an inaccurate paper-attribution comment in `fusion.py` while
  rechecking the FusioN paper (arXiv:2510.00931) methodology: the general
  language-drift finding (98.9% correctness, English-only fusion prompt) is
  real; "Arabic was weakest of eleven languages" was not supported by the
  paper's own per-language table (Arabic +1.2 win-rate; Japanese was the
  smallest gain, +0.2) and has been corrected.

**Part 2 — Dataset B FusioN 3-arm pilot (section 2), 20-anchor scale:**
- CLAUDE.md's no-network rule amended (third amendment): `glm-5.2:cloud`
  network access permitted, scoped to `build_dataset_b.py` only.
- `falcon-h1-arabic-34b` and `jais-2-8b` from the original spec were never
  real Ollama tags. Substituted `jwnder/jais-adaptive:7b` (teacher) and
  `command-r:35b` (fusor, after Hugging Face's CDN proved unreachable at any
  useful speed on this network for two different GGUF mirrors).
- Found and fixed a real bug: `build_dataset_b.py` read the wrong TSV
  columns for anchor glosses (`gloss_word`/`gloss_trap` vs the actual
  `meaning_en_word`/`meaning_en_trap`) — every anchor's gloss was silently
  empty. Also added `num_ctx` to `model_clients.GenConfig` (missing
  entirely): `command-r:35b`'s Ollama package defaults to a 131072-token
  context window, and allocating that KV cache for an 80-token completion
  pushed every fusor call past the 240s timeout. Pinned to 4096.
- Ran arms 1 (`qwen3.8:27b-mlx`) and 2 (`command-r:35b`) against the same 20
  confirmed-trap anchors and 5 teachers (`aya-expanse:8b`, `gemma4:e2b`,
  `llama3.1:8b`, `jwnder/jais-adaptive:7b`, `nemotron-3-nano:30b`, N=5 τ=0.3
  — matching the paper's own synthetic-data-generation setting). New
  `prep_council_fusion.py` pools + blind-relabels gate-passed output for a
  5-lens naturalness council (fluency/anchor-fidelity/degeneracy/coherence/
  register), reusing Dataset A's ≥4/5-agreement bar.
- Result: `command-r:35b` (35B) beat `qwen3.8:27b-mlx` (27B) on naturalness
  despite passing gates less often — 45% vs 13% of gate-passed sentences,
  25% vs 10% end-to-end yield. `qwen3.8:27b-mlx` repeatedly produced
  repetition-loop degeneracy that passed every mechanical gate but failed
  every naturalness lens. Full results:
  `experiments/d5_cohere_track/results/fusion_pilot.{json,md}` +
  `fusion_pilot_council_detail.json`.
- Arm 3 (`glm-5.2:cloud`) did NOT run: needs an Ollama subscription upgrade
  (HTTP 403). The `.env` Z.AI fallback key the user pre-authorized is valid
  (authenticates) but the account has no funded balance. Both blockers need
  a billing decision only the user can make.

---

## 2026-08-24 (addendum 7) — arm 3 unblocked: dramatic 3-arm result

User pushed back on the "insufficient balance" diagnosis (the Z.AI GLM
Coding Plan should work fine) — right call to debug further rather than
accept it:
- The generic endpoint (`api.z.ai/api/paas/v4`) really was checking a
  different balance than the coding plan funds. The coding plan's quota
  lives behind a SEPARATE dedicated endpoint (`api.z.ai/api/coding/paas/v4`)
  not mentioned on Z.AI's own quick-start page — only surfaced by reading
  `docs.z.ai/devpack/quick-start`. New `src/franco_eval/zai_client.py` is
  the one place in the repo that reads `.env`; wired into `build_dataset_b.py`
  as `--fusor "zai:<model>"`.
- Second bug: `glm-5.3`'s reasoning is mandatory and cannot be disabled —
  a bounded token budget (80, even 600 in isolated testing) was consumed
  entirely by the reasoning trace, leaving an empty `content` field
  (`finish_reason: "length"`). Fixed with `reasoning_effort: "low"` (Z.AI's
  documented way to shrink, not eliminate, the trace) plus a padded budget.
- Re-ran arm 3 (`zai:glm-5.3`) on the same 20 anchors/5 teachers and redid
  the blind council (41 candidates across all 3 arms this time). Result is
  far more decisive than the 2-arm version: naturalness tracks fusor scale
  monotonically -- `zai:glm-5.3` 93% (14/15) vs `command-r:35b` 36% (4/11)
  vs `qwen3.8:27b-mlx` 13% (2/15); end-to-end yield 70% vs 20% vs 10%.
  CLAUDE.md's Environment notes amendment updated to document the working
  Z.AI path. Full results superseded in
  `experiments/d5_cohere_track/results/fusion_pilot.{json,md}` +
  `fusion_pilot_council_detail.json`.
- Also flagged: an early debugging command (`python3 -c "...repr(content)"`
  meant to check only the `.env` key's shape) accidentally printed the full
  key value into the visible session output. User was told immediately;
  worth a rotation if that's a concern.

---

## 2026-08-24 (addendum 8) — second pass: scenario-grounded prompts, arena benchmark, XCOMET

Reviewing addendum 7's actual sentences found a real weakness: bare-anchor
prompts gave models nothing to write about, so every anchor converged on
the same few sentence shapes (reported-speech openers, "I want to X"). Full
write-up in `experiments/d5_cohere_track/results/fusion_pilot.md`'s
"Second pass" section; summary here.

- **Prompt fix**: one scenario (of a new 16-entry `SCENARIOS` tuple in
  `fusion.py`) drawn per anchor, given to every teacher AND the fusor for
  that anchor (never per-teacher — that would leave nothing comparable to
  fuse). `fusion_prompt()` also adopts the FusioN paper's own
  compare-strengths-and-weaknesses-then-fuse structure (Table 6, checked
  directly against the paper, not assumed).
- **Result**: both local arms roughly doubled their naturalness rate
  (`command-r:35b` 36%→67%, `qwen3.8:27b-mlx` 13%→27%); the already-strong
  `zai:glm-5.3` held flat within noise (93%→89%). Same blind 5-lens
  council, same ≥4/5 bar as addendum 7 — a real before/after, not a
  differently-measured number.
- **New arena benchmark** (`run_arena_judge.py`, no such pairwise-judge
  code existed anywhere in the repo before this): FusioN's fused output vs
  a Best-of-N baseline (first raw teacher draft passing the same gates, no
  extra model call). Win rate scales with fusor capability exactly like
  naturalness did — `zai:glm-5.3` 88%, `command-r:35b` 64%, `qwen3.8:27b-mlx`
  53% — a second, independent measurement confirming the same story.
- **DeepSeek external reference**: new `src/franco_eval/deepseek_client.py`
  (CLAUDE.md fourth amendment; user added a `DEEPSEEK=` key to `.env`
  themselves, with an explicit go-ahead to wire it up). Fixed the same
  mandatory-reasoning issue `zai_client.py` already hit. Once fixed, found
  a genuine blocker, not a bug: DeepSeek answers in full Arabic script
  despite an explicit Latin-only rule almost every time (0/45 usable) —
  echoes the paper's own language-leak finding in a plain single-shot
  context. Deferred rather than forced.
- **XCOMET-XL attempted and obtained**, through three real blockers in a
  row: a hard `huggingface-hub` version conflict with the main `.venv`
  (needs its own `.venv-comet`, Python 3.12 — comet's pinned
  `torchmetrics==0.10.3` imports private `functools` internals removed in
  3.14), a gated HF repo (user granted access on their existing HF login),
  and a real bug in comet's own code (unconditionally sets
  `multiprocessing_context="fork"` whenever MPS is available, crashing
  `num_workers=0` on any Apple Silicon Mac). Scored, but the result
  **disagrees** with both other measurements and looks like noise from a
  stretched application (a translation metric applied to non-translation
  monolingual generation) rather than a genuine third opinion — reported
  honestly rather than cherry-picked.
- **Two more real bugs found and fixed**: `run_arena_judge.py` originally
  clobbered the previous arm's output on each `--arm` invocation (fixed to
  pool multiple arms in one call); `deepseek_reference()`'s caching forced
  empty completions to cache as permanent failures, blocking retries
  (removed — `JsonlCache`'s existing skip-caching-failures behavior was
  correct all along).
- **Operational lesson**: running Ollama generation concurrently with the
  11GB XCOMET-XL PyTorch checkpoint exhausted this machine's 32GB RAM into
  swap and silently `SIGKILL`'d both processes with no log output at all.
  These now run strictly sequentially, never concurrently.
- New files: `src/franco_eval/deepseek_client.py`, `xcomet_metric.py`,
  `experiments/d5_cohere_track/training_data/run_arena_judge.py`,
  `requirements-comet.txt`, `.venv-comet/` (gitignored).

---

## 2026-08-25 (addendum 9) — correcting addendum 8's DeepSeek finding

Addendum 8 reported "DeepSeek can't reliably write Franco" (0/45 usable
references, script drift) as a genuine finding. **It was wrong**, caught
by the user pointing out DeepSeek writes Franco easily in a normal chat —
worth a live re-test instead of trusting the earlier conclusion.

- Root cause: `deepseek_client.py` used `reasoning_effort="low"`
  (mirroring the fix `zai_client.py` already needed). Under this
  pipeline's multi-constraint prompt (exact anchor word, exact spelling,
  exact position, ASCII-only, word-count range), low reasoning effort made
  the model rush and drop the Latin-script rule specifically — not a
  capability limit. Confirmed by testing `reasoning_effort="high"` on the
  same failing anchor plus 2 more: 3/3 correct. Fixed: `reasoning_effort`
  now `"high"`, token budget padded further (`num_predict*10`, min 800) to
  cover the extra reasoning tokens that costs.
- Re-ran the DeepSeek arena comparison for real: 27/45 usable references
  (up from 0/45), and 42 vs-BoN + 27 vs-DeepSeek comparisons re-judged
  fresh in one pass (the regenerated chunk files reshuffled candidate
  positions, so old verdicts couldn't be safely reused against the new
  private key).
- The corrected result is far more informative than the wrong one would
  have been: FusioN only beats DeepSeek's single-shot attempt when the
  fusor itself is frontier-tier — `zai:glm-5.3` comes out roughly even
  (54% win rate + ties) while both local fusors lose almost every time
  (`command-r:35b` 12%, `qwen3.8:27b-mlx` 0%). A real refinement of the
  paper's claim: fusion beats selection reliably regardless of fusor
  scale, but does not automatically beat asking a strong model directly
  unless the fusor is itself strong.
- `experiments/d5_cohere_track/results/fusion_pilot.{md,json}` and
  `fusion_pilot_v2_detail.json` all updated to the corrected numbers.

---

## 2026-08-25 (addendum 10) — redo the external reference with DeepSeek V4 Pro

User asked to redo the external-frontier-reference arena leg using
`deepseek-v4-pro` (GA 2026-08-13, 1.6T total/49B active — the larger
reasoning-tier sibling of `deepseek-v4-flash`'s 284B/13B active) in place
of `flash`, to see whether a stronger, pricier reference changes addendum
9's finding. Confirmed the model identifier via a live doc lookup rather
than guessing (same discipline as the earlier `falcon-h1-arabic-34b`
placeholder-name mistake) — `deepseek-v4-pro` and `deepseek-v4-flash` are
both real, current API model strings.

- **Bug caught before it could bite**: `run_arena_judge.py`'s DeepSeek
  reference `JsonlCache` was keyed on `(anchor, scenario)` only — no
  `model` field, despite the class's own docstring saying the cache is
  meant to key on `(model, rendering, prompt)`. Switching
  `--deepseek-model` from `flash` to `pro` without this fix would have
  silently replayed `flash`'s cached answers under the `pro` label. Fixed
  by adding `model` to the key; cleared the old 16-row flash-only cache
  (disposable — already fully reflected in the committed results).
- **New bug found, same failure shape as addendum 9's, different root
  cause**: the first real `deepseek-v4-pro` run got only 5/45 usable
  references. Live isolated test showed `finish_reason: "length"`, empty
  `content`, `reasoning_tokens: 800` (exactly the cap) — this is *not* a
  repeat of the `reasoning_effort` bug (effort was already `"high"`).
  `deepseek-v4-pro` is simply a bigger model that reasons longer in
  absolute tokens on the identical one-sentence task (1,100–2,750
  reasoning tokens observed live across 3 test anchors). The 800-token
  budget sized for `flash` left `pro` unable to ever finish. Raised
  `max_tokens` to a flat 4,500 (confirmed live, 3/3 clean); costs nothing
  extra for `flash` calls since DeepSeek bills tokens actually used, not
  the cap. Re-ran: 40/45 usable (better than `flash`'s 27/45).
- **Result is sharper, not just different**: against `deepseek-v4-flash`,
  the frontier fusor `zai:glm-5.3` came out roughly even (54% + ties).
  Against the stronger `deepseek-v4-pro`, it loses almost every time (6%,
  1/16) — as do both local fusors (`command-r:35b` 0%, `qwen3.8:27b-mlx`
  0%). Refines rather than contradicts addendum 9's finding: fusion's
  ability to match an external model directly depends on the fusor being
  *at least as strong as whatever it's being compared against*, not just
  "frontier-tier" in isolation. Fusion's dominance over Best-of-N is
  untouched by this (and looks even more decisive under the same
  re-judging pass: `zai:glm-5.3` 94%→100%, `command-r:35b` 64%→91% — noted
  as ordinary judge-to-judge variance on an already-decisive comparison,
  not a new finding, since the underlying sentence pairs didn't change).
- `experiments/d5_cohere_track/results/fusion_pilot.{md,json}` and
  `fusion_pilot_v2_detail.json` updated with the `deepseek-v4-pro` numbers
  alongside (not replacing) `flash`'s — this is a new comparison, not a
  correction. `CLAUDE.md`'s fourth amendment updated (was still describing
  the pre-addendum-9 `reasoning_effort="low"` state).

**Follow-up same session — is this a fusion weakness or a teacher-pool
ceiling?** User asked directly: is fusion losing to DeepSeek because
fusion itself is weak, or because the 5-teacher pool (8B–30B open models)
is simply outclassed by DeepSeek regardless of fusion? Tested with a new
`probe_teacher_ceiling.py`, comparing `bon_pick` (the single best raw
teacher draft, no fusor at all) against the same cached `deepseek-v4-pro`
references — **zero new model calls**, both sides already existed from
prior runs. Result: **0/41 (0%)** — every single raw teacher draft, across
all three arms, lost to `deepseek-v4-pro`. Confirms the hypothesis
decisively: the teacher pool is the bottleneck, not fusion. Fusion adds
exactly nothing on top of that 0% floor for the two weaker fusors (still
0/12 each), but the frontier fusor (`zai:glm-5.3`) lifts it to 1/16 — the
only arm where fusion beats DeepSeek-Pro at all when no raw draft ever
did, suggesting a frontier-scale fusor leans on its own generative
capability rather than purely recombining draft content. Added to
`fusion_pilot.{md,json}` and `fusion_pilot_v2_detail.json`.

---

## 2026-08-27 — code review of the D5 track, and one gate bug fixed

Full security + quality review of the uncommitted D5 Cohere track before
committing it (124 files, `407018f`). Security gate passed: `.env` untracked
and ignored, no hardcoded secrets in the staged diff, no `eval`/`exec`/
`shell=True`, no bare excepts, the 853MB `data/external/` correctly excluded.
Two files were kept out of the commit: `system_prompt.md` (a 180KB Claude
system-prompt dump in the repo root, zero project-term matches) and
`{filepath}.tmp` (litter from a broken formatter hook in the user's global
`~/.claude/settings.json`, which uses a `{filepath}` placeholder that Claude
Code never substitutes).

**Fixed — `fusion.py` rejected valid sentences on a string prefix.**
`check_sentence` asked `lowered.startswith(anchor.word)` to decide whether the
anchor was sentence-initial, but that matches a *prefix of the first token*,
not the first token. `3ayezha ... 3ayez ...` has the anchor at position 5 and
was still rejected as "sentence-initial". Egyptian clitics (-ha -ak -ni -lo)
make a stem a prefix of its own inflected forms constantly, so this discards
valid data as the anchor set grows; `mine_traps.py` already models exactly
this via its `CLITICS` constant, making `fusion.py` the lone outlier. Now
`words[0] == anchor.word`, comparing tokens. Confirmed by the user as a
correct reading of Egyptian morphology.

Measured before changing anything: the bug fired on **0 of 603** real audit
rows, and after the fix all 603 `gates_ok` values regenerate identically, so
no committed number moves and hard rule 2 holds. Regression test added
(`test_first_word_merely_prefixed_by_the_anchor_is_not_initial`), verified to
fail against the old line and pass against the new one. 272 tests pass.

**Reviewed and deliberately not changed (user's call, both measured latent):**

- `taxonomy.py:65` — `Rule(r"ee", "ei", 0.3, ...)` is the only one of the six
  diphthong rules missing the `(?![aeiou])` guard its siblings carry, so it
  can fold `tawzeea -> tawzeia`. **Native-speaker judgment: these read as odd
  but understandable, not implausible** — within tolerance for a 0.3-weight
  rule, so left as-is. Fires on 0 of 1899 committed variant rows (no base word
  in the set contains `ee`+vowel).
- `script_id.py:12` — `[؀-ۿ]` is the whole Arabic *block*, of which only 150
  of 256 codepoints are letters; Arabic-Indic digits and punctuation score
  `arabic_ratio == 1.0`, contradicting the docstring and disagreeing with
  `arabic_norm.py`'s letter-only `[ء-غف-يٱی]`. Out of scope for the current
  use case per the user (the probe never sees punctuation-only answers);
  measured 0 of 8011 cached model replies affected.

**Open MEDIUM findings, not yet actioned:** `write_manifest.py:39` `git()`
discards the return code, so a failed git call writes a blank `git_commit`
into the reproducibility manifest; `zai_client`/`deepseek_client` retry
permanent failures (`FileNotFoundError` is an `OSError`, `HTTPError` is a
`URLError`) and their `-> str` `chat()` can return `None` when the API sends
`content: null`, unwrapped at `run_arena_judge.py:70`; `model_clients.py:88`
`JsonlCache.__init__` dies on a truncated final line; `diversity.py:52`
`min()` over empty `anchors` raises a bare `ValueError`; `context.py:160`
buckets any non-`SAME` label as `DIFFERENT`; 21 functions exceed the 50-line
standard (worst: `build_dataset_a.generate()` at 138).

---

## 2026-08-27 — teacher-pool redesign: closing the scale gap the paper never had

User asked for a direct performance comparison against the FusioN paper's
own model choices, to check whether our pilot's "fusion loses to
DeepSeek" finding was fair. Verified live (not from memory) the paper's
synthetic-data-generation teacher pool (Gemma3-27B, Kimi-K2-Instruct
1T/32B active, Qwen3-235B-A22B 235B/22B active, DeepSeek-V3 671B/37B
active, Command-A 111B) against ours (aya-expanse:8b, gemma4:e2b,
llama3.1:8b, jais-adaptive:7b, nemotron-3-nano:30b, capped at 30B). The
paper's teacher pool already contained frontier-scale members roughly
comparable to their external reference (Gemini); ours is ~50x smaller
(by total params) than `deepseek-v4-pro`. The paper never actually tested
the regime we were testing — small teachers vs. a frontier reference —
so our near-zero win rate wasn't a fusion failure, it was a confound.

**New teacher pool** (6 teachers, up from 5 -- deviates from N=5 parity
with the paper, at the user's explicit direction, to genuinely test
whether a stronger pool changes the outcome): keep `llama3.1:8b` and
`gemma4:e2b`; drop `jwnder/jais-adaptive:7b` and `aya-expanse:8b`; swap
`nemotron-3-nano:30b` (local) for `nvidia/nemotron-3.5-lightning:free`
(OpenRouter, nominally free, 30B/3B active MoE -- see reliability note
below); add `zai:glm-5.3-flash` (320B/18B active) and
`deepseek:deepseek-v4-flash` (both already had clients, no new
credential); add `openrouter:minimax/minimax-m3:free` as the sixth.

**New external reference**: `glm-4.6` (357B/32B active) alongside the
existing `deepseek-v4-pro`, at the user's request to run both rather than
replace -- `deepseek-v4-pro` is the frontier-tier comparison, `glm-4.6` a
genuinely weaker one, giving two points instead of one near-zero data point.

**New infrastructure:**
- `src/franco_eval/openrouter_client.py` (new) -- one credential
  (`.env`'s `OPENROUTER=`, CLAUDE.md fifth amendment) via
  `openrouter.ai/api/v1/chat/completions` (OpenAI-compatible). Model
  strings are OpenRouter's own catalog names passed through verbatim.
- `build_dataset_b.py`: re-added the `"deepseek:"` teacher/fusor dispatch
  prefix (built once already this session, reverted when an earlier,
  narrower "add DeepSeek as a 6th teacher" idea was called off, now
  genuinely needed) and added a matching `"openrouter:"` prefix, same
  convention as the existing `"zai:"` one.
- `run_arena_judge.py`: generalized from a single hardcoded
  `--deepseek-model` to a repeatable `--reference "label=prefix:model"`
  argument, producing one `"vs_<label>"` comparison leg per reference per
  anchor per arm. The reference cache (renamed `reference_cache.jsonl`
  from `deepseek_ref_cache.jsonl`) is keyed on the full `prefix:model`
  spec, not just the bare model name, so different reference types or
  DeepSeek tiers can never collide -- same discipline as the earlier
  missing-`model`-key-field fix, applied proactively this time rather
  than caught after the fact.
- **Bug found and fixed wiring up `glm-4.6`**: same failure shape as the
  `glm-5.3`/`deepseek-v4-pro` reasoning-budget bugs, but a different
  cause -- live diagnosis showed `glm-4.6`'s actual reasoning need was
  tiny (~35 tokens for a test prompt, `finish_reason: "stop"` at
  `max_tokens=1500`), but the existing 300-token floor caused the model
  to dump its *entire* budget into reasoning and return empty content,
  not because it needed hundreds of reasoning tokens but because too
  tight a cap triggers degenerate behavior. Raised `zai_client.py`'s
  floor to 2500 (shared with `glm-5.3`, costs nothing extra for calls
  that don't need it).

**OpenRouter key hit its own spending limit** (`"Key limit exceeded
(total limit)"`, HTTP 403, on both `z-ai/glm-5.3-flash` and
`qwen/qwen3-235b-a22b` -- a cap on the key itself, needs the user to
raise it or fund the account; not a code issue). Worked around rather
than blocked on:
- `glm-5.3-flash` turned out reachable through the existing `zai:`
  client, untested until tried live (the Z.AI docs page didn't list it,
  the coding-plan endpoint accepts it anyway) -- now dispatched as
  `zai:glm-5.3-flash`, no OpenRouter credential needed at all.
- `qwen/qwen3-235b-a22b` (paid, blocked by the same limit) replaced with
  `openrouter:minimax/minimax-m3:free` -- genuinely free, a different
  family from everything else in the pool (avoids doubling up on GLM,
  which `glm-5.3-flash` already covers), confirmed working live.

**`nvidia/nemotron-3.5-lightning:free` reliability finding**: three
different failure modes across three attempts on OpenRouter's free-tier
routing for this model -- HTTP 404 ("Provider returned error", empty
body), a request that hung past a 90s timeout, and HTTP 200 with a
blank/non-JSON body. Corroborated by external reports of the same
failure shapes on other Nvidia-routed OpenRouter free models (a GitHub
issue describing the gateway stripping the `nvidia/` prefix before
forwarding to Nvidia's own API causing 404s; another user's report of a
different free Nemotron variant timing out with a 504). This looks like
a genuine instability in OpenRouter's free Nvidia routing, not a bug in
our client -- flagged, not yet swapped, since the user asked only about
the other two models this round.

**Explicitly deferred, at the user's direction**: the earlier "run this
overnight at 628 anchors" plan was called off mid-attempt -- a real
double-backgrounding bug (own `&` plus the tool's own backgrounding)
caused a false "completed" report after doing nothing, and even after
fixing the launch, the corrected job produced no output for several
minutes with no clear signal of being alive (it turned out to genuinely
still be running -- likely Ollama cold-loading multiple multi-GB models
sequentially, not a bug, but unconfirmed). User's call: no more large
unattended runs until there's a real measured per-anchor throughput
number, and now that the pipeline adds up to 3 external API calls per
anchor (OpenRouter, DeepSeek, Z.AI) on top of Ollama, that number needs
re-measuring, not reused from the old 5-teacher-all-local design.

272 tests pass throughout. Nothing committed this session.

---

## 2026-08-31 — "experiment b2": quick pass with a stronger teacher pool

User wanted this wrapped up fast rather than continuing to chase
OpenRouter's free-tier problems. Resolved the two blockers pragmatically:
`glm-5.3-flash` turned out reachable directly through the existing `zai:`
client (no OpenRouter needed at all); `qwen/qwen3-235b-a22b` (blocked by
the OpenRouter key's own spending limit) was replaced with
`openrouter:minimax/minimax-m3:free`, confirmed genuinely free and
working. Verified (via OpenRouter's own docs) that funding the account
would NOT have auto-fixed the key limit -- that's a separate per-key
setting -- so skipping it was the right call, not just the fast one.
`nvidia/nemotron-3.5-lightning:free` was retried twice more, a day apart,
producing the identical blank-body failure both times -- confirmed
persistent, not transient; left unresolved since the user redirected to
a simpler design before it needed fixing.

**Final design run** (user's explicit request, prioritizing speed): 5
teachers (`llama3.1:8b`, `gemma4:e2b`, `nemotron-3-nano:30b` local;
`deepseek:deepseek-v4-flash`, `zai:glm-5.3-flash` API), same 3 existing
fusor arms, single external reference `deepseek-v4-pro`, n=20, judged by
Sonnet. Results and interpretation are in
`results/fusion_pilot.md`'s "Experiment b2" section and the new
`results/fusion_pilot_overview.md` (a short, non-technical summary the
user asked for, meant to compress into an email).

**Three real bugs hit and fixed during this run, none affecting final
numbers:**
1. Ollama's local server had stopped running entirely (`Connection
   refused` on every local fusor call) -- restarted it, unrelated to any
   code change.
2. **Silent scope bug, the important one**: tonight's fresh
   `--smoke 20` run for the `qwen3.8:27b-mlx` arm wrote to
   `b2_arm_qwen_audit.jsonl` -- the same filename used by the interrupted
   `--smoke 0` (all 628 anchors) overnight attempt from days earlier
   (addendum: "experiment b2" originally). Since `build_dataset_b.py`
   opens output files in append mode, tonight's 20 rows landed on top of
   505 leftover rows from that old, explicitly-called-off run, silently
   inflating the arena benchmark's scope from ~20 pairs to 318. Caught
   when the user asked "how much is left" and the reference-cache growth
   rate didn't match a 20-anchor job; fixed by moving the stale files
   aside (`data/generated/fusion_pilot/_stale_overnight_backup/`) and
   regenerating the arm cleanly (identical 17/20 kept, confirming
   determinism). Lesson: reusing an output filename across unrelated runs
   is unsafe with an append-mode writer; should have used a fresh name
   or truncated first.
3. `run_arena_judge.py` crashed the entire run (twice) on a single
   dropped connection mid-run (`socket.gaierror`, DNS resolution
   failure) -- traced to this machine being tethered to a phone hotspot
   rather than stable WiFi (`networksetup -getairportnetwork en0` showed
   no WiFi association; the DNS server was a hotspot gateway address).
   `deepseek_client.chat_with_retries()` already retries a single call 3
   times before giving up, but the *script* still let that final
   exception kill every anchor after it. Fixed: one reference's failure
   is now skipped and counted as gated-out, not fatal to the run.

272 tests pass. Nothing committed this session.

---

## 2026-09-23 → 09-25 — v2 fine-tuning dataset: whole-corpus respell + fusion, held-out split, pair-level policy

Goal (user): scale the data-generation setup over all of NileChat into a
dataset ready to fine-tune on, teaching two rules at once -- respellings of one
word mean the same (SAME) and look-alike trap words do not (DIFFERENT) -- and
measurable on words never trained on.

**Design decisions (user):** respell the whole corpus with no model, fuse a
bounded anchor slice (FusioN kept, `deepseek-v4-flash` added as a teacher);
fine-tune evaluated on a HELD-OUT word split of this repo's own benchmark;
row types normalize (mixed spelling -> base) + meaning (word in context ->
Arabic), emitted by separate functions so other tactics need no regeneration.

**Built (all tested, stdlib unittest; 272 -> 562 tests):**
- `src/franco_eval/word_split.py` + `split_words.py` -> tracked
  `data/stimulus/nilechat/word_split.tsv`: 500 benchmark words split by
  union-find components over trap pairs (a word and its trap never straddle),
  salted hash; human arm always held out. 225 train / 275 held-out words; 149
  confirmed trap pairs on the train side = 298 anchors.
- `collect_training_sentences.py`: `--split` firewall (held-out side read from
  the split, which is read from stimuli_v1.csv itself), `--row-group-order
  hashed`, unlimited mode. Default path regenerates a_v1's sentences
  BYTE-IDENTICAL (verified). Whole corpus: 158,452 sentences from 571,524 docs
  (the 8-sentences-per-word cap binds, not the corpus).
- `src/franco_eval/substitution_policy.py`: v2-only pair-level filter layered
  AFTER licensing.py (which also generates the benchmark stimuli, so it is not
  touched). Encodes the a_v1 review's rulings as code: known trap partners,
  council-named bad pairs, held-out stimulus forms, REVIEW_ENGLISH,
  mid-sentence elongation, 7->h after t, fake ee-/ii- onsets.
  **NATIVE RULING 2026-09-25:** ع-deletion (3_TO_A) allowed only word-
  initially (3amr -> amr fine, n3m -> nam reads as نام), medial vowel drop
  allowed; word-initial ع-deletion and a vowel dropped beside a vowel "work in
  context" but each distinct pair needs a judge (`needs_judge`).
  `audit_policy.py` vs the 5,400 judged a_v1 rows: residual error 6.1%
  (rules only, honest) vs 17.9% baseline.
- `build_dataset_v2.py` (master records; per-sentence RNG + hash split, so
  adding fused data never reshuffles NileChat rows), `src/franco_eval/
  emitters.py` + `emit_v2.py` (normalize/meaning rows; templates deliberately
  not the Level-3 probe's wording; meaning rows need a single-word Arabic --
  10 anchors carry two readings). Row files under data/generated/v2/final/
  (gitignored, too large); stats + sample tracked in data/training/v2/.
- Fusion run (`build_dataset_b.py --split --scenarios-per-anchor 4`): 1,192
  jobs in 4.6 h, 1,135 gate-passed, 0 model failures. Teacher-major prefetch
  (cold nemotron load 60 s vs 2.5 s warm) roughly halved the projected time.
  Fused sentences + audit TRACKED in data/training/v2/ (cloud output, not
  regenerable).
- 5-lens Opus council on the fused sentences (50 judges, 3.9M tokens):
  707/1,122 accepted at >=4/5, kappa 0.121; the fluency lens alone rejected
  343 otherwise-clean sentences (English calques) -- user ruled keep >=4/5.
  Verdicts in data/stimulus/fusion_v2_votes.tsv, per-judge files copied to
  data/stimulus/council_v2/fused/.
- Expansion setup (four parallel agents): anchor bank (`anchor_bank.py`,
  `data/stimulus/v2_anchor_bank.tsv`, 284 trap + 52,254 bare), scenario bank
  (`scenario_bank.py`, `data/stimulus/scenarios_v2.tsv`: ids 0-15 =
  fusion.SCENARIOS verbatim, 315 new pending the user's review in
  scenarios_v2_review.csv), row templates widened to 10/11, checkpointed
  runner (`plan_fusion_v2.py` frozen plans, batches, quota errors from Z.AI /
  DeepSeek / OpenRouter docs classified and never retried -> pause exit 75,
  `run_fusion_v2.sh` wait-and-resume loop), council tooling committed as the
  method of record (`training_data/council/{fused,pair}_council.js`,
  council_jobs.py, compare_councils.py, sample_audit.py).

**Bugs found and fixed:** (1) Dataset A's firewall missed the 418 probe TRAP
words (164 respelled in 762 a_v1 rows) -- v2 firewalls from stimuli_v1.csv
itself. (2) 14 anchor words sit in two trap pairs -> 13 duplicate fused
sentence ids (kept first; runner now refuses duplicates). (3) Throughput
report timed only audited jobs (0.0 s/job on a failing run). (4) BoN
attribution used shifted teacher indices after a teacher failure. (5) glm-5.3
wrote non-Egyptian digit letters (4/6/9: maba5las4, fe6or) into 9 base texts
-> new fusion gate; master build re-gates old fused text. (6) JsonlCache died
on a torn last line after a kill -- now skipped and isolated, file never
rewritten.

**Dataset v2 as emitted:** mixed_train 258,322 / mixed_val 28,698 rows
(normalize 285,451 incl. 918 fused; meaning 1,569). Held back: 18,057 rows
carrying 6,804 flagged pairs awaiting the pair council (115 Opus judges,
approved, canary running). Contaminates stimuli_v1 BY DESIGN on the train
side: evaluate only on the held-out side of word_split.tsv.

**Open:** scenario review (315); expansion pilot (~2,784 anchors, ~10 h);
Sonnet-vs-Opus calibration before any bulk Sonnet council; Z.AI coding-plan
usage policy (docs: "officially supported tools" only -- scripted volume
risks a fair-use freeze, code 1313); ~504 unjudged train-side trap candidates
(~280-360 would confirm) are the only route to more meaning rows.

Nothing committed.

---

## 2026-09-26 (addendum) — pair council: Opus hit the session limit; Sonnet/medium calibrated and found unusable

- The 115-judge Opus pair council finished 92/115 judges before the session
  limit (chunks c0-c17 complete). User ruling: finish with Sonnet on MEDIUM
  effort ("a verdict is a narrow right/wrong call"). Policy updated in
  `council/council_io.py` (POLICY_PANELS: sonnet/medium = bulk) and memory.
- Calibration before trusting it: 5 Sonnet/medium judges re-judged chunk c0
  (300 pairs Opus already judged). Result
  (`results/council_calibration_pairs_v2_c0_sonnet_medium_vs_opus.md`):
  Sonnet rejects 1.0% vs Opus 26.3%; it misses 96% of Opus's rejects; Fleiss'
  kappa across its lenses -0.006 (Opus 0.626) -- four of five lenses accept
  99-100%, only `skeptic` works. Accepted collisions include 2eamet -> 2amet
  (value-of -> she-stood-up), 2almania -> almana (Germany -> trust), 32ol ->
  a2ol (minds -> I say). Even a 5/5 bar still misses 60%. Sonnet/medium
  verdicts for c18-c22 are NOT merged pending the user's decision.
- New `council/assemble_panels.py`: one verdict dir from several panels, each
  owning whole chunks, with `panel_map.meta` (not .json -- merge_council reads
  every *.json as a judge file; that crash was hit and fixed).
- Merged the Opus panel (c0-c17) into `data/stimulus/pairs_v2_votes.tsv`:
  5,400 pairs, 3,559 accepted, kappa 0.377. By case: word-initial
  ع-deletion 2,842/3,887 (73%, all judged); vowel dropped beside a vowel
  717/1,513 (47%), with 1,404 more (all in c18-c22) unjudged.
- Re-emitted: mixed_train 266,782 / mixed_val 29,622 (normalize 294,503 incl.
  1,263 fused; meaning 1,901). Held back: 5,031 rows with a rejected pair,
  3,974 with an unjudged pair, 1,230 renderings of council-rejected fused
  sentences.

---

## 2026-09-26 (addendum 2) — pair council finished (mixed panel), dataset v2 final for this round

- User ruling: the last 1,404 pairs (chunks c18-c22, all "vowel dropped beside
  a vowel") are decided by the Sonnet/medium panel at 5/5 instead of Opus.
- Sonnet run: 30/30 judges returned (4.0M tokens, ~134k each vs Opus ~183k --
  the saving is small). Found: judges' self-reported counts are not reliable --
  skeptic_c19 reported 0 rejections but its file holds 88 (only mismatch of 30).
  Merges read the files, never the self-reports.
- New `council/combine_votes.py` (tested): each panel merged at its own bar,
  joined with a `panel` column; a pair decided twice is refused.
  `data/stimulus/pairs_v2_votes.tsv` now covers all 6,804 pairs:
  opus/xhigh>=4 3,559/5,400 accepted; sonnet/medium>=5 1,058/1,404.
  Per-panel TSVs `pairs_v2_votes_{opus,sonnet}.tsv`; per-judge files, chunks and
  panel maps tracked in `data/stimulus/council_v2/pairs/` (7.2 MB).
- Estimated cost of the Sonnet bar: Opus accepted 47% of this case, Sonnet 5/5
  accepted 75% -> ~390 of its accepts (~37%) likely Opus rejections; they touch
  2,795 emitted rows (0.9%), so ~1,000 rows (~0.35%) likely carry a collision.
- Final emit: mixed_train 269,306 / mixed_val 29,904 (normalize 297,298 incl.
  1,274 fused; meaning 1,912). Held back: 6,210 rows with a rejected pair, 1,230
  renderings of rejected fused sentences; nothing left unjudged.
- New `data/training/v2/DATASHEET.md`: files, formats, the held-out evaluation
  rule, provenance, verification, sized limitations, regeneration.

---

## 2026-09-26 (addendum 3) — round committed; judges recalibrated; scenario bank accepted; pilot fusion launched with a watchdog

- **Committed** the v2 round as `08813c2` on new branch `v2-dataset` (harness
  rule: branch before committing to main). Row files stay gitignored (user:
  the final dataset is not committed). `.codex/`, `AGENTS.md` and the
  `{filepath}.tmp` hook litter left out. Pre-commit scan: none of the three
  real `.env` values in the staged diff.
- **Judge policy, second change today:** the user proposed Opus on LOW effort.
  Calibrated on fused c0 against Opus x-high
  (`results/council_calibration_fused_v2_c0_opus_low_vs_xhigh.md`): kappa
  0.451 (Sonnet/medium on pairs was 0.055), close on anchor sense (0.68) and
  coherence (0.67), lenient on degeneracy/fluency/register; ~60k tokens per
  judge vs ~79k, 2.5x faster. At >=4/5, 29% of its accepts are x-high rejects;
  at 5/5, 12%. User chose >=4/5 for the pilot. `council_io.POLICY_PANELS`:
  opus/low = bulk; memory updated.
- **Scenario bank accepted.** An agent found 30 same-situation paraphrases
  that Jaccard missed and reworded the later id in each (ids 0-15 untouched;
  max Jaccard now 0.429); all 331 accepted via `apply_review`.
- `build_dataset_v2.py --fused` and `emit_v2.py --fusion-votes` are now
  repeatable so expansion rounds join the same master/emit. Defaults reproduce
  the current dataset exactly (row counts and file hashes verified).
- **Pilot launched** 2026-09-26 03:30: plan `data/generated/v3/fusion_plan.jsonl`
  (sha256 304a3f59...), 2,781 jobs = 281 trap + 2,500 bare anchors (500 per
  frequency band) x 1 of 331 scenarios, via `run_fusion_v2.sh` (batches of 200,
  60-min wait on a quota pause). Z.AI: user is fine running until the limit and
  waiting.
- **Watchdog:** session cron `71bf152f`, every 5 h at :23, 7-day expiry. It
  relaunches the loop from its checkpoint if the loop died (never on a fatal
  account error), and when the plan finishes runs the Opus-low council (>=4/5)
  into `data/stimulus/fusion_v3_pilot_votes.tsv`, reports by source and band,
  then deletes itself. It never rebuilds, emits or commits.

---

## 2026-09-26 (addendum 4) — a model outage now pauses the fusion loop instead of stopping it

- The pilot stopped twice (after 200 and 600 of 2,781 jobs), both times on 5
  empty deepseek-v4-flash replies in a row, exactly 3 h apart (04:49, 07:49);
  both times a test call worked again minutes later and the watchdog relaunched.
  User ruling: pause and retry instead of stopping.
- `fusion_jobs.RunGuard`: the breaker now reports `paused_model_down:<model>`,
  mapped to new exit code 76 (`EXIT_MODEL_DOWN`), distinct from the quota
  pause (75). `run_fusion_v2.sh`: on 76 it sleeps `--outage-wait-minutes`
  (default 10) and resumes; after `--max-outage-pauses` (default 12) outage
  pauses in a row with no progress (the checkpoint's jobs_done unchanged) it
  stops, so a truly dead model (Ollama crashed) still surfaces for the
  watchdog. Any progress resets the count.
- Tests first: updated the two status asserts, and added the exit-code mapping
  and three loop tests (outage resumes, no-progress cap gives up, progress
  resets the count). The suite passes.
- The running loop was restarted to pick this up (SIGTERM -> checkpoint
  `interrupted` at 800/2,781 -> relaunched; cached calls replay).

---

## 2026-09-26 (addendum 5) — DeepSeek "outages" were its reasoning budget; balance ran out

- 13:14: DeepSeek returned HTTP 402 Insufficient Balance (classified as a
  quota pause, so the loop slept hourly). The user recharged; a restart at
  13:45 then tripped the model-down breaker again within a minute.
- Root cause, measured with one real teacher prompt and the client's exact
  settings: deepseek-v4-flash at reasoning_effort "high" spent 3,795 / 4,463 /
  1,109 reasoning tokens on ONE short sentence; the 4,463 call hit the
  4,500-token cap (finish_reason "length") and returned a sentence cut
  mid-word. Calls whose reasoning used the whole cap came back EMPTY: the
  earlier "outages" at 04:49 and 07:49 were this, not DeepSeek downtime. The
  same long reasoning explains the drained balance (reasoning is billed as
  output).
- Fix (tests first, `tests/test_deepseek_client.py`): `MAX_TOKENS` 4,500 ->
  12,000 (billed per token used, so early finishers cost the same); a
  `finish_reason == "length"` reply now raises new `model_clients.TruncatedReply`
  (transient, so retried) instead of passing a cut-off sentence on as a draft.
  Previously a truncated reply that happened to pass the gates could have
  become a teacher draft; drafts from the first 800 jobs are not
  re-examined for this.
- The loop was restarted to load the fix (checkpoint 800/2,781; cached calls
  replay).

---

## 2026-09-26 (addendum 6) — DeepSeek budget: measured cost; a spent balance now ends the pilot

- The user added $5 of DeepSeek credit and does not want to add more. Measured
  from the balance endpoint (`api.deepseek.com/user/balance`) against the
  cache's DeepSeek answer count: ~$0.0035-0.0056 per answer (the high end from
  a 27-answer window during the slow, long-reasoning tail). Finishing the
  pilot's ~1,780 remaining DeepSeek answers would take ~$6-10; the $4.51 left
  covers ~800-1,300, i.e. the pilot should reach ~1,800-2,300 of 2,781 jobs.
- That costs little: the plan is anchor-major with trap anchors first, and all
  281 trap jobs (the only source of meaning rows) are done; every remaining
  job is a bare anchor (normalize rows only, next to ~297k from NileChat).
  Estimated pilot contribution: ~+3,200 rows at ~2,000 jobs vs ~+4,200 at
  2,781; the dataset is ~300k rows either way.
- User decision: stop when the credit is gone and use what was finished.
  `deepseek_client.classify`: HTTP 402 is now `FatalAPIError`, not
  `QuotaExhausted` -- a spent prepaid balance never refills by waiting, so a
  quota pause would have retried hourly forever. Tests updated first
  (`tests/test_api_errors.py`); the suite passes. The loop was restarted to load it.
- Watchdog replaced: cron `63f4961a` (was `71bf152f`), same schedule. It now
  treats `failed_fatal:deepseek` + 402 as the end of the pilot and runs the
  Opus-low council (>=4/5) on whatever was finished.
- Free alternative, not taken: finish the remaining jobs with the other 4
  teachers (Z.AI flat plan + local models) as a separately labeled plan.

---

## 2026-09-26 (addendum 7) — pilot stopped and judged; exact NileChat counts; trap-scenario run launched; GLM council built

- **Why fusion rows are few** (the user asked): fusion yields at most one
  sentence per job (~20 s of model time each), so volume comes from the
  NileChat respell. The current dataset has 299,210 rows: 296,024 (98.9%)
  respelled NileChat -> original, 1,274 respelled fused -> fused, 1,912
  meaning rows, over 151,704 NileChat originals and 703 fused sentences.
- **Exact NileChat** (same splitter and filters as the collector, full pass):
  571,524 docs, 13,877,740 sentences, 206,506,425 words; 6,488,963 pass every
  filter; 152,707 are used (2.4%), because the 8-sentences-per-word cap binds.
- **Pilot stopped at 1,000/2,781** (user; saves DeepSeek credit; all 281 trap
  jobs were done). 956 gate-passed. Opus-low council (>=4/5, 45 judges, 2.7M
  tokens): `data/stimulus/fusion_v3_pilot_votes.tsv`, 364/956 accepted. Trap
  anchors 189/270 (70%); bare 175/686 (26%, 18-33% by band). Bare anchor-lens
  rejections are mostly genuine: rare formal words forced into chat scenarios
  ("el makhderat keteer awi" about rent). Bare-anchor fusion is not worth scaling.
- **Trap-scenario run launched** 2026-09-26 14:46 (user: trap words only, keep
  fusion, drop DeepSeek, ~2 days): plan `data/generated/v4/fusion_plan.jsonl`
  (sha256 c12d0781...), 9,533 jobs = 284 trap words x 35 of 331 scenarios
  minus 407 done; teachers llama3.1:8b, gemma4:e2b, nemotron-3-nano:30b,
  zai:glm-5.3-flash; fusor zai:glm-5.3; 48-hour cap ends 2026-09-28 14:46.
- **GLM council** (user: judge with glm-5.3 on the flat Z.AI plan, no Claude
  tokens): new `council/glm_council.py`. Lens texts and framing are read from
  the committed `fused_council.js` / `pair_council.js` (everything before the
  agent-only "Read the chunk file" step), so the criteria are identical. Items
  are inlined in batches of 25, verdicts are matched by sid, a malformed batch
  is re-asked up to 3 times, quota sleeps and retries, and it resumes by file.
  8 tests. Caveat: glm-5.3 is also the fusor, so it judges its own sentences;
  calibration against Opus x-high on the 1,122 v2 fused sentences is running
  (`data/generated/v2/council/fused_votes_glm`).

---

## 2026-09-26 (addendum 8) — GLM-5.3 council calibrated; bar 3/5 for the trap run; watchdog runs it

- GLM council hardening, found live: GLM garbled near-identical sids
  (fs_f2dan_00 -> fs_f2dan_0) until a batch failed 3 retries and the first
  calibration crashed at 29/50 files. Items are now keyed by a small index i
  (1..N) mapped back to sids, and a batch that fails every re-ask leaves only
  its file pending (exit 3) instead of killing the run. Tests updated first;
  the suite passes. The calibration was redone cleanly on one prompt format
  (the partial run is kept at fused_votes_glm_sidformat_partial).
- Calibration, GLM-5.3 vs Opus x-high on the 1,122 v2 fused sentences
  (`results/council_calibration_fused_v2_glm53_vs_opus_xhigh.md`): at >=4/5,
  kappa 0.348, GLM rejects 20.1% vs 37.0%, and 27.8% of its accepts are
  x-high rejects (close to Opus-low's 29%); per-lens kappa is anchor 0.48,
  register 0.09. Label-critical error (accepted sentences whose word sense
  x-high's anchor lens calls wrong): x-high itself 0.4%; GLM >=3/5 12.1%,
  >=4/5 7.9%, 5/5 3.1%. Requiring the anchor lens on top of >=4/5 barely
  helps (27.8% -> 27.4%): GLM's misses are mostly style.
- User bar: >=3/5 (keeps 93%). Expected on the trap run: ~20,000 new meaning
  rows, ~12% (~2,400) with a wrong-sense label -- sized here and to be stated
  in the datasheet when they join the dataset.
- Watchdog `76630cdc` (replaces `55dc17f3`): fusion health and the 48-hour cap,
  then chunk -> GLM council (detached, resumable) -> merge at --min-agree 3 into
  `data/stimulus/fusion_v4_trap_votes.tsv` -> report -> delete itself. No
  Claude judges, no rebuild, emit or commit.
- Trap run at 16:03: 200/9,533, ~20 s/job, so the cap will stop it at roughly
  8,500 jobs.

---

## 2026-10-17 — Public release prep: history audit, README rewrite, D1 restored, licenses, CI

- **Secret audit of all history before going public.** A scanner (session
  scratchpad, not committed) checked 18 credential formats covering every
  provider this project used, plus secret-named assignments and high-entropy
  strings. Positive control: 9 planted fake keys in a throwaway repo, all
  caught, including after deletion. Scope: all 22 commits on both branches (587
  blobs) and every object in the local ODB (10,164 blobs, incl. 490 dropped-stash
  commits). **Zero credentials.** `.env` (live Z.AI/DeepSeek/OpenRouter keys)
  was never committed; the deleted `.env.example` held empty placeholders; all
  old collector/test scripts read keys from env vars. GitHub side: no forks,
  PRs (so no hidden `refs/pull/*`), releases or Actions runs; all 490 stash
  commits verified absent on GitHub via the commits API (control: a pushed
  commit FOUND, a fake SHA absent). No scraped tweets were ever committed (raw
  collector output was gitignored); the only phone number is the fake
  `01012345678` test fixture; no `/Users/` paths or hostnames anywhere.
- **History (user ruling): one fresh commit for the public repo.** The 22-commit
  history (2026-05-05 → 2026-09-28) was replaced on GitHub by one "initial public
  release" commit (force-pushed main, remote v2-dataset deleted); it is kept
  privately in the local branches `archive/main` and `archive/v2-dataset`. SHAs
  in this file and in the d5 manifest refer to that archive.
- **D1 restored (user ruling).** `experiments/d1_benchmark/run_showcase.py`
  came back from `407018f^`: it was deleted while still the README's reproduce
  command and the only generator of `variant_clusters.tsv`, the human arm's
  input. With today's library, `variant_clusters.tsv` and `pairs.csv` are
  byte-identical and the attested headline is unchanged (0.850 → 0.871). The
  synthetic sweep moved by ≤ 0.033 (the 2026-08-22 taxonomy anchoring fixes), so
  `results/showcase.*` and `data/benchmark/retrieval_v0.jsonl` were regenerated
  (deterministic across runs and in a fresh copy; no code reads retrieval_v0).
- **Stale numbers corrected**, each re-derived from committed JSON:
  `results/README.md` Level-3 table now matches `level3_meaning.json` (it
  predated the final regeneration, e.g. qwen over-norm 0.140 → 0.138, aya
  agreement 0.261 → 0.265); joint ρ 0.20 → 0.24 (what F9 prints and
  `series.frag_vs_flip_points` recomputes over 44 cells); "pre-registered
  prediction held" → **half held** (DIGIT_SAFE is worst of the n ≥ 20 classes
  on the NileChat arm, but FINAL_Y_I is mid-pack and TAA_MARBUTA, added after
  pre-registration, is least); 221 → 587 tests. The d5 README AUC band went from
  0.55–0.62 to 0.56–0.63; `01_benchmark.md` now says 31 rules and 57 clusters,
  and its dead notebooks link is gone.
- **README rewritten for public readers** around the d5 results (three levels,
  the v2 dataset, the Fusion-of-N pilot), with light/dark figures via
  `<picture>`, a "how the numbers stay honest" section, a quickstart verified
  from a fresh copy, limitations, license and acknowledgements (NileChat
  card's own citation; the FusioN paper's Cohere Labs affiliation checked on
  arXiv).
- **New files:** `LICENSE` (MIT, code), `data/LICENSE.md` (CC BY-NC 4.0 and
  what derives from NileChat), `CITATION.cff`, `.github/workflows/tests.yml`
  (stdlib suite on Python 3.14 with nothing installed, read-only token,
  SHA-pinned actions).
- **Tidy:** `docs/superpowers/specs/*` → `docs/design/`; stray 0-byte
  `{filepath}.tmp` removed; `.gitignore` now covers `.env.*`,
  `.claude/settings.local.json` and `.kilo/` in-repo (before, only a global
  ignore caught settings.local.json); `AGENTS.md` regenerated from CLAUDE.md (it
  was stale and had a `.Codex/agents/` typo); CLAUDE.md structure, test count
  and history updated; `training_data/README.md` points to v2; the research
  plan says where pre-release history lives.
- **Verified:** a fresh copy of exactly the 444 files to be published, run in a
  bare Python 3.14 venv with no packages, passes all 587 tests (12
  dataset-contract tests skip without `data/generated/`; locally they pass
  against the real 299k-row build) and reproduces the D1 showcase
  byte-for-byte. All 52 relative links and images resolve. The release tree
  scans clean: 0 credential hits, 22.5 MB.
- **Independent fact-check** (franco-evaluator, read-only, every README number
  re-derived from committed JSON/TSV). The arithmetic held; the framing did not,
  and each point below was re-verified before fixing. Overclaims removed from
  README, DATASHEET and the results README:
  - held-out words "never appear in training": they are never respelled,
    created, anchored or asked about, but 577 of the 598 held-out-side words occur
    unchanged as context in 90% of train rows;
  - "158,452 sentences" respelled: 152,707 (5,745 of the 158,452 collected had
    < 2 usable words);
  - anchor Arabic/meaning "never model-written": they come from the LLM trap
    council;
  - Best-of-N here = the first gated teacher draft, with no reward model, and the
    win rates count ties as non-wins;
  - "no configuration beat a frontier model" holds only for deepseek-v4-pro
    (glm-5.3 roughly tied v4-flash);
  - the Llama 3.1 mirror has no signature pin;
  - the trap run is 8,695 judged / 7,153 accepted (the "9.6k" was planned jobs);
  - the ρ analysis deviates from the per-item pre-registration (now disclosed,
    p ≈ 0.12);
  - DIGIT_SAFE "as pre-registered" (the n ≥ 20 cutoff is post hoc, and 48% of
    those pairs carry a second class);
  - temperature 0 makes the ceiling ≈ 1 by construction;
  - only the best council κ (0.875) was quoted; all four are now reported
    (0.875 / 0.39 / 0.29 / 0.12);
  - human-arm SAME/DIFFERENT labels are Claude-drafted partitions, so they are
    added to the native-review limitation;
  - D1's +0.021 is noisy (per-seed −0.018 to +0.052);
  - the 93% co-skeleton figure is not regenerable, so it was softened;
  - on Linux CI, 10 more tests skip (macOS `caffeinate`).
- **Open follow-ups, not fixed here:**
  - `merge_council.fleiss_kappa` truncates every item to the minimum rater
    count; one trap (shagara/shagar) has 4 votes, which drops a lens everywhere.
    Recomputed over all 5 judges, κ = 0.885 vs the committed 0.875.
  - The 93% co-skeleton statistic should be written by `build_wordset.py` into
    `wordset_diagnostic.json`.
  - The f3 "Claim" string in `render_all.py` / `figures/index.md` still states
    the prediction as the claim.
- **First CI run (Linux) caught a real bug the Mac never showed.**
  `council_io.wilson_interval(0, 10)` returned 2.8e-17 instead of 0.0 under
  glibc. The bug is general: even on macOS, 315 of n = 1..2000 leave residue at
  0 successes and 601 at n successes (e.g. 0.9999999999999999 for 4/4). Tests
  first: a new subTest over 8 values of n (RED on macOS), then exact endpoints
  at 0 and n successes (GREEN). No committed output contained the residue, so
  no published number changes. 588 tests. The fix was folded into the single
  release commit before the repo went public.
