# Franco-Arabic (Egyptian Arabizi) Research Plan — Orthographic Robustness

*Prepared 2026-07-23. Grounded in a verified literature and dataset sweep (two Opus research agents, mid-2026 web state) plus this repo's existing assets (seed data, cleaning pipeline, fusion generator, NileChat eval notebook).*

> **Amendment 2026-08-20 — the collection pipeline is retired.** The
> scrape-and-fuse data acquisition approach has been dropped. `collect_x.py`,
> `collect_sources.py`, `clean_franco.py`, and `fusion_generate.py` exist only
> in git history before commit `ab700b1` (pre-release history, kept privately by
> the author since the 2026-10-17 public release); all API credentials and scrape-era
> working data have been deleted. **§4 (data plan), §5 line on YouTube/X
> comments, and the §8 risk mitigations that name these scripts are stale as
> written** — any "fresh collection" they assume needs a new sourcing decision
> (licensed corpus download, manual authoring, or a fresh consented collection
> designed from scratch). Everything measured in `results/` today derives from
> `data/seed/*.jsonl` alone.

---

## 1. Problem framing

Your instinct — "this is not just fine-tune on Franco data" — is correct. But the framing "is it a normalization problem, a tokenization problem, or a transliteration problem?" has the order backwards. Those are **interventions**, not the problem. Nobody can currently say which intervention matters for Egyptian Arabizi because **no measurement instrument exists**. Verified facts:

- **No canonical Latin-side orthography for Arabizi exists.** CODA/CODA* standardize the *Arabic-script* side only; Buckwalter/ALA-LC are deterministic MSA romanization schemes, not descriptions of how people write Franco. "Normalize to canonical Romanized form" therefore means *you defining the standard* — a possible contribution, but not something to assume.
- **Transliteration to Arabic script is a mature line** (Darwish 2014; Al-Badrashiny et al. 2014 "3ARRIB"; Shazal, Usman & Habash 2020 — 80.6% word accuracy with seq2seq). The gold Egyptian parallel corpus is LDC-locked (LDC2017T07, paid). Recent transformer work exists for Moroccan/Tunisian; Egyptian is explicitly called "underexplored" in 2025 work.
- **The robustness gap is empirically real.** Al Ghanim et al. (EMNLP 2024) showed the same harmful prompts that GPT-4 and Claude refuse in Arabic script succeed as jailbreaks when written in Arabizi — direct evidence that models treat semantically identical content inconsistently across orthographies. Riabi et al. (2021) showed character-level models beat subword models on NArabizi; DziriBERT outperforms MARBERT on romanized input.
- **Nothing measures it.** No benchmark renders the same Egyptian Arabizi content in multiple naturalistic spelling variants with a prediction-consistency metric. No matched-condition comparison of raw-subword vs. normalization vs. transliteration vs. byte-level pipelines exists for any Arabizi. No Arabic evaluation infrastructure (Open Arabic LLM Leaderboard, AraGen, Dolphin/ORCA, DialectalArabicMMLU) has an Arabizi track.

**Corrected framing:** this is a **robustness-measurement problem over orthographic variation**, where normalization (to a Romanized canon you define), transliteration (to Arabic script), and robust modeling (byte-level / augmentation) are *competing interventions to be evaluated under matched conditions*, not assumptions. Classification/retrieval are the task vehicles. Tokenization enters as an *analysis lens* (segmentation instability across variants), not an engineering deliverable.

**Main contribution type: evaluation methodology + data.** Secondary: empirical guidance on which pipeline buys the most robustness per unit of effort.

**Dialect scope: keep Egyptian.** Largest Arabizi-writing population, you are a native speaker (annotation and plausibility-judgment bottlenecks disappear), this repo already has a human-authored seed set and collection/cleaning pipeline, and two 2025 Egyptian LLM families exist to evaluate (UBC-NLP NileChat-3B; MBZUAI-Paris Nile-Chat 4B/12B/MoE — two distinct projects, don't conflate). Design the benchmark schema dialect-agnostic so NArabizi (Algerian) and TArC (Tunisian) plug in later — that's your multi-dialect extension story, not your starting point.

---

## 2. Candidate directions

None of these requires training a new tokenizer. A and D require no pretraining at all — only fine-tunes and analysis.

### Direction A — "2ol-tany" (*say it again*): a variation-calibrated robustness benchmark + controlled pipeline bake-off

- **Problem statement:** Given the same Egyptian Arabizi content realized in k plausible spelling variants, measure (i) prediction consistency and (ii) accuracy degradation of current models, and determine which of four input-handling strategies — raw subword, light normalization, transliteration-to-Arabic-script, byte-level — is most robust under matched fine-tuning budgets.
- **Why it matters:** Arabizi is the daily written register of tens of millions of Egyptians; every deployed model silently varies its behavior with spelling (the jailbreak result shows this has safety implications, not just quality ones). The field has intervention proposals but no instrument to compare them.
- **Novelty:** Moderate-high. Consistency benchmarking exists for paraphrases (PARAREL) and typos (MulTypo 2026, Arabic *script* only); the with/without-spelling-variation test-set design exists for Roman Urdu POS (2025). Nobody has done either for Egyptian Arabizi, and nobody anywhere has done the matched pipeline comparison.
- **Risks:** Synthetic variants may not match real variation (mitigated by calibrating on mined real variant pairs and testing on held-out *human* variants); annotation bandwidth.
- **MVP (end of week 4):** 300–500 human-authored messages × 4 variants, sentiment labels with agreement stats, 3 systems, consistency + degradation table.
- **Ideal final deliverable:** ~1–2k messages, 2–3 tasks, 8+ systems including both NileChats, degradation curves, tokenizer-instability analysis, public HF dataset with datasheet, workshop-paper-style report.

### Direction B — Modern Egyptian Arabizi→Arabic transliterator via a synthetic-data recipe

- **Problem statement:** Train a lightweight (ByT5-small-class) Arabizi→Arabic-script transliterator *without* the paid LDC corpus, using reverse-romanization of free Egyptian-script corpora (ETC, ASTD) with sampled orthographic variation, plus a small gold set you annotate; evaluate CER/word accuracy and the downstream gain when prepended to Arabic-script models.
- **Why it matters:** The only strong Egyptian transliterators are 2014–2020 systems trained on paid data; a free, reproducible recipe would be widely used (the only open Franco→Arabic model on HF is an undocumented personal T5-small).
- **Novelty:** Low-moderate — the task is classic; the *free synthetic recipe + modern model + downstream evaluation* packaging is the new part.
- **Risks:** Synthetic-to-real gap in romanization style; you'd be evaluated against LDC-trained numbers you can't fully reproduce.
- **MVP:** Rule-based Egyptian romanizer + ByT5-small on 50k synthetic pairs, CER on 200 hand-annotated gold sentences.
- **Ideal final deliverable:** Released model + synthetic-data generator + gold test set + downstream-delta results.

### Direction C — Variant-invariant sentence embeddings via contrastive fine-tuning

- **Problem statement:** Fine-tune a multilingual encoder (multilingual-e5 / LaBSE) with a contrastive objective pulling spelling variants of the same content together; evaluate retrieval@k and classification robustness under variant noise.
- **Why it matters:** Retrieval/RAG and moderation over Arabizi content break when "a7a" and "aha" land in different embedding neighborhoods.
- **Novelty:** Moderate-high as a method application; but evaluation is subtle and it *needs Direction A's benchmark to exist first*.
- **Risks:** Easy to fool yourself (the perturbation engine generates both training positives and test sets — circularity must be controlled); 4–8 weeks is tight for method + eval.
- **MVP:** e5 + kNN classifier on the Direction-A benchmark, before/after contrastive fine-tuning.
- **Ideal final deliverable:** A released "franco-robust" embedding model + retrieval benchmark results.

### Direction D — Tokenizer segmentation-instability study (analysis-only)

- **Problem statement:** Quantify, across major tokenizers (Gemma, Llama, Qwen, XLM-R, mT5, and Arabic-script tokenizers post-transliteration), how token fertility and segmentation boundaries diverge across spelling variants of the same word/sentence ("a7a" → `a·7·a` vs "aha" → `aha`), and whether instability predicts downstream prediction flips.
- **Why it matters:** Extends the tokenizer-fairness literature (Petrov et al. 2023) to the unstudied case of *intra-language orthographic variation* in Latin script with digit intrusions, which breaks BPE merges in a characteristic way.
- **Novelty:** Moderate. Strong as a *chapter*; thin as a standalone 8-week project.
- **Risks:** Correlation-only findings; no intervention.
- **MVP:** Fertility + boundary-divergence table across 6 tokenizers on 1k variant pairs. (~3 days of work.)
- **Ideal final deliverable:** The analysis chapter inside Direction A, with the flip-correlation result.

---

## 3. Direction ranking table

| Direction | Research value | Feasibility (4–8 wk) | Internship signal | Dataset availability | Evaluation clarity |
|---|---|---|---|---|---|
| **A — Benchmark + bake-off** | **High** (fills verified gap; negative results still publishable) | **High** (no pretraining; fine-tunes fit Colab) | **High** (shows measurement taste, the #1 research-intern skill) | **High** (self-created from own seed + free corpora) | **High** (accuracy, consistency, CER — all crisp) |
| B — Free transliterator | Medium (classic task, new recipe) | High | Medium-high | Medium (gold is paid; synthetic recipe needed) | High (CER/word acc) |
| C — Invariant embeddings | High (method contribution) | Medium (needs A first; circularity traps) | High if it works | Medium | Medium (subtle) |
| D — Tokenizer instability | Medium (good chapter, thin alone) | High (days, not weeks) | Medium | High | High |

---

## 4. Recommended path

**Direction A as the spine, absorbing D as its analysis chapter and a minimal B as one of the compared systems.** Concretely: build the benchmark (A), include the tokenizer-instability study as descriptive evidence (D), and implement the transliteration arm with an off-the-shelf or quickly-trained model (mini-B). Direction C becomes the natural *follow-up* — ideally the internship project itself, proposed on top of your benchmark.

Why this wins:

1. **Measurement-first is the credible move.** Proposing "we need byte-level models for Arabizi" without evidence is advocacy; showing a table where byte-level wins (or doesn't) under matched conditions is research. Either outcome is interesting: if raw subword + variant augmentation closes the gap, that's a deployable, citable negative result about normalization being unnecessary.
2. **It produces durable artifacts regardless of results:** a public dataset with a datasheet, a perturbation library, and a leaderboard-style repo — exactly what a research lead can evaluate quickly.
3. **It leverages what you already have:** the human-authored seed conversations, `clean_franco.py`'s Arabizi scoring, the collection pipeline, and your NileChat eval notebook (both NileChat families become evaluated systems, giving your benchmark immediate relevance to two active groups — UBC-NLP and MBZUAI-Paris).
4. **It's honest about the mock-fusion data:** the 10k fusion-generated examples are fine for chat fine-tuning experiments but must NOT be benchmark material; the benchmark draws only from human-authored and freshly-collected text.

---

## 5. Execution plan

### Milestones by week (8-week plan; 4-week cut line marked)

**Week 1 — Variation taxonomy + real-variant mining.**
Deliverables: `docs/variant_taxonomy.md` — an inventory of Egyptian Franco phenomena with examples and estimated frequencies (digit↔letter maps: 7↔h, 3↔a/aa, 2↔a/q/', 5↔kh/x, 8↔gh, 6↔t, 9↔s; vowel epenthesis/deletion: "ezayak"↔"ezayyak"↔"izayak"; e↔i, o↔u; gemination; sh↔ch; taa marbuta -a↔-ah; hamza drops; elongation "gedddan"; English code-switch spellings). Plus a mining script: cluster your corpus by a "Franco skeleton key" (map digits→phoneme class, collapse vowels and gemination — Soundex-for-Arabizi) and confirm clusters with edit distance, producing `variant_clusters.tsv` of *real, attested* variant pairs ranked by frequency.
**Your contribution (this genuinely shapes the project):** hand-author the top ~20 variant rules with plausibility weights as a native speaker. This taxonomy is the scientific core — a wrong weight here propagates into every downstream number, and no model can substitute your intuition about what Egyptians actually type.

**Week 2 — Perturbation engine + canonical normalizer.**
`franco_perturb`: character-level rewrite rules sampled by mined frequencies, with an intensity parameter. Calibration gate: 100 generated variants judged by you; ≥85% must read as natural. Also `franco_norm` v0: a *documented* canonical Romanized form (always digits: 7 not h, 3 not a; single vowels; no gemination) + deterministic mapper — this both defines the "light normalization" arm and is a mini-contribution (no such canon exists; say so and cite CODA* as the Arabic-script analog). Unit tests for both.

**Week 3 — Benchmark v0 construction.**
Select 500–800 *human-authored* messages (seed set + fresh collection via your pipeline; exclude mock-fusion data). Label: 3-way sentiment + ~8-way topic/intent. Double-annotate 150 (recruit one friend), report Cohen's κ, adjudicate. Render each message in k=4 forms: original, canonical, 2 calibrated synthetic variants; for a 150-message subset add 1 *human-written* variant (you rewrite, or mined). Annotate a 200-sentence gold Arabizi→Arabic transliteration set (CODA-guided, sanity-checked with `camel_tools`).

**Week 4 — Wave-1 systems + first results table. ← 4-week cut line: this alone is a presentable mini-study.**
Char-n-gram TF-IDF + linear SVM; XLM-R-base fine-tune; ByT5-small fine-tune. Report clean accuracy, per-variant accuracy, consistency C@k, worst-variant gap.

**Week 5 — Wave-2 pipelines.**
Normalization arm (`franco_norm` → XLM-R). Transliteration arm (off-the-shelf `mohamedtolba/franco-arabic` T5-small, else ByT5-small trained on 50k synthetic reverse-romanized ETC/ASTD pairs → MARBERT or CAMeLBERT-DA downstream). LLM arms: NileChat-3B, Nile-Chat-4B, Gemma-class few-shot, each ± normalization pre-processing. Embedding arm: multilingual-e5 + kNN.

**Week 6 — Robustness analyses.**
Degradation-vs-intensity curves (perturbation probability 0.1→0.5). Held-out **real-human-variant** test vs. synthetic-variant test (the defense against "you only tested on your own noise model"). Tokenizer-instability chapter: fertility + segmentation-boundary divergence across variants for 6 tokenizers, correlated with per-example prediction flips.

**Week 7 — Ablations + error analysis.**
Key ablation: **training-time variant augmentation** — fine-tune each wave-1 model on perturbed training data and measure how much of the robustness gap closes (if augmentation ≈ byte-level, that's the headline). Perturbation-type ablations (digits-only vs. vowels-only). Data-scale curve (500/2k/10k training examples). Error analysis by phenomenon.

**Week 8 — Packaging.**
HF dataset + datasheet + annotation guidelines; README with leaderboard table; 4-page workshop-style report; 10-slide deck for the mentor conversation.

### Repo structure

```
franco-robust/
├── README.md                  # leaderboard table front and center
├── data/
│   ├── raw/  interim/         # gitignored
│   └── benchmark/             # versioned: test sets, variants, gold translit
├── src/franco_robust/
│   ├── mining/                # skeleton-key clustering, variant pair extraction
│   ├── perturb/               # calibrated variant generator
│   ├── normalize/             # canonical Romanized form + mapper
│   ├── translit/              # reverse-romanizer, translit front-end wrappers
│   ├── tasks/                 # dataset builders, model wrappers per arm
│   └── eval/                  # consistency, degradation, tokenizer metrics
├── configs/                   # one YAML per (system, task) run
├── notebooks/                 # Colab fine-tune/eval notebooks
├── results/                   # committed CSVs + plots per run
├── tests/
└── docs/                      # variant_taxonomy.md, datasheet.md, annotation_guidelines.md
```

---

## 6. Data and benchmarks

### Truly public (verified live, mid-2026)

| Resource | What | Use here |
|---|---|---|
| **UBC-NLP/nilechat-arabizi-egy** (HF, CC-BY-NC-4.0) | 572k rows Egyptian Arabizi — **synthetic** (Command-R-Plus transliteration of Egyptian-script web text) | Variant mining at scale; NOT benchmark text (synthetic + contamination risk for NileChat evals) |
| **ETC — Egyptian Tweets Corpus** (HF `faisalq/ETC`, CC-BY-4.0) | 34.3M Egyptian tweets, Arabic script | Source for synthetic reverse-romanization (transliterator training) |
| **ASTD** (GitHub) | 10k Egyptian tweets, 4-way sentiment, Arabic script | Reverse-romanize *with labels carried through* — free labeled Arabizi classification data |
| **MBZUAI-Paris Nile-Chat suite** (HF, Gemma license) | EgyptianBench 11.4k incl. Arabizi↔Arabic transliteration; SFT/DPO sets | Comparison eval + systems to benchmark |
| **NArabizi treebank + enrichments** (Algerian; UD, NER, sentiment layers) / **TArC** (Tunisian, 43k tokens) / **DODa, DarijaBERT-arabizi** (Moroccan) | Open cross-dialect Arabizi | Method transfer; multi-dialect extension later |
| Habibi lyrics (EGY subset), ArSAS, EDC | Egyptian-script corpora | Additional reverse-romanization sources |
| Your repo: 148 seed conversations + ~60 slang additions | Human-authored Egyptian Franco | **Benchmark core** |

### Weakly supervised
YouTube/X comments collected via your existing `collect_sources.py`/`collect_x.py` + `clean_franco.py` Arabizi scoring; spontaneously co-occurring variant pairs mined by skeleton-key clustering (labels come free: same-cluster = same word).

### Synthetic (you generate)
Calibrated perturbation variants of benchmark items; reverse-romanized ETC/ASTD pairs for the transliteration arm; the existing 10k fusion set (chat fine-tuning experiments only — never benchmark).

### Manual (you annotate)
~650 sentiment/topic labels (150 double-annotated), 200 gold transliterations, ~150 hand-written real variants, plausibility judgments. Total: roughly 3–4 focused days for a native speaker — the single highest-leverage manual investment in the project.

### Key locked resource (name it, don't buy it)
LDC2017T07 (BOLT Egyptian SMS/chat, 287k Arabizi words gold-transliterated to CODA) — cite as the paid gold standard your free recipe substitutes for.

### The simplest benchmark that still looks serious ("2ol-tany v0")
300 human-authored messages, 3-way sentiment, double-annotated 20% with reported κ, k=4 renderings each (original / canonical / calibrated-synthetic / **real human variant**), 3 systems (SVM, XLM-R, NileChat-3B few-shot), reporting clean accuracy, consistency C@k, worst-variant gap, bootstrap CIs. What makes it serious despite the size: the calibration against mined real variation, the real-vs-synthetic variant split, and the agreement statistics. This is ~1 week of work once the perturbation engine exists.

---

## 7. Modeling options

### The mandated comparison (these are the benchmark's arms)

| Strategy | What it is | Cost | Expected behavior | Verdict for weeks 1–8 |
|---|---|---|---|---|
| **Standard fine-tuning (raw subword)** | XLM-R / Gemma-LoRA on raw Franco | Lowest | Decent clean accuracy, poor consistency (BPE fragments digit-infixed words unpredictably) | Mandatory baseline |
| **Light normalization + fine-tuning** | `franco_norm` canonical form → same models | Low | Should recover much consistency *if* the canon is good; errors compound where normalizer guesses wrong | Mandatory arm; cheapest intervention |
| **Transliteration pipeline** | Arabizi→Arabic script → MARBERT/CAMeLBERT-DA | Medium | Unlocks strong Arabic-script models; transliterator CER becomes the bottleneck — report it separately | Mandatory arm (off-the-shelf first; train ByT5-small on synthetic pairs only if needed) |
| **Character/byte-level** | ByT5-small / CANINE fine-tune | Medium (slower training) | Literature predicts best consistency (Riabi 2021, ByT5 noise results); watch clean-accuracy cost | Mandatory arm |
| **Retrieval/embedding-based** | multilingual-e5 / LaBSE + kNN; optional contrastive fine-tune | Low | Measures representation robustness directly; the contrastive upgrade is Direction C | kNN arm mandatory; contrastive = stretch goal |

### By complexity tier
- **Easiest baseline:** char-n-gram TF-IDF + linear SVM (embarrassingly strong on noisy short text — if neural systems don't beat it under perturbation, that *is* a finding), fastText.
- **Moderate:** XLM-R-base, ByT5-small, CAMeLBERT-DA/MARBERT (post-transliteration), multilingual-e5. All fit free-Colab budgets you already work in.
- **Stronger research baseline:** NileChat-3B (UBC-NLP), Nile-Chat-4B (MBZUAI-Paris), Gemma-class LoRA fine-tunes via your existing Unsloth setup; few-shot GPT/Claude-class models as reference points if API budget allows.

### Is custom tokenizer work necessary at the start? **No.**
Benchmark the four pipelines above first — that's precisely the question your instrument answers. Do include the tokenizer *analysis* (fertility + segmentation divergence across variants; ~3 days, one strong figure). Tokenizer *engineering* (vocabulary extension, retraining, transliteration-aware merges) is justified only if the analysis shows segmentation instability strongly predicts prediction flips AND byte-level models close the gap at unacceptable inference cost. That conditional finding is itself the perfect *follow-up* project to pitch for the internship — grounded in your own measurements.

---

## 8. Evaluation design

**Task metrics.** Classification: macro-F1 (clean and per-variant). Transliteration arm: CER/word accuracy against your 200-sentence gold (report separately from downstream, so pipeline errors are attributable). Retrieval: recall@k over variant queries.

**Consistency (the benchmark's signature metric).**
- `C@k` = fraction of items where the model's prediction is identical across all k renderings.
- Mean pairwise agreement (more forgiving, better for large k).
- **Variant Sensitivity Gap** = clean accuracy − worst-variant accuracy.

**Degradation curves.** Accuracy/F1 vs. perturbation intensity (per-word perturbation probability 0.1→0.5); report slope, not just endpoints.

**The mandated test-time robustness test (two levels).**
1. *Synthetic:* every test item served in k calibrated variants at test time — no system sees perturbed test text during training (except the augmentation-ablation arms, reported separately).
2. *Real:* a held-out set of **human-produced** variants (hand-written + mined attested pairs) that the perturbation engine never saw. Systems ranked differently on real vs. synthetic noise would itself be an important finding about synthetic-robustness evaluation.

**Tokenizer-instability metrics.** Fertility ratio per variant; boundary divergence = 1 − Jaccard overlap of token-boundary offset sets between variant pairs; point-biserial correlation with per-example prediction flips.

**Rigor guardrails.** Bootstrap 95% CIs on every headline number; Cohen's κ on double-annotated labels; contamination control — benchmark text must not derive from fineweb-edu-Egypt/LHV-Egypt (NileChat's synthetic Arabizi sources), so both NileChats are evaluated fairly; report zero-shot and fine-tuned results separately (LLMs have unknown incidental Arabizi exposure from pretraining — a confound to surface, not hide).

---

## 9. Risks and scope control

| Risk | Likelihood | Mitigation |
|---|---|---|
| Synthetic variants don't match real variation | High — the central methodological threat | Calibrate rules on mined attested pairs; plausibility gate (≥85%); held-out real-variant test set |
| All systems statistically tie | Medium | Still a paper: "augmentation suffices; normalization unnecessary" is deployable guidance. Design tasks hard enough (slang-heavy, short texts) to create headroom |
| Transliterator errors dominate its pipeline arm | Medium | Report translit CER separately; include an oracle-transliteration ceiling (your gold set as input) |
| Annotation agreement too low on sentiment | Medium | Collapse to polarity; written guidelines; adjudication pass |
| Data scarcity for benchmark text | Low | Seed set + your collection pipeline already suffice for 500–800 items |
| LLM pretraining contamination | Medium | Source control (above); fresh-scrape subset with post-2025 timestamps |
| PII / licensing on scraped text | Medium | Your `clean_franco.py` PII stripping already handles this; keep provenance + license fields (already in your schema); CC-BY-NC data stays out of anything commercial |
| Scope creep | High | Hard rules below |

**Hard scope rules:** No tokenizer training. No pretraining from scratch. No RL. Fixed system list frozen at week 5. First results table exists by end of week 4 or the system list shrinks. Anything that can't produce a number within one week gets cut. The mock-fusion 10k set never touches the benchmark.

**Expected failure modes to plan around:** perturbation engine over-generating implausible forms (gate catches it); ByT5 training slow on T4 (use `-small`, short sequences — your texts are ≤280 chars); NileChat inference memory on free Colab (4-bit quantize, batch 1); embedding arm looking artificially good because kNN memorizes near-duplicates (dedupe by skeleton key before splitting — your cleaning pipeline's dedupe is exact-match only, which is insufficient here).

---

## 10. How to present this to a research lead

**Elevator pitch (use verbatim):**

> "Romanized Arabic — Arabizi — is the everyday written form of Arabic for tens of millions of users, and it has no standard orthography: the same word surfaces in many spellings, with digits standing in for Arabic phonemes. We know this breaks models — the same prompts that safety-tuned LLMs refuse in Arabic script succeed as jailbreaks in Arabizi (EMNLP 2024) — but the field has no instrument to measure it and no evidence on which fix works. I'm building the first orthographic-robustness benchmark for Egyptian Arabizi: the same content rendered in multiple attested spelling variants, with prediction-consistency metrics, plus a matched-condition comparison of the four standard input strategies — raw subword, normalization, transliteration to Arabic script, and byte-level modeling. The variant generator is calibrated on spelling variation mined from real data and validated against held-out human-written variants."

**Contribution bullets:**
1. **Data/evaluation:** the first spelling-variant consistency benchmark for Egyptian Arabizi (an unoccupied niche — verified against 2026 literature), with a datasheet, agreement statistics, and a real-vs-synthetic variant split.
2. **Empirical:** the first controlled comparison of normalization vs. transliteration vs. byte-level vs. raw-subword pipelines on Arabizi under matched fine-tuning budgets, including the cheapest intervention of all (training-time variant augmentation) as an ablation.
3. **Analysis:** a quantification of tokenizer segmentation instability under orthographic variation and its correlation with prediction flips — extending tokenizer-fairness results (Petrov et al. 2023) to intra-language variation.

**Positioning one-liners for related work:** NileChat (2025) evaluates Egyptian in both scripts but in separate tables with no consistency metric; MulTypo (2026) does matched typo-consistency but on Arabic script, not Arabizi; Riabi et al. (2021) showed char-models help on Algerian Arabizi but compared no normalization or transliteration arms; Al Ghanim et al. (2024) proved cross-orthography inconsistency exists but only in a safety framing. This project is the measurement layer those four lines are missing.

**Why this framing lands with multilingual-LM people:** it's not "Franco-Arabic is cool" — it's *orthographically unstandardized romanized text is a blind spot of current multilingual evaluation*, Egyptian Arabizi is the highest-density instance (huge population, extreme variation, zero benchmarks), and the benchmark schema generalizes to romanized Hindi/Urdu, Greeklish, and undiacritized Vietnamese. The safety angle (jailbreak-via-Arabizi) gives it immediate relevance beyond dialect NLP.

**The ask (have one ready):** feedback on the consistency-metric design (C@k vs. pairwise agreement under label noise), whether the matched-budget protocol is convincing, and whether a multi-dialect extension (NArabizi/TArC plug in directly) or the contrastive variant-invariant embedding follow-up is the stronger internship-scale continuation.

---

## Key references (verified)

- Shazal, Usman & Habash 2020, *A Unified Model for Arabizi Detection and Transliteration* — aclanthology.org/2020.wanlp-1.15
- Al-Badrashiny et al. 2014, *Automatic Transliteration of Romanized Dialectal Arabic* — aclanthology.org/W14-1604
- Darwish 2014, *Arabizi Detection and Conversion to Arabic* — aclanthology.org/W14-3629
- Ryskina et al. 2020, *Phonetic and Visual Priors for Decipherment of Informal Romanization* — aclanthology.org/2020.acl-main.737
- Riabi, Sagot & Seddah 2021, char-models on NArabizi — aclanthology.org/2021.wnut-1.47
- Al Ghanim et al. 2024, *Jailbreaking LLMs with Arabic Transliteration and Arabizi* — arxiv.org/abs/2406.18725
- Zhao et al. 2026, *MulTypo* — arxiv.org/abs/2510.09536
- Petrov et al. 2023, *Tokenizers Introduce Unfairness Between Languages* — arxiv.org/abs/2305.15425
- Pruthi et al. 2019, *Combating Adversarial Misspellings* — aclanthology.org/P19-1561
- NileChat (UBC-NLP) — arxiv.org/abs/2505.18383; Nile-Chat (MBZUAI-Paris) — arxiv.org/abs/2507.04569
- Roman Urdu POS under lexical variation, LRE 2025 — link.springer.com/article/10.1007/s10579-025-09865-w
- CODA* guidelines — camel-guidelines.readthedocs.io/en/latest/orthography/
- Datasets: LDC2017T07 (paid gold); UBC-NLP/nilechat-arabizi-egy; faisalq/ETC; ASTD (github.com/mahmoudnabil/ASTD); NArabizi (parsiti.github.io/NArabizi); TArC (github.com/eligugliotta/tarc)
