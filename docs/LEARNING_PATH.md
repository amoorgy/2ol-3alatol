# Learning Path

Everything you need to understand this project, in watch/read order. Every URL
was verified live (YouTube oEmbed API / page fetch) in July 2026 by dedicated
research agents. Total: ~9h video + ~5h reading, spread over the first two weeks.

**Key negative finding:** no quality video content on Arabizi exists anywhere —
the CAMeL Lab channel and the 23-episode Arabic NLP Series were enumerated in
full to confirm. Track 3 is papers by necessity.

---

## Track 1 — Foundations: how models process text (~4h15m video + 1h15m read)

Watch in this order. Items 1–3 build the mental model; item 4 is the
load-bearing one for this project.

| # | Resource | Length | Level | Why it matters here |
|---|---|---|---|---|
| 1 | [Transformers, the tech behind LLMs — 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) | 27m | beginner | The end-to-end mental model: text → tokens → embeddings → logits — the exact pipeline the benchmark probes. |
| 2 | [Attention in transformers, step-by-step — 3Blue1Brown](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 26m | beginner | How a token's representation is rewritten by its neighbours — why one respelled character can cascade into a different prediction. |
| 3 | [Vectoring Words (Word Embeddings) — Computerphile](https://www.youtube.com/watch?v=gQddtTdmG_8) | 17m | beginner | Why similar text should land close in vector space — the assumption behind measuring representation drift between "a7a" and "aha". |
| 4 | [**Let's build the GPT Tokenizer — Andrej Karpathy**](https://www.youtube.com/watch?v=zduSFxRajkE) | 2h14m | intermediate | **The single most important resource for this project.** The closing section on tokenization quirks explains most failure modes the benchmark will surface: a digit inside a word forces a BPE merge boundary the vocabulary never learned, so "a7a" becomes a rare multi-token sequence while "aha" is one common token. Watch that section twice. |
| 5 | [Byte Latent Transformer: Patches Scale Better Than Tokens (Paper Explained)](https://www.youtube.com/watch?v=loaTGpqfctI) | 36m | intermediate | Why byte-level models exist and how they buy noise robustness — the argument for the byte-level arm of the bake-off. |
| 6 | [ByT5 paper](https://arxiv.org/abs/2105.13626) | 30m read | advanced | Its synthetic-noise evaluation protocol is a near-direct template for the variant-consistency experiment you are building. |
| 7 | [What is LoRA — explained by its inventor (Edward Hu)](https://www.youtube.com/watch?v=DhRoTONcyZE) | 7m | beginner | What a low-rank update actually changes — so you can reason about whether LoRA can fix orthographic brittleness or only paper over it. |
| 8 | [Unsloth Gemma fine-tuning guide](https://unsloth.ai/docs/models/gemma-4/train) | 45m read | intermediate | Model-specific reference for the base model this repo fine-tunes, with chat-template and known-bug details. |

Alternative for #7 if too terse: [LoRA explained — AI Coffee Break](https://www.youtube.com/watch?v=KEv-F5UkhxU) (8m).
Optional depth on token-free models: [CANINE paper](https://arxiv.org/abs/2103.06874).

## Track 2 — Research craft: how credible evaluation is done (~4h30m)

This track is what makes the project *research* instead of a demo.

| # | Resource | Length | Level | Why it matters here |
|---|---|---|---|---|
| 1 | [Douwe Kiela — Dynabench: Rethinking Benchmarking in AI](https://www.youtube.com/watch?v=3LP24xp5Bro) | 1h18m | intermediate | Why a benchmark can look rigorous and still measure nothing: annotator artifacts, spurious cues, saturation — the credibility bar yours must clear. |
| 2 | [Sebastian Ruder — Challenges and Opportunities in NLP Benchmarking](https://www.ruder.io/nlp-benchmarking/) | 16m read | intermediate | A citable checklist of what a defensible benchmark needs; use it as the design spec. |
| 3 | [**Marco Tulio Ribeiro — CheckList: Behavioral Testing of NLP Models**](https://www.youtube.com/watch?v=VqiTtdY58Ts) | 1h01m | intermediate | **The methodological backbone**: CheckList's Invariance test (perturb input meaning-preservingly, assert the prediction doesn't change) is literally what this benchmark is. Borrow its reporting format. |
| 4 | [Christopher Potts — Adversarial Testing (XCS224U)](https://www.youtube.com/watch?v=486mTOQnhgU) | 14m | beginner | The trap that bites hardest: distinguishing a fair meaning-preserving perturbation from a degenerate one, and reporting without overclaiming. |
| 5 | [Pruthi et al. 2019 — Combating Adversarial Misspellings](https://arxiv.org/abs/1905.11268) | 45m read | advanced | Closest prior art: one character edit drops BERT sentiment accuracy 90.3%→45.8%; defines a "sensitivity" metric reusable for variant inconsistency. (No talk recording exists — verified.) |
| 6 | [Christopher Potts — Datasets (XCS224U)](https://www.youtube.com/watch?v=zFtA0fjaXPE) | 22m | beginner | How a research-grade dataset is specified: annotation protocol, guideline iteration, data statements. |
| 7 | [Cohen's Kappa — inter-rater reliability](https://www.youtube.com/watch?v=z4CiQPV0Mgw) | 11m | beginner | Every human-validated claim ("these two spellings are the same word") needs an agreement number, not one person's opinion. |
| 8 | [Bootstrapping Main Ideas — StatQuest](https://www.youtube.com/watch?v=Xz0x-8-cgaQ) | 9m | beginner | The machinery behind the paired bootstrap test — how you show pipeline differences are real, not test-set noise. |
| 9 | [Christopher Potts — Model Evaluation (XCS224U)](https://www.youtube.com/watch?v=7zZRaoHr-8g) | 22m | intermediate | Baselines, seeds, and which significance test to use when comparing systems. |
| 10 | [Evaluation Measures for Search & Recommenders — James Briggs](https://www.youtube.com/watch?v=BD9TkvEsKwM) | 31m | beginner | Recall@K / MRR / NDCG with runnable Python — the vocabulary of the retrieval slice this repo's showcase already implements. |

Optional: [HELM overview — Yifan Mai](https://www.youtube.com/watch?v=fLoYcXLG6ro) (29m) — HELM's robustness metric is "invariance under typo perturbation", conceptually adjacent to this project.

## Track 3 — Arabic & Arabizi specifics (~2h video + ~4h read)

| # | Resource | Length | Level | Why it matters here |
|---|---|---|---|---|
| 1 | [Nizar Habash — A Short Introduction to Arabic NLP](https://www.youtube.com/watch?v=vQL5CUlbw0k) | 1h06m | beginner | Morphological richness, orthographic ambiguity, dialect variation — the three interacting problems this benchmark measures. Watch first. |
| 2 | [Arabic NLP Series Ep. 3: Understanding the Arabic Script](https://www.youtube.com/watch?v=pGK7I3FYUJA) | 16m | beginner | What the Arabic-script side of a transliteration pipeline contains (hamza variants, taa marbuta, Unicode traps). |
| 3 | [Arabic NLP Series Ep. 20: Arabic Tokenization](https://www.youtube.com/watch?v=QZObcrAO448) | 25m | intermediate | Why BPE fragments Arabic and how segmentation interacts with morphology. |
| 4 | [CODA paper — Habash, Diab & Rambow 2012](https://aclanthology.org/L12-1328/) | 35m read | intermediate | The theoretical charter for the normalization arm: dialectal Arabic has no standard spelling; here's how to impose one (Egyptian is the worked example). |
| 5 | [CODA normalization as seq2seq — ArabicNLP 2024 talk](https://www.youtube.com/watch?v=3sXYk_c5cv8) | 14m | intermediate | The closest existing system to the "normalize first" arm, including its evaluation. |
| 6 | [Shazal, Usman & Habash 2020 — Arabizi detection + transliteration](https://aclanthology.org/2020.wanlp-1.15/) | 40m read | intermediate | Reference implementation of the transliteration arm; its 80.6% word accuracy is the realistic error ceiling that arm inherits. |
| 7 | [Riabi, Sagot & Seddah 2021 — char models on NArabizi](https://aclanthology.org/2021.wnut-1.47/) | 40m read | intermediate | Direct precedent for the byte/char arm: a char LM on 99k noisy sentences matches large multilingual models. |
| 8 | [Petrov et al. 2023 — Tokenizers Introduce Unfairness](https://proceedings.neurips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html) | 45m read | intermediate | The measurement vocabulary + fairness framing for tokenizer effects. |
| 9 | [Al Ghanim et al. 2024 — Jailbreaking LLMs with Arabizi](https://aclanthology.org/2024.emnlp-main.1034/) | 40m read | advanced | The existence proof that models behave differently on the same content in Arabizi vs Arabic script — your strongest "why this matters". |
| 10 | [NileChat 2025](https://arxiv.org/abs/2505.18383) | 1h read | advanced | The most relevant model to evaluate: trained on Egyptian in both scripts including (synthetic) Arabizi. |

Optional Habash talks (all verified live): [A Thousand and One Sources of Bias in Arabic CL](https://www.youtube.com/watch?v=_mzHVeA2APA) (54m — best fairness framing for this project), [ELRA Zampolli Prize lecture 2024](https://www.youtube.com/watch?v=tA7Fv2Xf9gg) (42m), [CLSP JHU deep dive pt.1](https://www.youtube.com/watch?v=YH6KEkTSEBA) / [pt.2](https://www.youtube.com/watch?v=LotoF0UxCJ8).

---

## Suggested schedule

- **Days 1–2:** Track 1 items 1–4 (get to the Karpathy tokenizer video fast — it explains *why this project exists*).
- **Days 3–4:** Track 2 items 1–4 (benchmark credibility + CheckList), then read `docs/BACKGROUND.md` again — it will click differently.
- **Week 2, alongside coding:** Track 3 + the remaining statistics items, one per day.
