# Background: every concept in this project, in plain language

Read this before (and again after) the learning path. Each section says what a
thing is, why this project cares, and where it lives in the repo.

## 1. The problem in one paragraph

Franco (Arabizi) is Arabic typed in Latin letters with digits standing in for
Arabic sounds (3=ع, 7=ح, 2=ء...). It has **no standard spelling**: the same word
appears as "a7a"/"aha", "ezayak"/"ezzayyak"/"izayak", "khalas"/"5alas". A model
that understands one spelling may fail on another — same meaning, different
behavior. It's proven this matters: safety-tuned LLMs that refuse harmful
prompts in Arabic script comply when the *same prompt* is written in Arabizi
(Al Ghanim et al., EMNLP 2024). Nobody has an instrument that measures this
brittleness for Egyptian Arabizi. This repo builds that instrument.

## 2. Tokenization — why "a7a" and "aha" are different worlds to a model

LLMs don't read characters; they read **tokens** — chunks from a fixed
vocabulary learned by Byte-Pair Encoding (BPE), which greedily merges frequent
character pairs. "aha" is frequent in web text → likely one token. "a7a" has a
digit in the middle — a pattern BPE rarely merged → it shatters into 2–3 rare
tokens. So two spellings of the same word enter the model as *completely
different inputs*. The number of tokens per word is called **fertility**;
unstable segmentation across variants is the mechanism we suspect drives
inconsistent predictions. (Karpathy's tokenizer video, Track 1 #4, is the
deep-dive; Direction 4 measures exactly this.)

## 3. Embeddings — meaning as geometry

Models represent text as vectors; similar meaning should mean nearby vectors.
Under spelling variation this breaks: "a7a" and "aha" can land far apart
because their tokens differ. Retrieval (finding the right document for a
query) is the cleanest way to observe this — which is why our showcase task is
retrieval. `src/franco/retrieval.py` implements the simplest embedding-free
version: documents as bags of character 3-grams, cosine similarity for
scoring. It's deliberately dumb — a **baseline** every fancier system must beat.

## 4. The three intervention families

When input spelling is chaotic you can:

1. **Normalize** — map every variant to one canonical Romanized form before the
   model sees it. Nobody ever defined a canonical Franco, so
   `src/franco/normalize.py` defines one ("Franco canon v0"): prefer digits for
   unambiguous phonemes (kh→5, gh→8, q→2), unify vowels (i→e, u→o), collapse
   doubled letters. It's *lossy on purpose* — consistency, not faithfulness.
   Its documented blind spot: a plain "h" could be ه or ح, and you can't know
   without a dictionary.
2. **Transliterate** — convert Arabizi to Arabic script and use strong
   Arabic-script models. Mature research line (best system: ~80.6% word
   accuracy — errors that the downstream model inherits). Direction 2.
3. **Model robustly** — skip preprocessing; use byte-level models or train on
   spelling-augmented data so the model itself absorbs variation. Directions 1
   (as bake-off arms) and 3.

The point of this project is that **nobody has measured which family wins for
Arabizi under fair conditions** — so we don't argue, we measure.

## 5. The skeleton key — our mining trick

`src/franco/skeleton.py` maps any spelling to a phonemic fingerprint: replace
digraphs (kh→x), map digits to phoneme classes (7 and h both → h), drop
vowels, collapse repeats. "khalas" and "5alas" both become `xls`. Group a
corpus's words by fingerprint and words spelled differently but pronounced the
same **collide** — giving us *real, attested* variant pairs (we found
`mesh(129)/msh(50)` and `7aga(127)/haga(48)` in our own seed data). Those real
pairs are the evidence that calibrates the perturbation rules — so our
synthetic variants imitate what Egyptians actually do, not what we imagine.

## 6. The perturbation engine — controlled noise

`src/franco/taxonomy.py` lists ~24 attested spelling habits (digit↔letter
swaps, vowel swaps, degemination...) each with a plausibility weight.
`src/franco/perturb.py` samples them to rewrite text at a chosen **intensity**
(probability each word gets rewritten) with 1–3 stacked edits per word —
because real variation is compound ("ezzayyak"→"izayak" differs by three
habits at once). Everything is seeded → identical benchmark on every machine.

## 7. Invariance testing and the consistency metric

Borrowed from CheckList (Track 2 #3): perturb the input in a
*meaning-preserving* way, then assert the model's output does not change.
Our metrics (`src/franco/metrics.py`):

- **recall@1** — did the system retrieve the right document?
- **C@k (consistency)** — given k spelling variants of the same query, did the
  system give the *same answer* for all k? Note: a model can be consistently
  wrong — consistency and accuracy are deliberately separate dials.
- **degradation curve** — plot both metrics as intensity rises; a robust
  system's curve is flat.

## 8. What the showcase actually shows

`experiments/d1_benchmark/run_showcase.py`, seconds on a laptop, no
dependencies: 489 human-authored seed messages as corpus; queries are 4-word
fragments (like a real search); two systems — raw char-trigram retrieval vs
the same retrieval over the Franco canon; averaged over 5 seeds. It evaluates
on TWO kinds of noise, and the difference between them is itself a finding:

- **Synthetic** (taxonomy-generated, intensities 0.2–0.8): normalization wins
  every cell — but a reported *leakage diagnostic* (40%→17% of variants
  normalize to the exact normalized source) shows part of that win is
  circular, because the normalizer inverts part of the rule set that made
  the noise. This trap was caught by adversarial review of this very repo —
  remember it: it is the standard failure mode of synthetic robustness evals.
- **Attested** (queries rewritten with real alternate spellings mined from
  the corpus): normalization still wins, 0.871 vs 0.850 recall@1 — a smaller,
  honest gain with no circularity.

That contrast — synthetic noise flatters interventions, real variation is the
test that counts — is the methodological core of the whole benchmark design.

## 9. Statistics you'll need to defend the numbers

- **Cohen's κ** — agreement between two human annotators beyond chance. Any
  label in the benchmark needs a reported κ, or reviewers dismiss it.
- **Bootstrap confidence intervals** — resample the test set with replacement
  many times to see how much a metric wobbles; the standard way to claim
  "system A beats system B" is a *paired* bootstrap. (Track 2 #8–9.)

## 10. Fine-tuning and LoRA — for the later arms

Full fine-tuning rewrites all weights; **LoRA** trains tiny low-rank update
matrices instead (cheap enough for a free Colab T4; QLoRA = same on a 4-bit
quantized base). Wave-2 arms of Direction 1 fine-tune small models (XLM-R,
ByT5) and LoRA-tune LLMs on identical data, differing only in input handling —
"matched conditions" is what makes the comparison publishable.

## 11. Glossary

| Term | Meaning here |
|---|---|
| Arabizi / Franco | Arabic in Latin letters + digit phonemes; Egyptian flavor |
| Orthographic variation | Many valid spellings of one word |
| Fertility | Tokens per word after tokenization |
| Canon | The single normalized spelling this repo defines |
| Attested | Actually observed in real data (vs synthetic) |
| Intensity | Probability each word of a text gets perturbed |
| C@k | Fraction of items where all k variants get the same prediction |
| Meaning (probe answer) | The one Arabic word a model gives for a Franco word in its carrier sentence |
| Matched conditions | Same data, budget, and eval for every compared system |
| CODA | Conventional orthography for dialectal Arabic — Arabic-script side only |
