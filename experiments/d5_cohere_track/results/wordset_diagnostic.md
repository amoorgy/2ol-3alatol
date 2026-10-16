# NileChat word-set diagnostic

**Provenance: machine-derived.** `nilechat-arabizi-egy` is Egyptian Arabic
transliterated into Arabizi by Command R+ (CC-BY-NC-4.0). Under hard rule 1
it is a measurement corpus and probe stimulus only, never a benchmark
artifact, and every headline computed from it must also be shown on the
human-authored arm in `data/benchmark/pairs.csv`.

Shards: [0, 1, 2] | documents scanned: 571,524 | word types kept: 2,106,304

## Corpus hygiene

| property | value |
|---|---|
| documents with Arabic script, a prime digraph, or a URL | 483,735 (84.6%) |
| frequency-table prune events | 5 |

A word can lose at most one count per prune event, so the worst-case
undercount is 5 against a `--min-count` of 200.

## Spelling variance — is there any?

The corpus was written by one model applying one convention, so the
premise that humans spell a word five ways may not survive contact with
it. It partly does: Command R+ was inconsistent *across documents*.

| tier | skeletons | with >1 spelling | share |
|---|---|---|---|
| 1 — raw `skeleton()` | 967 | 393 | 40.6% |
| 2 — confirmed | 967 | 315 | 32.6% |

**Tier 2 is the reportable number.** Tier 1 overstates it because
`skeleton()` deletes vowels, so it merges words that merely share a
consonant frame. Tier 2 keeps only forms within edit distance 2 of the
cluster head that `classify_edge` can also explain orthographically.

Neither tier licenses mining the SAME arm from this corpus: orthographic
explainability is not word identity. `men`~`meen` (من / مين) reads as a
clean VOWEL_LENGTH edit and is two different words. Variants are
therefore *generated* from a known base word, which makes same-word
identity true by construction.

## Selection

36,939 of 2,106,304 types passed every gate; 500 were selected under stratum quotas.

| gate | rejected |
|---|---|
| length | 1,296 |
| numeric | 12,805 |
| excluded_digit | 52,531 |
| foreign_letters | 131,191 |
| english_stop | 144 |
| min_count | 1,849,157 |
| min_doc_freq | 1,134 |
| proper_noun | 2,874 |
| no_feature | 18,233 |

| stratum | quota | filled |
|---|---|---|
| digit | 170 | 170 |
| diph | 80 | 80 |
| vlong | 70 | 70 |
| digraph | 70 | 70 |
| gem | 60 | 60 |
| finalyi | 50 | 50 |

Quotas exist because the top 500 by raw frequency yield too few
gemination and diphthong words to report per-phenomenon — the same
"too thin to defend" flaw the d5 README already lists as open.

26 of the selected words also appear in the
human-authored arm; that overlap is recorded, not excluded, because
cross-arm comparison needs it.
