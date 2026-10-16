# Level 1 — can a tokenizer tell a real variant from a look-alike?

**Provenance.** The `human` arm is human-authored (`data/benchmark/pairs.csv`); the `nilechat` arm is machine-derived probe stimulus (Command R+ transliteration) and is never a benchmark artifact. Headlines must hold on the human arm.

`separability AUC` = probability a DIFFERENT pair fragments more than a SAME pair; 0.5 = the tokenizer offers the model no signal.

## Arm: human (human_authored, 67 SAME / 35 DIFFERENT)

| tokenizer | vocab | frag SAME | frag DIFFERENT | AUC | 95% CI |
|---|---|---|---|---|---|
| mt5-base | 250100 | 0.750 | 0.803 | **0.561** | [0.445, 0.670] |
| mistral-7b | 32768 | 0.698 | 0.756 | **0.576** | [0.454, 0.697] |
| byt5 | 256 | 0.219 | 0.278 | **0.589** | [0.467, 0.702] |
| nilechat-3b | 151643 | 0.696 | 0.771 | **0.606** | [0.492, 0.716] |
| qwen3.5-9b | 248044 | 0.701 | 0.776 | **0.609** | [0.501, 0.722] |
| llama-3.1-8b | 128000 | 0.697 | 0.778 | **0.613** | [0.501, 0.726] |
| aya-expanse-8b | 255000 | 0.702 | 0.793 | **0.619** | [0.510, 0.723] |
| gpt-oss-20b | 199998 | 0.711 | 0.804 | **0.624** | [0.510, 0.732] |
| gemma-4-e2b | 262144 | 0.687 | 0.786 | **0.634** | [0.520, 0.745] |

## Arm: nilechat (machine_derived, 1792 SAME / 314 DIFFERENT)

| tokenizer | vocab | frag SAME | frag DIFFERENT | AUC | 95% CI |
|---|---|---|---|---|---|
| byt5 | 256 | 0.207 | 0.270 | **0.586** | [0.554, 0.619] |
| gpt-oss-20b | 199998 | 0.680 | 0.760 | **0.609** | [0.573, 0.645] |
| mistral-7b | 32768 | 0.625 | 0.716 | **0.616** | [0.582, 0.650] |
| aya-expanse-8b | 255000 | 0.670 | 0.757 | **0.619** | [0.584, 0.654] |
| qwen3.5-9b | 248044 | 0.659 | 0.752 | **0.622** | [0.588, 0.657] |
| gemma-4-e2b | 262144 | 0.673 | 0.766 | **0.624** | [0.590, 0.658] |
| llama-3.1-8b | 128000 | 0.645 | 0.740 | **0.626** | [0.592, 0.660] |
| nilechat-3b | 151643 | 0.641 | 0.738 | **0.627** | [0.593, 0.662] |
| mt5-base | 250100 | 0.682 | 0.795 | **0.633** | [0.602, 0.664] |

## Models sharing a tokenizer

- `34c655bfe58a`: aya-expanse-32b, aya-expanse-8b, command-a, command-r7b-arabic

## Mirrors standing in for gated repos

| tokenizer | repo used | stands in for | pinned signature | verified |
|---|---|---|---|---|
| aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | CohereLabs/aya-expanse-8b | `34c655bfe58a` | yes |
| nilechat-3b | ranwakhaled/nilechat-3b-ideal-noquant | UBC-NLP/NileChat-3B | `ba9132633c65` | yes |
