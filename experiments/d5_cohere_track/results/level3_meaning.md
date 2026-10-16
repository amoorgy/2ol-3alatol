# Level 3 — in-context word-meaning consistency

**Provenance: machine-derived probe stimulus** (Command R+ Arabizi). Headlines must also be shown on the human-authored arm.

Variants of one word SHOULD agree; the trap SHOULD disagree. The two failure rates are reported side by side and never combined.

| model | floor | agreement | under-robust | over-norm | ceiling | non-answer |
|---|---|---|---|---|---|---|
| qwen3.5:9b | PASS | 0.515 | 0.485 | 0.138 | 1.000 | 0.000 |
| gemma4:e2b | PASS | 0.436 | 0.564 | 0.105 | 1.000 | 0.040 |
| aya-expanse:8b | PASS | 0.265 | 0.735 | 0.074 | 1.000 | 0.002 |
| llama3.1:8b | FAIL | 0.236 | 0.764 | 0.137 | 0.962 | 0.152 |

How each column is computed, denominators, and the None-handling rules:
[meaning_metrics_reference.md](meaning_metrics_reference.md).
