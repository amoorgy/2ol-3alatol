# Multi-agent architecture

**Motto: great change only needs a few steps.** The pipeline is five steps;
each step is owned by one specialized agent (`.claude/agents/`), all running
Opus at x-high reasoning effort. Every agent has a narrow scope, hard rules,
and one kind of artifact it produces — so any run of the pipeline is a handful
of clean handoffs, not one giant blob of work.

## The pipeline and its agents

```
        ┌─────────────┐   variant_clusters.tsv   ┌──────────────────┐
data ──▶│ franco-miner │─────────────────────────▶│ franco-perturber │
        └─────────────┘   (attested evidence)     └──────────────────┘
                                                     │ calibrated rules
                                                     ▼
        ┌────────────────────────┐  benchmark JSONL ┌──────────────────┐
        │ franco-benchmark-builder│◀────────────────│  taxonomy/perturb │
        └────────────────────────┘                  └──────────────────┘
                     │ data/benchmark/*
                     ▼
        ┌──────────────────┐   results/*.{json,md}   ┌───────────────┐
        │ franco-evaluator │────────────────────────▶│ franco-scribe │
        └──────────────────┘                         └───────────────┘
                                                        docs, reports
```

| Agent | Owns | Produces | Never does |
|---|---|---|---|
| `franco-miner` | corpora, `mine.py`, `skeleton.py` | variant clusters, mining reports | let synthetic text into benchmark corpora |
| `franco-perturber` | `taxonomy.py`, `perturb.py`, `normalize.py` | calibrated rules, canon decisions | change weights without mined evidence + native-speaker sign-off |
| `franco-benchmark-builder` | `data/benchmark/` | benchmark JSONL, splits, datasheet | blur the real-vs-synthetic variant split |
| `franco-evaluator` | `experiments/`, `metrics.py`, `results/` | reproducible result tables | report a number without a seed + command |
| `franco-scribe` | `README`, `docs/`, `changes.md` | docs in sync with reality | write a number that isn't in `results/` |

## How to use them

In Claude Code, just describe the task — the agent descriptions route it — or
name one explicitly:

```
> use franco-miner to re-mine variant clusters at a lower edit-distance threshold
> use franco-perturber to recalibrate taxonomy weights against the new clusters
> use franco-evaluator to add a fragment-width ablation to the showcase
```

A full benchmark refresh is exactly four handoffs:
miner → perturber → benchmark-builder → evaluator, then scribe updates docs.

## Design rules

1. **One agent, one artifact type.** If a task produces both rules and results,
   it's two tasks.
2. **Evidence flows forward only.** The perturber consumes the miner's clusters;
   the evaluator consumes the builder's benchmark. No agent edits an upstream
   artifact.
3. **The human owns the taxonomy weights.** Agents propose with evidence; the
   native speaker approves (see the marked block in `src/franco/taxonomy.py`).
4. **Everything reproducible.** Fixed seeds, committed scripts, tests green
   before any agent finishes.
