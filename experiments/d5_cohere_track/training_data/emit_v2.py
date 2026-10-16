"""v2 master records -> the fine-tuning files, one row type per emitter.

Reads   data/generated/v2/master.jsonl
        data/stimulus/fusion_v2_votes.tsv   (council verdicts on fused sentences)
        data/stimulus/pairs_v2_votes.tsv    (council verdicts on flagged pairs)
Writes  data/generated/v2/final/{normalize,meaning,mixed}_{train,val}.jsonl
        data/training/v2/stats.json         (tracked: counts, hashes, seeds)
        data/training/v2/sample.jsonl       (tracked: 200 rows to eyeball)

The row files live under data/generated (gitignored): at ~300k rows they are
too large to track, and hard rule 2 is kept by tracking everything they
regenerate from instead -- word_split.tsv, the committed verdict TSVs, the
fused sentences (which are NOT regenerable, being cloud-model output) and
these scripts -- plus the sha256 of every emitted file in stats.json.

Fused sentences enter only with a full council panel that accepted them;
unjudged ones are counted and held back, never silently admitted -- the same
contract as a_v1's emit(). NileChat rows are gated by the substitution
policy, whose estimated residual error is results/substitution_policy_audit.md,
and a row carrying a pair the policy flags for a judge (needs_judge) enters
only once the pair council accepted that pair.

    .venv/bin/python experiments/d5_cohere_track/training_data/emit_v2.py
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.emitters import meaning_row, normalize_row, pair_status
from franco_eval.substitution_policy import needs_judge, variant_from_rule_path

MASTER = ROOT / "data" / "generated" / "v2" / "master.jsonl"
FUSION_VOTES = ROOT / "data" / "stimulus" / "fusion_v2_votes.tsv"
PAIR_VOTES = ROOT / "data" / "stimulus" / "pairs_v2_votes.tsv"
FINAL = ROOT / "data" / "generated" / "v2" / "final"
TRACKED = ROOT / "data" / "training" / "v2"
TASKS = (("normalize", normalize_row), ("meaning", meaning_row))
MIX_SEED = 42
SAMPLE_SIZE = 200


def load_votes(path: pathlib.Path) -> dict[str, bool]:
    """id -> accepted, for full panels (n_judges >= 4) only."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["sentence_id"]: row["valid"] == "true"
                for row in csv.DictReader(handle, delimiter="\t")
                if int(row.get("n_judges", 0)) >= 4}


def make_flagged():
    """sub -> needs a judge?  Cached per (word, rendering, rule_path)."""
    cache: dict[tuple[str, str, str], bool] = {}

    def is_flagged(sub: dict) -> bool:
        key = (sub["word"], sub["rendering"], sub["rule_path"])
        if key not in cache:
            cache[key] = bool(needs_judge(sub["word"], variant_from_rule_path(
                sub["rendering"], sub["rule_path"])))
        return cache[key]

    return is_flagged


def gate(record: dict, fusion_votes: dict[str, bool], pair_votes: dict[str, bool],
         is_flagged, stats: collections.Counter) -> bool:
    if record["source"] == "fusion":
        verdict = fusion_votes.get(record["sentence_id"])
        if verdict is None:
            stats["fusion_rows_unjudged_held_back"] += 1
            return False
        if not verdict:
            stats["fusion_rows_council_rejected"] += 1
            return False
    status = pair_status(record, pair_votes, is_flagged)
    if status != "ok":
        stats[f"{status}_held_back"] += 1
        return False
    return True


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_all_votes(paths: list[pathlib.Path]) -> dict[str, bool]:
    """Union of several rounds' fused-sentence verdicts; a sentence given two
    different verdicts is an error, never silently resolved."""
    merged: dict[str, bool] = {}
    for path in paths:
        for sid, accepted in load_votes(path).items():
            if merged.get(sid, accepted) != accepted:
                raise ValueError(f"{sid}: conflicting verdicts across {paths}")
            merged[sid] = accepted
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusion-votes", type=pathlib.Path, action="append",
                        default=None, help="fused-sentence verdict TSV, repeatable "
                        f"(default: {FUSION_VOTES.name}; expansion rounds add theirs)")
    args = parser.parse_args()
    votes_paths = args.fusion_votes or [FUSION_VOTES]
    fusion_votes = load_all_votes(votes_paths)
    pair_votes = load_votes(PAIR_VOTES)
    is_flagged = make_flagged()
    stats: collections.Counter = collections.Counter()
    FINAL.mkdir(parents=True, exist_ok=True)
    handles = {(task, split): (FINAL / f"{task}_{split}.jsonl").open("w", encoding="utf-8")
               for task, _fn in TASKS for split in ("train", "val")}
    try:
        with MASTER.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not gate(record, fusion_votes, pair_votes, is_flagged, stats):
                    continue
                for task, build in TASKS:
                    row = build(record)
                    if row is not None:
                        handles[(task, record["split"])].write(
                            json.dumps(row, ensure_ascii=False) + "\n")
                        stats[f"{task}_{record['split']}"] += 1
                        stats[f"{task}_from_{record['source']}"] += 1
    finally:
        for h in handles.values():
            h.close()

    # the mixed file a trainer can point at directly: every row type,
    # deterministically shuffled so no task arrives as one long block
    files = {}
    for split in ("train", "val"):
        lines = [line for task, _fn in TASKS
                 for line in (FINAL / f"{task}_{split}.jsonl").open(encoding="utf-8")]
        random.Random(f"{MIX_SEED}:{split}").shuffle(lines)
        mixed = FINAL / f"mixed_{split}.jsonl"
        mixed.write_text("".join(lines), encoding="utf-8")
        if split == "train":
            TRACKED.mkdir(parents=True, exist_ok=True)
            (TRACKED / "sample.jsonl").write_text("".join(lines[:SAMPLE_SIZE]),
                                                  encoding="utf-8")
    for path in sorted(FINAL.glob("*.jsonl")):
        files[path.name] = {"rows": sum(1 for _ in path.open(encoding="utf-8")),
                            "sha256": sha256(path)}

    master_stats = json.loads((MASTER.parent / "master_stats.json").read_text())
    report = {"rows": dict(sorted(stats.items())), "files": files,
              "master": master_stats, "master_sha256": sha256(MASTER),
              "fusion_votes": [str(p.relative_to(ROOT)) for p in votes_paths if p.exists()],
              "pair_votes_present": PAIR_VOTES.exists(),
              "mix_seed": MIX_SEED,
              "provenance": "nilechat_machine_derived + fusion_synthetic; "
                            "trains on the TRAIN side of word_split.tsv only -- "
                            "evaluate on its held-out side"}
    (TRACKED / "stats.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(report["rows"], indent=1))
    for name, info in files.items():
        print(f"  {name:28s} {info['rows']:8,d} rows")
    print(f"wrote {FINAL}/ and {TRACKED}/stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
