"""Diagnostic probe: does Best-of-N (a single raw teacher draft, no fusion)
already lose to DeepSeek at close to fusion's own loss rate?

Motivated by a direct question during analysis of run_arena_judge.py's
results: fusion loses to a single DeepSeek call for the local fusors, and
even the frontier fusor only ties Flash / loses to Pro. Is that a fusion
weakness, or is it simply that the 5-teacher pool (aya-expanse:8b,
gemma4:e2b, llama3.1:8b, jwnder/jais-adaptive:7b, nemotron-3-nano:30b --
all 8B-30B open models) is outclassed by DeepSeek (284B-1.6T total params)
regardless of what a fusor does with their drafts? A fusor can only
recombine/polish what's in its input drafts -- it cannot inject knowledge
or fluency the teachers never produced. If bon_pick (the single BEST raw
teacher draft, untouched by any fusor) already loses to DeepSeek at a
similar rate, that points to the teacher pool as the real ceiling, not a
fusion-specific shortfall.

Needs ZERO new model calls: bon_pick is already in each arm's --audit
JSONL (build_dataset_b.py's BoN computation), and DeepSeek references are
already cached in deepseek_ref_cache.jsonl from run_arena_judge.py's own
runs. This script only reads existing files and produces blind judge
chunks, same convention as run_arena_judge.py / prep_council_fusion.py.

    .venv/bin/python experiments/d5_cohere_track/training_data/probe_teacher_ceiling.py \
        --audit data/generated/fusion_pilot/v2_arm_qwen_audit.jsonl --arm qwen3.8:27b-mlx \
        --audit data/generated/fusion_pilot/v2_arm_commandr_audit.jsonl --arm command-r:35b \
        --audit data/generated/fusion_pilot/v2_arm_zai_audit.jsonl --arm zai:glm-5.3 \
        --deepseek-model deepseek-v4-pro
"""

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

DEEPSEEK_CACHE = ROOT / "data" / "generated" / "fusion_pilot" / "deepseek_ref_cache.jsonl"
CHUNK_DIR = pathlib.Path("/tmp/teacher_ceiling_chunks")
KEY_OUT = ROOT / "data" / "generated" / "fusion_pilot" / "teacher_ceiling_key.json"
POSITION_SEED = 53


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", type=pathlib.Path, action="append",
                        required=True, dest="audits")
    parser.add_argument("--arm", action="append", required=True, dest="arms")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    if len(args.audits) != len(args.arms):
        raise SystemExit(f"{len(args.audits)} --audit vs {len(args.arms)} --arm")

    ds_cache: dict[tuple[str, str], str] = {}
    if DEEPSEEK_CACHE.exists():
        for line in DEEPSEEK_CACHE.open(encoding="utf-8"):
            row = json.loads(line)
            if row["model"] == args.deepseek_model:
                ds_cache[(row["anchor"], row["scenario"])] = row["text"]

    rng = random.Random(POSITION_SEED)
    key: dict[str, dict] = {}
    judgeable: list[dict] = []

    for audit_path, arm in zip(args.audits, args.arms):
        with audit_path.open(encoding="utf-8") as handle:
            for row in handle:
                row = json.loads(row)
                if not row["gates_ok"] or not row.get("bon_pick"):
                    continue
                ds_text = ds_cache.get((row["anchor"], row["scenario"]))
                if ds_text is None:
                    continue
                comparison_id = f"{arm}:{row['anchor']}:bon_vs_deepseek"
                candidates = [row["bon_pick"], ds_text]
                order = [0, 1]
                rng.shuffle(order)
                labeled = {f"Candidate {i + 1}": candidates[side]
                          for i, side in enumerate(order)}
                key[comparison_id] = {"arm": arm, "bon_is": f"Candidate {order.index(0) + 1}"}
                judgeable.append({"comparison_id": comparison_id,
                                  "anchor": row["anchor"], "arabic": row["arabic"],
                                  **labeled})

    KEY_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEY_OUT.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    size = args.chunk_size
    for i in range(0, len(judgeable), size):
        (CHUNK_DIR / f"c{i // size}.json").write_text(
            json.dumps(judgeable[i:i + size], ensure_ascii=False), encoding="utf-8")

    print(f"{len(judgeable)} comparisons -> {CHUNK_DIR}")
    print(f"private key (NOT for judges) -> {KEY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
