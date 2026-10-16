"""Blind pairwise arena benchmark for the Dataset B fusion pilot: FusioN's
fused output vs. a Best-of-N baseline, and vs. one or more external
frontier models' single-shot attempts -- the two comparisons the FusioN
paper itself runs (arXiv:2510.00931), applied to our anchors.

Reads one arm's --audit JSONL (build_dataset_b.py writes "fused",
"bon_pick", "anchor", "arabic", "scenario" per row). For each --reference,
generates one single-shot completion per anchor (same scenario as the
arm's own attempt, cached so a re-run doesn't re-spend budget), gated
through the same fusion.check_sentence() as everything else -- an
ungated reference isn't a fair comparison.

References are specified as "label=prefix:model", same dispatch prefixes
build_dataset_b.py uses ("deepseek:", "zai:", "openrouter:") -- e.g.
"deepseek_pro=deepseek:deepseek-v4-pro" or "glm46=zai:glm-4.6". Running
multiple references produces one "vs_<label>" comparison leg per anchor
per arm per reference -- e.g. testing fusion against both a frontier
reference (deepseek-v4-pro) and a genuinely weaker one (glm-4.6) in the
same pass, added 2026-08-25 after finding deepseek-v4-pro alone gave a
near-zero win rate for every arm and wasn't distinguishing them.

Every comparison is written blind: two candidates labeled "Candidate 1" /
"Candidate 2" with randomized position (controls for position bias), the
real side identity kept in a private key file never shown to judges.

Pools every arm passed in ONE combined chunk set (same convention as
prep_council_fusion.py) -- comparisons across arms never conflict (each
has its own comparison_id), and pooling means one judging pass covers
every arm instead of each --arm invocation silently clobbering the last
one's chunk/key files (a real bug caught running this the first time).

The reference cache is keyed on (anchor, scenario, model) where "model"
is the full "prefix:model" spec, not just the bare model name -- keeping
different reference types (and different DeepSeek tiers) from ever
colliding. An earlier version keyed on (anchor, scenario) only, missing
"model" entirely; caught before a --deepseek-model switch could silently
replay a different tier's cached answers under the wrong label.

A single dropped connection (observed live: a tethered/hotspot network
losing DNS resolution partway through a multi-minute run, twice in a
row) no longer kills the whole run -- that one anchor's reference is
skipped and counted as gated out, since the underlying client already
retries a transient failure before giving up. Already-cached anchors
from a prior partial run are untouched on retry.

    .venv/bin/python experiments/d5_cohere_track/training_data/run_arena_judge.py \
        --audit data/generated/fusion_pilot/arm_qwen_audit.jsonl --arm qwen3.8:27b-mlx \
        --audit data/generated/fusion_pilot/arm_commandr_audit.jsonl --arm command-r:35b \
        --reference "deepseek_pro=deepseek:deepseek-v4-pro" \
        --reference "glm46=zai:glm-4.6"
"""

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval import deepseek_client, openrouter_client, zai_client
from franco_eval.fusion import Anchor, check_sentence, teacher_prompt
from franco_eval.model_clients import JsonlCache

REFERENCE_CACHE = ROOT / "data" / "generated" / "fusion_pilot" / "reference_cache.jsonl"
CHUNK_DIR = pathlib.Path("/tmp/arena_judge_chunks")
KEY_OUT = ROOT / "data" / "generated" / "fusion_pilot" / "arena_key.json"
POSITION_SEED = 41


def clean_reply(raw: str) -> str:
    for line in raw.strip().splitlines():
        line = line.strip().strip('"“”').strip()
        if len(line.split()) >= 4:
            return line
    return raw.strip().strip('"').strip()


def parse_reference(spec: str) -> tuple[str, str]:
    """'label=prefix:model' -> (label, 'prefix:model')."""
    label, _, model_spec = spec.partition("=")
    if not label or not model_spec:
        raise SystemExit(f"--reference must be 'label=prefix:model', got {spec!r}")
    return label, model_spec


def dispatch(model_spec: str, prompt: str, *, timeout: int) -> str:
    """Same prefix convention as build_dataset_b.py's ask()."""
    if model_spec.startswith("zai:"):
        return zai_client.chat_with_retries(
            model_spec.removeprefix("zai:"), prompt, timeout=timeout)
    if model_spec.startswith("deepseek:"):
        return deepseek_client.chat_with_retries(
            model_spec.removeprefix("deepseek:"), prompt, timeout=timeout)
    if model_spec.startswith("openrouter:"):
        return openrouter_client.chat_with_retries(
            model_spec.removeprefix("openrouter:"), prompt, timeout=timeout)
    raise SystemExit(f"--reference model must be prefixed zai:/deepseek:/openrouter:, "
                     f"got {model_spec!r}")


def reference_completion(anchor: Anchor, scenario: str, model_spec: str,
                         cache: JsonlCache, timeout: int) -> str | None:
    """One single-shot attempt, no fusion, gated the same as everything
    else -- an ungated reference wouldn't be a fair comparison."""
    record = {"anchor": anchor.word, "scenario": scenario, "model": model_spec}
    cached = cache.get(record)
    if cached is not None:
        text = cached["text"]
    else:
        raw = dispatch(model_spec, teacher_prompt(anchor, scenario), timeout=timeout)
        text = clean_reply(raw)
        # JsonlCache.put() already skips caching a falsy "raw" -- empty
        # completions are failures, not answers, so they retry next run.
        # An earlier version of this line forced caching anyway (`raw or
        # " "`), which meant a single bad call (e.g. a token-budget miss)
        # got cached as a permanent "confirmed empty," blocking all retries.
        cache.put({**record, "text": text, "raw": raw})
    return text if text and check_sentence(text, anchor).ok else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", type=pathlib.Path, action="append",
                        required=True, dest="audits")
    parser.add_argument("--arm", action="append", required=True, dest="arms",
                        help="fusor model name, one per --audit, same order")
    parser.add_argument("--reference", action="append", required=True,
                        dest="references", help="'label=prefix:model', repeatable")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    if len(args.audits) != len(args.arms):
        raise SystemExit(f"{len(args.audits)} --audit vs {len(args.arms)} --arm "
                         "-- pass one --arm per --audit, same order")

    references = [parse_reference(spec) for spec in args.references]

    cache = JsonlCache(REFERENCE_CACHE, ("anchor", "scenario", "model"))
    rng = random.Random(POSITION_SEED)
    key: dict[str, dict] = {}
    judgeable: list[dict] = []
    outcomes = {"vs_bon": 0, "no_bon": 0}
    for label, _ in references:
        outcomes[f"vs_{label}"] = 0
        outcomes[f"{label}_gated_out"] = 0

    for audit_path, arm in zip(args.audits, args.arms):
        with audit_path.open(encoding="utf-8") as handle:
            for row in handle:
                row = json.loads(row)
                if not row["gates_ok"]:
                    continue
                anchor = Anchor(row["anchor"], row["arabic"], "", row["context_type"])
                fused = row["fused"]

                if row.get("bon_pick"):
                    comparison_id = f"{arm}:{anchor.word}:vs_bon"
                    candidates = [fused, row["bon_pick"]]
                    fused_slot = 0
                    order = [0, 1]
                    rng.shuffle(order)
                    labeled = {f"Candidate {i + 1}": candidates[side]
                              for i, side in enumerate(order)}
                    key[comparison_id] = {
                        "arm": arm, "baseline": "bon",
                        "fused_is": f"Candidate {order.index(fused_slot) + 1}",
                    }
                    judgeable.append({"comparison_id": comparison_id,
                                      "anchor": anchor.word, "arabic": anchor.arabic,
                                      **labeled})
                    outcomes["vs_bon"] += 1
                else:
                    outcomes["no_bon"] += 1

                for label, model_spec in references:
                    try:
                        ref_text = reference_completion(anchor, row["scenario"],
                                                         model_spec, cache, args.timeout)
                    except (OSError, TimeoutError) as exc:
                        # A dropped connection mid-run (observed live: a
                        # tethered/hotspot connection losing DNS resolution
                        # partway through a multi-minute run) shouldn't
                        # kill every anchor after it -- the client's own
                        # chat_with_retries() already retried this specific
                        # call before giving up. Skip just this one anchor;
                        # already-cached anchors are unaffected on retry.
                        print(f"  {arm}:{anchor.word}:{label} network error, skipping: {exc}")
                        outcomes[f"{label}_gated_out"] += 1
                        continue
                    if ref_text is None:
                        outcomes[f"{label}_gated_out"] += 1
                        continue
                    comparison_id = f"{arm}:{anchor.word}:vs_{label}"
                    candidates = [fused, ref_text]
                    order = [0, 1]
                    rng.shuffle(order)
                    labeled = {f"Candidate {i + 1}": candidates[side]
                              for i, side in enumerate(order)}
                    key[comparison_id] = {
                        "arm": arm, "baseline": model_spec, "reference_label": label,
                        "fused_is": f"Candidate {order.index(0) + 1}",
                    }
                    judgeable.append({"comparison_id": comparison_id,
                                      "anchor": anchor.word, "arabic": anchor.arabic,
                                      **labeled})
                    outcomes[f"vs_{label}"] += 1

    KEY_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEY_OUT.write_text(json.dumps(key, indent=1, ensure_ascii=False),
                       encoding="utf-8")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    size = args.chunk_size
    for i in range(0, len(judgeable), size):
        (CHUNK_DIR / f"c{i // size}.json").write_text(
            json.dumps(judgeable[i:i + size], ensure_ascii=False),
            encoding="utf-8")

    print(f"{len(judgeable)} comparisons -> {CHUNK_DIR}  outcomes: {outcomes}")
    print(f"private key (NOT for judges) -> {KEY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
