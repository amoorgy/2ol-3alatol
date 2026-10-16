"""Chunk the v2 substitution pairs that need a judge, for the 5-lens council.

Native ruling 2026-09-25: word-initial ع-deletion and a vowel dropped beside
another vowel read correctly in context, so the policy allows them -- but no
rule can prove a given pair (3nd -> and lands on an English word). Each
distinct (word, form) flagged by substitution_policy.needs_judge() is judged
ONCE, shown in up to two real sentences from the master file, and the
verdict applies to every row carrying that pair. Unjudged pairs are held
back by emit_v2.py, never admitted.

Judge rows come back as {"sid", "rid", "v", "why"} with sid == rid ==
"word->form", and merge with Dataset A's council code:

    merge_council.py --pairs-v2   ->  data/stimulus/pairs_v2_votes.tsv

Deterministic: pairs sorted by id; examples are each pair's first two
occurrences in master order.

    .venv/bin/python experiments/d5_cohere_track/training_data/prep_council_pairs_v2.py
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.substitution_policy import needs_judge, variant_from_rule_path

MASTER = ROOT / "data" / "generated" / "v2" / "master.jsonl"
CHUNK_DIR = ROOT / "data" / "generated" / "v2" / "council" / "pair_chunks"
EXAMPLES_PER_PAIR = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-size", type=int, default=300)
    args = parser.parse_args()

    verdict: dict[tuple[str, str, str], str] = {}
    pairs: dict[str, dict] = {}
    with MASTER.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for sub in row["substitutions"]:
                key = (sub["word"], sub["rendering"], sub["rule_path"])
                if key not in verdict:
                    verdict[key] = needs_judge(sub["word"], variant_from_rule_path(
                        sub["rendering"], sub["rule_path"]))
                if not verdict[key]:
                    continue
                pair_id = f"{sub['word']}->{sub['rendering']}"
                entry = pairs.setdefault(pair_id, {
                    "sid": pair_id, "rid": pair_id, "word": sub["word"],
                    "form": sub["rendering"], "why_flagged": verdict[key],
                    "examples": []})
                if len(entry["examples"]) < EXAMPLES_PER_PAIR:
                    entry["examples"].append({"base": row["base_text"],
                                              "respelled": row["text"]})

    judgeable = [pairs[k] for k in sorted(pairs)]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("*.json"):
        old.unlink()
    size = args.chunk_size
    for i in range(0, len(judgeable), size):
        (CHUNK_DIR / f"c{i // size}.json").write_text(
            json.dumps(judgeable[i:i + size], ensure_ascii=False), encoding="utf-8")
    n_chunks = (len(judgeable) + size - 1) // size
    print(f"{len(judgeable):,} flagged pairs -> {n_chunks} chunks of <= {size} "
          f"in {CHUNK_DIR} (x5 lenses = {n_chunks * 5} judge calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
