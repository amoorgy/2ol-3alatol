"""Write the v2 held-out word split: data/stimulus/nilechat/word_split.tsv.

Source of truth is the probe's own stimulus file (every base word and every
trap it tests), not a list derived from it -- so nothing the probe measures
can slip past the firewall. The human arm is always held out: hard rule 1's
headline claims must be shown on it, so it can never be trained on.

Deterministic (salted hash, see franco_eval.word_split); tracked output.

    python3 experiments/d5_cohere_track/training_data/split_words.py
"""

import argparse
import collections
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.word_split import HELDOUT, split_words

STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"
HUMAN_ARM = ROOT / "data" / "benchmark" / "pairs.csv"
OUT = ROOT / "data" / "stimulus" / "nilechat" / "word_split.tsv"
SALT = "v2-heldout-2026-09-23"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heldout-fraction", type=float, default=0.5)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    roles: dict[str, set[str]] = collections.defaultdict(set)
    pairs: list[tuple[str, str]] = []
    with STIMULI.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "base":
                roles[row["word_type"]].add("benchmark_word")
            elif row["role"] == "trap" and row["trap_word"]:
                roles[row["trap_word"]].add("trap")
                pairs.append((row["word_type"], row["trap_word"]))
    human: set[str] = set()
    with HUMAN_ARM.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for word in (row["a"].lower(), row["b"].lower()):
                human.add(word)
                roles[word].add("human_arm")

    sides = split_words(sorted(roles), pairs, salt=SALT,
                        heldout_fraction=args.heldout_fraction,
                        forced_heldout=frozenset(human))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "side", "roles"])
        for word in sorted(sides):
            writer.writerow([word, sides[word], "+".join(sorted(roles[word]))])

    counts = collections.Counter((sides[w], r) for w in sides for r in roles[w])
    for (side, role), n in sorted(counts.items()):
        print(f"{side:8s} {role:15s} {n:5d}")
    trap_pairs_train = sum(1 for a, _b in pairs if sides[a] != HELDOUT)
    print(f"trap pairs on the train side: {trap_pairs_train}/{len(pairs)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
