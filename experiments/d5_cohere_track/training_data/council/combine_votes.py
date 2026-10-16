"""Combine per-panel merged vote TSVs into one, with a `panel` column.

Two panels judged the v2 pair council at different bars: Opus x-high on
c0-c17 at >=4/5, and Sonnet medium on c18-c22 at 5/5 (user ruling
2026-09-26, after calibration showed Sonnet medium misses most collisions at
4/5 -- results/council_calibration_pairs_v2_c0_sonnet_medium_vs_opus.md). Each
panel is merged on its own with its own --min-agree; this joins the results,
records which panel and bar decided every row, and refuses a pair decided twice.

    python3 experiments/d5_cohere_track/training_data/council/combine_votes.py \\
        --out data/stimulus/pairs_v2_votes.tsv \\
        --part "opus/xhigh>=4=data/stimulus/pairs_v2_votes_opus.tsv" \\
        --part "sonnet/medium>=5=data/stimulus/pairs_v2_votes_sonnet.tsv"
"""

import argparse
import csv
import pathlib


def combine(parts: list[tuple[str, pathlib.Path]], out: pathlib.Path) -> int:
    header: list[str] | None = None
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for label, path in parts:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if header is None:
                header = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != header:
                raise ValueError(f"{path}: header {reader.fieldnames} != {header}")
            for row in reader:
                sid = row["sentence_id"]
                if sid in seen:
                    raise ValueError(f"{sid} decided by both {seen[sid]} and {label}")
                seen[sid] = label
                rows.append({**row, "panel": label})
    if header is None:
        raise ValueError("no parts given")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header + ["panel"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--part", action="append", required=True,
                        help="LABEL=TSV (the label records panel and bar)")
    args = parser.parse_args()
    parts = []
    for spec in args.part:
        label, _, path = spec.rpartition("=")
        parts.append((label, pathlib.Path(path)))
    n = combine(parts, args.out)
    print(f"combined {n} rows from {len(parts)} panels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
