"""Chunk the v2 fused sentences into judge files for the 5-lens council.

Every row here already passed fusion.py's mechanical gates; the council
judges what gates cannot: is it natural Egyptian Arabizi, and -- because
these sentences feed the meaning rows -- is the anchor used in exactly the
sense its council-confirmed Arabic names? Judges see the Arabic and gloss.

Judge rows come back as {"sid", "rid", "v", "why"} (rid == sid: one fused
sentence, one verdict) and merge with the same code as Dataset A's council:

    merge_council.py --fusion-v2   ->  data/stimulus/fusion_v2_votes.tsv

Chunks and verdicts live under data/generated/v2/council/, not /tmp: the
a_v1 council died twice at the session limit and its verdicts had to be dug
out of transcripts. A durable directory makes a resume a directory listing.

    .venv/bin/python experiments/d5_cohere_track/training_data/prep_council_v2.py

Expansion runs pass their own fused files and a fresh chunk dir (--fused is
repeatable; a sentence_id seen in an earlier file wins, as in
build_dataset_v2):

    .venv/bin/python experiments/d5_cohere_track/training_data/prep_council_v2.py \\
        --fused data/generated/v2/fused_expansion.jsonl \\
        --chunk-dir data/generated/v2/council/fused_expansion_chunks

A chunk dir holds the files a council's judges read, and their verdicts are
only valid for exactly those items. So an existing chunk may only be
rewritten identically or extended at its end (the extended chunk's verdict
files then hold too few rows and council_jobs.py re-queues them); any other
change is refused unless --replace, which leaves every verdict judged on the
old chunks stale.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
# the TRACKED copy: fused sentences are cloud-model output and cannot be
# regenerated, so the committed file (with its audit) is the source
FUSED = ROOT / "data" / "training" / "v2" / "fused_sentences.jsonl"
CHUNK_DIR = ROOT / "data" / "generated" / "v2" / "council" / "fused_chunks"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-size", type=int, default=115)
    parser.add_argument("--fused", type=pathlib.Path, action="append", default=None,
                        help=f"fused sentences JSONL, repeatable (default: {FUSED.name})")
    parser.add_argument("--chunk-dir", type=pathlib.Path, default=CHUNK_DIR)
    parser.add_argument("--replace", action="store_true",
                        help="rewrite chunks even if that changes items a council judged")
    args = parser.parse_args(argv)
    if args.fused is None:
        args.fused = [FUSED]
    return args


def judge_item(row: dict) -> dict:
    return {"sid": row["sentence_id"], "rid": row["sentence_id"],
            "anchor": row["anchor"]["word"], "arabic": row["anchor"]["arabic"],
            "gloss": row["anchor"]["gloss_en"], "sentence": row["text"]}


def read_judgeable(paths: list[pathlib.Path]) -> list[dict]:
    judgeable = []
    seen: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                if row["sentence_id"] in seen:
                    continue  # repeated id: first row wins, as in build_dataset_v2
                try:
                    item = judge_item(row)
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no} ({row.get('sentence_id')}): "
                                     f"missing field {exc}") from exc
                seen.add(row["sentence_id"])
                judgeable.append(item)
    return judgeable


def unsafe_rewrites(chunk_dir: pathlib.Path, new: dict[str, str]) -> list[str]:
    """Existing chunks the new set would change other than by appending."""
    problems = []
    for path in sorted(chunk_dir.glob("c*.json")):
        if not re.fullmatch(r"c\d+\.json", path.name):
            continue
        try:
            old_items = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"{path.name} is unparsable and would be replaced")
            continue
        if path.name not in new:
            problems.append(f"{path.name} would be deleted")
            continue
        new_items = json.loads(new[path.name])
        if new_items[:len(old_items)] != old_items:
            problems.append(f"{path.name} would change items a council may have judged")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        judgeable = read_judgeable(args.fused)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    size = args.chunk_size
    chunks = {f"c{i // size}.json": json.dumps(judgeable[i:i + size], ensure_ascii=False)
              for i in range(0, len(judgeable), size)}
    chunk_dir = args.chunk_dir
    if chunk_dir.exists() and not args.replace:
        problems = unsafe_rewrites(chunk_dir, chunks)
        if problems:
            print(f"ERROR: refusing to rewrite {chunk_dir}: " + "; ".join(problems[:5])
                  + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
                  + ". Chunk into a fresh --chunk-dir, or pass --replace to invalidate "
                    "every verdict judged on these chunks.", file=sys.stderr)
            return 2
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # same end state as unlinking everything and rewriting, but a chunk a live
    # judge is reading never disappears when its content does not change
    for old in chunk_dir.glob("*.json"):
        if old.name not in chunks:
            old.unlink()
    for name, text in chunks.items():
        path = chunk_dir / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    n_chunks = len(chunks)
    print(f"{len(judgeable):,} fused sentences -> {n_chunks} chunks of <= {size} "
          f"in {chunk_dir} (x5 lenses = {n_chunks * 5} judge calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
