"""Blind N-arm council audit prep for the Dataset B fusion pilot.

Pools accepted sentences from several fusor-arm output files (one per
build_dataset_b.py run), relabels each anchor's candidates with an opaque
arm label (shuffled per anchor so position never leaks identity across
anchors), and chunks them into judge files for a naturalness council --
the same "many lenses, majority bar" convention build_dataset_a.py's
prep_council() uses, applied to a different judging question (does this
read like a natural Egyptian sentence, not "is this word substitution
valid"). Every row here already passed fusion.py's mechanical gates
(build_dataset_b.py only writes gate-passed rows to its --out file), so the
council's job is exactly the part the gates can't check.

The label -> real fusor-model mapping is written to a PRIVATE key file,
never shown to the council, and is the only place arm identity survives
until after judging.

    .venv/bin/python experiments/d5_cohere_track/training_data/prep_council_fusion.py \
        --arm data/generated/fusion_pilot/arm_qwen.jsonl \
        --arm data/generated/fusion_pilot/arm_falcon.jsonl \
        --arm data/generated/fusion_pilot/arm_glm.jsonl
"""

import argparse
import json
import pathlib
import random

CHUNK_DIR = pathlib.Path("/tmp/council_fb_chunks")
KEY_OUT = (pathlib.Path(__file__).resolve().parents[3] / "data" / "generated"
          / "fusion_pilot" / "arm_key.json")
LABEL_SEED = 17
LABELS = ["Arm A", "Arm B", "Arm C", "Arm D", "Arm E"]


def load_arm(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, dest="arms",
                        help="one accepted-sentences JSONL per fusor arm; "
                             "repeat for each arm")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(LABEL_SEED)
    by_anchor: dict[str, list[dict]] = {}
    for arm_path in args.arms:
        for row in load_arm(pathlib.Path(arm_path)):
            fusor = row["doc_id"].removeprefix("fusion:")
            by_anchor.setdefault(row["anchor"]["word"], []).append(
                {**row, "_fusor": fusor})

    key: dict[str, str] = {}
    judgeable: list[dict] = []
    for anchor_word, rows in by_anchor.items():
        order = list(range(len(rows)))
        rng.shuffle(order)
        for slot, i in enumerate(order):
            row = rows[i]
            label = LABELS[slot]
            row_id = f"{anchor_word}:{label}"
            key[row_id] = row["_fusor"]
            judgeable.append({
                "row_id": row_id, "arm": label,
                "anchor": anchor_word, "arabic": row["anchor"]["arabic"],
                "sentence": row["text"],
            })

    KEY_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEY_OUT.write_text(json.dumps(key, indent=1, ensure_ascii=False),
                       encoding="utf-8")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("*.json"):
        old.unlink()
    size = args.chunk_size
    for i in range(0, len(judgeable), size):
        (CHUNK_DIR / f"c{i // size}.json").write_text(
            json.dumps(judgeable[i:i + size], ensure_ascii=False),
            encoding="utf-8")
    n_chunks = (len(judgeable) + size - 1) // size
    print(f"{len(judgeable):,} candidate sentences from {len(by_anchor):,} "
          f"anchors -> {n_chunks} chunks of <= {size} in {CHUNK_DIR}")
    print(f"private arm key (NOT for judges) -> {KEY_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
