"""Build the labelled same/different pair set — the shared test set for both levels.

Two sections:
  SAME      pairs that really are one word spelled two ways
  DIFFERENT pairs that look like variants but mean different things

Section DIFFERENT is what stops a model from scoring well by answering "same" to
everything, so the two sections together give a confusion matrix rather than a
one-sided robustness score.

Labels come from meaning partitions in data/benchmark/probe_annotations.tsv:
pairs inside one partition are SAME, pairs crossing partitions are DIFFERENT.

    python3 experiments/d5_cohere_track/e1_variant_probe/build_pairs.py
"""

import argparse
import collections
import csv
import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.phenomena import UNRELATED, classify_edge  # noqa: E402

HAND = ROOT / "data" / "benchmark" / "probe_hand.json"
MINED = ROOT / "data" / "benchmark" / "variant_clusters.tsv"
ANNOT = ROOT / "data" / "benchmark" / "probe_annotations.tsv"
OUT = ROOT / "data" / "benchmark" / "pairs.csv"

FIELDS = [
    "pair_id", "label", "subcategory", "a", "b",
    "ortho_classes", "n_edits", "source", "cluster_id",
    "arabic", "meaning_en", "reason", "confirmed",
]


def load_annotations() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in ANNOT.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("cluster_id"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        cid, groups, subcat, reason = parts[0], parts[1], parts[2], parts[3]
        out[cid] = {
            "groups": [g.split() for g in groups.split("|")] if groups != "dedupe" else [],
            "subcategory": subcat,
            "reason": reason,
            "dedupe": groups == "dedupe",
        }
    return out


def load_clusters() -> list[dict]:
    clusters = []
    for c in json.loads(HAND.read_text(encoding="utf-8")):
        clusters.append({
            "cluster_id": c["cluster_id"], "source": "hand",
            "variants": c["variants"], "arabic": c.get("arabic", ""),
            "meaning_en": c.get("meaning_en", ""),
        })
    lines = MINED.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        skel, _total, forms = line.split("\t")
        variants = [c.split(":")[0] for c in forms.split() if ":" in c]
        if len(variants) >= 2:
            clusters.append({
                "cluster_id": f"mined_{skel}", "source": "mined_attested",
                "variants": variants, "arabic": "", "meaning_en": "",
            })
    return clusters


def group_of(form: str, groups: list[list[str]]) -> int:
    for i, g in enumerate(groups):
        if form in g:
            return i
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    annotations = load_annotations()
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    skipped_unrelated = 0

    for cluster in load_clusters():
        ann = annotations.get(cluster["cluster_id"])
        if ann and ann["dedupe"]:
            continue
        groups = ann["groups"] if ann else [list(cluster["variants"])]
        # forms not mentioned in any partition join the first group
        for form in cluster["variants"]:
            if group_of(form, groups) == -1:
                groups[0].append(form)

        for a, b in itertools.combinations(cluster["variants"], 2):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            edge = classify_edge(a, b)
            same = group_of(a, groups) == group_of(b, groups)
            if same and (UNRELATED in edge.classes or not edge.classes):
                # claimed identical in meaning but not explainable orthographically
                skipped_unrelated += 1
                continue
            rows.append({
                "pair_id": f"{cluster['cluster_id']}::{a}~{b}",
                "label": "SAME" if same else "DIFFERENT",
                "subcategory": (edge.classes[0].lower() if same
                                else (ann or {}).get("subcategory", "lexical")),
                "a": a, "b": b,
                "ortho_classes": "+".join(edge.classes) or "none",
                "n_edits": edge.n_edits,
                "source": cluster["source"],
                "cluster_id": cluster["cluster_id"],
                "arabic": cluster["arabic"],
                "meaning_en": cluster["meaning_en"],
                "reason": (ann or {}).get("reason", ""),
                "confirmed": "",   # native reviewer sets yes/no
            })

    rows.sort(key=lambda r: (r["label"], r["subcategory"], r["a"]))
    for i, r in enumerate(rows, 1):
        r["pair_id"] = f"p{i:04d}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = collections.Counter(r["label"] for r in rows)
    print(f"wrote {args.out.relative_to(ROOT)}: {len(rows)} pairs")
    print(f"  SAME      {counts['SAME']}")
    print(f"  DIFFERENT {counts['DIFFERENT']}")
    if skipped_unrelated:
        print(f"  skipped {skipped_unrelated} pairs claimed SAME but "
              f"orthographically unexplainable (needs review)")

    for label in ("SAME", "DIFFERENT"):
        sub = collections.Counter(r["subcategory"] for r in rows if r["label"] == label)
        print(f"\n  {label} by subcategory:")
        for k, v in sub.most_common():
            print(f"    {k:<18} {v}")

    print("\n  DIFFERENT section (the traps):")
    for r in rows:
        if r["label"] == "DIFFERENT":
            print(f"    {r['a']:>12} / {r['b']:<12} {r['subcategory']:<9} "
                  f"[{r['ortho_classes']}] {r['reason'][:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
