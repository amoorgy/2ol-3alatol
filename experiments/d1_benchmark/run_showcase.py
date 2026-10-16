#!/usr/bin/env python3
"""Direction-1 showcase: does spelling variation break retrieval, and does
normalization fix it — on synthetic AND on attested (real) variation?

End-to-end, seconds on a laptop, zero dependencies:

  1. Load the human-authored seed corpus (deduplicated; synthetic and
     re-exported records are excluded from mining by provenance).
  2. Mine attested variant clusters (real spelling pairs from real text).
  3. Build a retrieval benchmark: fragment queries (short word-windows, like
     a real search — whole messages are trivially identifiable) x k variants
     per intensity, with ONE fixed query set across all intensities and
     seeds averaged for variance control.
  4. Evaluate RAW char-trigram retrieval vs the same retrieval over the
     Franco canon, on BOTH synthetic perturbations and attested-variant
     substitutions. Because normalize.py inverts much of taxonomy.py by
     construction, the synthetic table alone would flatter normalization —
     so the report includes a leakage diagnostic (fraction of variants that
     normalize to the exact normalized source) and the attested table is
     the honest headline.
  5. Write data/benchmark/ artifacts + results/showcase.{json,md}.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from franco.io_utils import read_jsonl, seed_messages, write_jsonl  # noqa: E402
from franco.metrics import consistency_at_k, recall_at_1  # noqa: E402
from franco.mine import clusters_to_tsv, mine_variants  # noqa: E402
from franco.normalize import normalize  # noqa: E402
from franco.perturb import variants  # noqa: E402
from franco.retrieval import CharNgramIndex  # noqa: E402

SEED_FILES = (
    REPO / "data/seed/egyptian_franco_conversations.jsonl",
    REPO / "data/seed/egyptian_franco_multiturn.jsonl",
)
INTERIM = REPO / "data/interim/clean_franco_records.jsonl"
INTENSITIES = (0.2, 0.4, 0.6, 0.8)
SEEDS = (13, 17, 29, 41, 53)


def fragment(text: str, width: int, rng: random.Random) -> str:
    """A random `width`-word window of `text` (the whole text if shorter)."""
    tokens = text.split()
    if len(tokens) <= width:
        return text
    start = rng.randrange(len(tokens) - width + 1)
    return " ".join(tokens[start : start + width])


def build_mining_corpus(corpus: list[str]) -> list[str]:
    """Corpus + interim records, minus synthetic/re-exported text, deduped."""
    texts = list(corpus)
    if INTERIM.exists():
        rows = read_jsonl(INTERIM)
        kept = [
            r.get("text", "")
            for r in rows
            if r.get("source") != "local_seed"
            and "fusion" not in str(r.get("provenance", {}).get("file", ""))
        ]
        kept = [t for t in kept if t]
        print(
            f"interim: kept {len(kept)} of {len(rows)} records "
            f"(excluded seed re-exports and synthetic text by provenance)"
        )
        texts += kept
    deduped = list(dict.fromkeys(texts))
    if len(deduped) < len(texts):
        print(f"mining corpus deduplicated: {len(texts)} -> {len(deduped)} texts")
    return deduped


def build_benchmark(
    fragments: dict[int, str], k: int, seed: int
) -> tuple[dict[float, list[dict]], int]:
    """One fixed query set evaluated at every intensity.

    A qid is kept only if it yields k distinct variants at EVERY intensity,
    so each cell of the table scores the same queries. Returns (rows grouped
    by intensity, dropped-query count).
    """
    rng = random.Random(seed)
    per_intensity: dict[float, list[dict]] = {i: [] for i in INTENSITIES}
    dropped = 0
    for qid, frag in fragments.items():
        candidate = {}
        for intensity in INTENSITIES:
            vs = variants(frag, k, intensity, rng)
            if len(vs) < k:
                dropped += 1
                break
            candidate[intensity] = vs
        else:
            for intensity, vs in candidate.items():
                per_intensity[intensity].append(
                    {"qid": qid, "fragment": frag, "intensity": intensity,
                     "variants": vs}
                )
    return per_intensity, dropped


def attested_queries(
    fragments: dict[int, str], clusters: list[dict], rng: random.Random
) -> list[dict]:
    """Real-variation queries: swap words for attested alternate spellings."""
    alternates: dict[str, list[str]] = {}
    for cluster in clusters:
        forms = [w for w, _ in cluster["forms"]]
        for form in forms:
            alts = [x for x in forms if x != form]
            if alts:
                alternates[form] = alts
    rows = []
    for qid, frag in fragments.items():
        swapped, n_swaps = [], 0
        for token in frag.split():
            alts = alternates.get(token.lower())
            if alts:
                swapped.append(rng.choice(alts))
                n_swaps += 1
            else:
                swapped.append(token)
        if n_swaps:
            rows.append({"qid": qid, "query": " ".join(swapped), "swaps": n_swaps})
    return rows


def top1(index: CharNgramIndex, query: str) -> int | None:
    hits = index.search(query, topk=1)
    return hits[0][0] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=120)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--width", type=int, default=4, help="query window in words")
    args = parser.parse_args()
    if args.queries < 1 or args.k < 2 or args.width < 1:
        parser.error("need --queries >= 1, --k >= 2, --width >= 1")

    corpus = seed_messages(SEED_FILES)
    if not corpus:
        parser.error(f"no seed messages found under {SEED_FILES[0].parent}")
    print(f"corpus: {len(corpus)} unique human-authored messages")

    clusters = mine_variants(build_mining_corpus(corpus))
    tsv_path = REPO / "data/benchmark/variant_clusters.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.write_text(clusters_to_tsv(clusters), encoding="utf-8")
    print(f"mined {len(clusters)} attested variant clusters -> {tsv_path}")
    for cluster in clusters[:8]:
        forms = ", ".join(f"{w}({n})" for w, n in cluster["forms"][:4])
        print(f"  [{cluster['skeleton']}] {forms}")

    systems = {
        "raw": (CharNgramIndex().fit(corpus), lambda t: t),
        "normalized": (CharNgramIndex().fit([normalize(d) for d in corpus]), normalize),
    }

    cells: dict = {
        name: {i: {"recall": [], "consistency": []} for i in INTENSITIES}
        for name in systems
    }
    clean = {name: [] for name in systems}
    attested = {name: {"recall": [], "n": []} for name in systems}
    leakage = {i: [] for i in INTENSITIES}
    row_counts, all_rows = [], []

    for seed in SEEDS:
        rng = random.Random(seed)
        qids = sorted(rng.sample(range(len(corpus)), min(args.queries, len(corpus))))
        fragments = {qid: fragment(corpus[qid], args.width, rng) for qid in qids}
        bench, dropped = build_benchmark(fragments, args.k, seed)
        n_rows = len(bench[INTENSITIES[0]])
        if n_rows == 0:
            raise SystemExit(f"seed {seed}: no benchmark rows survived — aborting")
        row_counts.append(n_rows)
        if dropped:
            print(f"seed {seed}: kept {n_rows} queries, dropped {dropped} "
                  f"(could not yield {args.k} distinct variants at every intensity)")
        all_rows.extend(r for i in INTENSITIES for r in bench[i])

        for intensity in INTENSITIES:
            rows = bench[intensity]
            leakage[intensity].append(
                sum(
                    normalize(v) == normalize(r["fragment"])
                    for r in rows
                    for v in r["variants"]
                ) / (len(rows) * args.k)
            )
        for name, (index, transform) in systems.items():
            kept = {qid: frag for qid, frag in fragments.items()
                    if any(r["qid"] == qid for r in bench[INTENSITIES[0]])}
            clean[name].append(
                recall_at_1(
                    [top1(index, transform(f)) for f in kept.values()],
                    list(kept.keys()),
                )
            )
            for intensity in INTENSITIES:
                rows = bench[intensity]
                groups = [
                    [top1(index, transform(v)) for v in r["variants"]] for r in rows
                ]
                predictions = [p for g in groups for p in g]
                gold = [r["qid"] for r in rows for _ in r["variants"]]
                cells[name][intensity]["recall"].append(recall_at_1(predictions, gold))
                cells[name][intensity]["consistency"].append(consistency_at_k(groups))

        att_rows = attested_queries(fragments, clusters, rng)
        for name, (index, transform) in systems.items():
            if att_rows:
                attested[name]["recall"].append(
                    recall_at_1(
                        [top1(index, transform(r["query"])) for r in att_rows],
                        [r["qid"] for r in att_rows],
                    )
                )
                attested[name]["n"].append(len(att_rows))

    write_jsonl(REPO / "data/benchmark/retrieval_v0.jsonl", all_rows)
    print(f"benchmark: {len(all_rows)} rows -> data/benchmark/retrieval_v0.jsonl")

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    lines = [
        "# Showcase: retrieval robustness under Franco spelling variation",
        "",
        f"{mean(row_counts):.0f} fragment queries ({args.width}-word windows) x "
        f"{args.k} variants per intensity; corpus of {len(corpus)} human-authored "
        f"seed messages; mean over seeds {SEEDS}.",
        "",
        "## Synthetic perturbation (taxonomy-generated variants)",
        "",
        "`norm=source` is the leakage diagnostic: the fraction of variants that "
        "normalize to the exact normalized source — an oracle head-start the "
        "normalized arm gets because normalize.py inverts part of taxonomy.py "
        "by construction. Read the normalized column with that discount; the "
        "attested table below is free of this artifact.",
        "",
        "| intensity | recall@1 raw | recall@1 normalized | C@3 raw | C@3 normalized | norm=source |",
        "|---|---|---|---|---|---|",
        f"| 0.0 (clean) | {mean(clean['raw']):.3f} | {mean(clean['normalized']):.3f} "
        f"| — | — | — |",
    ]
    for intensity in INTENSITIES:
        raw, norm = cells["raw"][intensity], cells["normalized"][intensity]
        lines.append(
            f"| {intensity} | {mean(raw['recall']):.3f} | {mean(norm['recall']):.3f} "
            f"| {mean(raw['consistency']):.3f} | {mean(norm['consistency']):.3f} "
            f"| {mean(leakage[intensity]):.0%} |"
        )
    lines += [
        "",
        "## Attested variation (real spelling swaps from mined clusters)",
        "",
        "Queries where words are replaced by *attested* alternate spellings "
        "(mined from the corpus itself), not synthetic rules — the honest test.",
        "",
        "| system | recall@1 | n queries (mean/seed) |",
        "|---|---|---|",
    ]
    for name in systems:
        stats = attested[name]
        if stats["recall"]:
            lines.append(
                f"| {name} | {mean(stats['recall']):.3f} | {mean(stats['n']):.0f} |"
            )
    report = "\n".join(lines) + "\n"

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    payload = {
        "config": {**vars(args), "seeds": SEEDS, "intensities": INTENSITIES},
        "row_counts": row_counts,
        "clean": clean,
        "synthetic": cells,
        "leakage": leakage,
        "attested": attested,
    }
    (out / "showcase.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "showcase.md").write_text(report, encoding="utf-8")
    print()
    print(report)
    print("wrote results/showcase.json and results/showcase.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
