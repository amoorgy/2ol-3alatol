"""Audit a bulk council: sample what it ACCEPTED, re-judge with Opus, estimate.

Bulk fleets run Sonnet (policy), and Sonnet is lenient: in a_v1 it rejected
2.5% of rows against Opus's 17.6%. This tool measures how much of that
leniency leaks into a bulk run's accepts, in two steps.

draw -- a stratified random sample of the items the bulk panel accepted
(>= min-agree/5), written as audit chunks c<N>.json holding the ORIGINAL
items unchanged (same keys, so the audit judges see exactly the prompt
contract the bulk judges saw, and nothing about the bulk verdict), shuffled
across strata, plus sample.json (strata, population sizes, seed, hashes of
every bulk file read). Strata (--strata, repeatable, combined):

    agreement            unanimous | dissent:<lens>   (the lens that objected)
    item:<field>         a chunk-item field, e.g. item:why_flagged
    source:<dotted.key>  a field of the --source JSONL row whose
                         sentence_id == sid, e.g. source:anchor.context_type

Allocation is proportional to stratum size (largest remainder) with a floor
of --min-per-stratum, so thin strata are oversampled; estimate reweights.

estimate -- after the audit panel (fused_council.js / pair_council.js with
opus/xhigh, chunkDir = the audit dir) has written its verdicts: the bulk
panel's false-accept rate against the audit panel, i.e. the share of bulk
accepts the audit panel rejects at >= reference-min-agree/5. Stratified
estimator sum_h W_h p_h (W_h = N_h / N) with a Wilson 95% interval at
Kish's effective sample size. The audit panel is a reference, not ground
truth: the rate is "vs <label>", never "error rate".

    python3 experiments/d5_cohere_track/training_data/council/sample_audit.py draw \\
        --council fused --chunk-dir <bulk chunks> --verdict-dir <bulk votes> \\
        --label sonnet/high --out-dir <audit chunks> --n 300 --seed 20260925
    python3 experiments/d5_cohere_track/training_data/council/sample_audit.py estimate \\
        --sample-dir <audit chunks> --verdict-dir <audit votes> --label opus/xhigh \\
        --name <name>   # -> experiments/d5_cohere_track/results/council_audit_<name>.*
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shlex
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import council_io as cio  # noqa: E402

DEFAULT_SEED = 20260925
MANIFEST = "sample.json"
SCRIPT = "python3 experiments/d5_cohere_track/training_data/council/sample_audit.py"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="sample bulk-accepted items into audit chunks")
    d.add_argument("--council", required=True, choices=sorted(cio.COUNCILS))
    d.add_argument("--chunk-dir", required=True, type=pathlib.Path)
    d.add_argument("--verdict-dir", required=True, type=pathlib.Path, help="bulk panel votes")
    d.add_argument("--label", required=True, help="bulk panel, e.g. sonnet/high")
    d.add_argument("--out-dir", required=True, type=pathlib.Path, help="audit chunk dir")
    d.add_argument("--n", type=int, default=300,
                   help="target sample size (per-stratum floors can raise it)")
    d.add_argument("--min-per-stratum", type=int, default=20)
    d.add_argument("--seed", type=int, default=DEFAULT_SEED)
    d.add_argument("--strata", action="append", default=None,
                   help="agreement | item:<field> | source:<dotted.key> (default: agreement)")
    d.add_argument("--source", type=pathlib.Path, help="JSONL joined on --source-key == sid")
    d.add_argument("--source-key", default="sentence_id")
    d.add_argument("--min-agree", type=int, default=cio.MIN_AGREE,
                   help="the threshold that defined the bulk panel's accepts")
    d.add_argument("--chunk-size", type=int,
                   help="audit chunk size (default: the largest bulk chunk, to match load)")
    d.add_argument("--allow-incomplete", action="store_true",
                   help="sample among complete items even if the bulk panel is unfinished")
    d.add_argument("--replace", action="store_true", help="overwrite an existing audit dir")
    e = sub.add_parser("estimate", help="false-accept rate from the audit verdicts")
    e.add_argument("--sample-dir", required=True, type=pathlib.Path)
    e.add_argument("--verdict-dir", required=True, type=pathlib.Path, help="audit panel votes")
    e.add_argument("--label", required=True, help="audit panel, e.g. opus/xhigh")
    e.add_argument("--name", required=True, help="results/council_audit_<name>.*")
    e.add_argument("--reference-min-agree", type=int, default=cio.MIN_AGREE)
    e.add_argument("--allow-incomplete", action="store_true",
                   help="estimate from the audited items only (flagged in the report)")
    e.add_argument("--results-dir", type=pathlib.Path, default=cio.RESULTS_DIR)
    return parser.parse_args(argv)


def command(argv: list[str] | None) -> str:
    return f"{SCRIPT} {shlex.join(sys.argv[1:] if argv is None else argv)}"


# -------------------------------------------------------------------- draw

def dotted(row: dict | None, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def load_source(path: pathlib.Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.setdefault(str(row.get(key)), row)   # first row wins
    return rows


def stratum_of(specs: list[str], item: dict, verdict: cio.Verdict, lenses: tuple[str, ...],
               source_row: dict | None) -> str:
    parts = []
    for spec in specs:
        if spec == "agreement":
            objected = cio.dissent(verdict, lenses)
            parts.append("unanimous" if not objected else "dissent:" + "+".join(objected))
            continue
        kind, _, field = spec.partition(":")
        value = item.get(field) if kind == "item" else dotted(source_row, field)
        parts.append(f"{field}={'(missing)' if value is None else value}")
    return " | ".join(parts)


def check_specs(specs: list[str], source: pathlib.Path | None) -> None:
    for spec in specs:
        kind, sep, field = spec.partition(":")
        if spec == "agreement" or (kind in ("item", "source") and sep and field):
            if kind == "source" and source is None:
                raise cio.CouncilDataError(f"--strata {spec} needs --source")
            continue
        raise cio.CouncilDataError(
            f"--strata {spec!r}: expected agreement, item:<field> or source:<dotted.key>")


def allocate(sizes: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, then a per-stratum floor."""
    total = sum(sizes.values())
    quota = {h: n * size / total for h, size in sizes.items()}
    alloc = {h: int(q) for h, q in quota.items()}
    spare = n - sum(alloc.values())
    for h in sorted(sizes, key=lambda h: (-(quota[h] - alloc[h]), h))[:max(spare, 0)]:
        alloc[h] += 1
    return {h: min(sizes[h], max(floor, alloc[h])) for h in sizes}


def draw(args: argparse.Namespace) -> dict:
    lenses = cio.COUNCILS[args.council]
    specs = args.strata or ["agreement"]
    check_specs(specs, args.source)
    if args.n < 1 or args.min_per_stratum < 1:
        raise cio.CouncilDataError("--n and --min-per-stratum must be >= 1")
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.replace:
        raise cio.CouncilDataError(
            f"{out_dir} is not empty; pass --replace to overwrite it (any audit verdicts "
            f"already judged on it become stale)")
    chunk_dir = args.chunk_dir.resolve()
    chunks = cio.load_chunks(chunk_dir)
    panel = cio.load_panel(args.verdict_dir, chunks, lenses)
    pending, _ = cio.pending_jobs(chunks, args.verdict_dir, lenses)
    source = load_source(args.source, args.source_key) if args.source else None

    strata: dict[str, list[dict]] = {}
    n_items = n_incomplete = n_rejected = n_no_source = 0
    for items in chunks.values():
        for item in items:
            n_items += 1
            verdict = cio.final_verdict(panel, cio.item_key(item), lenses, args.min_agree)
            if verdict is None:
                n_incomplete += 1
                continue
            if not verdict.accept:
                n_rejected += 1
                continue
            row = source.get(item["sid"]) if source is not None else None
            n_no_source += source is not None and row is None
            strata.setdefault(stratum_of(specs, item, verdict, lenses, row), []).append(item)
    if n_incomplete and not args.allow_incomplete:
        raise cio.CouncilDataError(
            f"bulk panel incomplete: {n_incomplete:,}/{n_items:,} items lack a verdict on "
            f"some lens ({len(pending)} pending jobs). Finish it (council_jobs.py) or pass "
            f"--allow-incomplete")
    if not strata:
        raise cio.CouncilDataError("the bulk panel accepted nothing; there is nothing to audit")

    names = sorted(strata)
    alloc = allocate({h: len(strata[h]) for h in names}, args.n, args.min_per_stratum)
    sample = []
    for h in names:
        pool = sorted(strata[h], key=cio.item_key)
        picked = random.Random(f"{args.seed}|{h}").sample(pool, alloc[h])
        sample += [(h, item) for item in picked]
    random.Random(f"{args.seed}|order").shuffle(sample)

    size = args.chunk_size or max(len(c) for c in chunks.values())
    if size < 1:
        raise cio.CouncilDataError("--chunk-size must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    entries = []
    for start in range(0, len(sample), size):
        name = f"c{start // size}"
        batch = sample[start:start + size]
        (out_dir / f"{name}.json").write_text(
            json.dumps([item for _, item in batch], ensure_ascii=False), encoding="utf-8")
        entries += [{"sid": item["sid"], "rid": item["rid"], "stratum": h, "chunk": name}
                    for h, item in batch]

    manifest = {
        "kind": "council_audit_sample", "council": args.council, "lenses": list(lenses),
        "command": command(args.argv), "seed": args.seed, "strata_spec": specs,
        "n_requested": args.n, "min_per_stratum": args.min_per_stratum,
        "chunk_size": size,
        "bulk": {"label": args.label, "chunk_dir": cio.rel(chunk_dir),
                 "verdict_dir": cio.rel(args.verdict_dir), "min_agree": args.min_agree,
                 "stamp": cio.read_stamp(args.verdict_dir),
                 "chunk_sha256": cio.chunk_hashes(chunk_dir, chunks),
                 "verdict_sha256": cio.verdict_hashes(args.verdict_dir, chunks, lenses)},
        "source": None if args.source is None else {
            "path": cio.rel(args.source), "key": args.source_key,
            "sha256": cio.sha256_file(args.source), "n_items_without_row": n_no_source},
        "population": {"n_items": n_items, "n_incomplete": n_incomplete,
                       "n_rejected": n_rejected,
                       "n_accepted": sum(len(v) for v in strata.values())},
        "strata": [{"stratum": h, "N": len(strata[h]), "n": alloc[h]} for h in names],
        "sample": entries,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- estimate

def estimate(args: argparse.Namespace) -> dict:
    sample_dir = args.sample_dir.resolve()
    manifest_path = sample_dir / MANIFEST
    if not manifest_path.exists():
        raise cio.CouncilDataError(f"{manifest_path} not found: run `draw` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lenses = tuple(manifest["lenses"])
    chunks = cio.load_chunks(sample_dir)
    audit_keys = {cio.item_key(i) for c in chunks.values() for i in c}
    sampled = {(e["sid"], e["rid"]): e for e in manifest["sample"]}
    if set(sampled) != audit_keys:
        raise cio.CouncilDataError(f"audit chunks in {sample_dir} do not match {MANIFEST}")

    warnings = []
    bulk = manifest["bulk"]
    bulk_chunks = cio.ROOT / bulk["chunk_dir"]
    if bulk_chunks.is_dir():
        now = cio.verdict_hashes(cio.ROOT / bulk["verdict_dir"], cio.load_chunks(bulk_chunks),
                                 lenses)
        if now != bulk["verdict_sha256"] or \
                cio.chunk_hashes(bulk_chunks, cio.load_chunks(bulk_chunks)) != bulk["chunk_sha256"]:
            warnings.append("bulk chunks or verdicts changed since the draw: the stratum sizes "
                            "(weights) may be stale; redraw before reporting")
    else:
        warnings.append(f"bulk chunk dir {bulk['chunk_dir']} not found; cannot verify the draw "
                        f"is current")

    panel = cio.load_panel(args.verdict_dir, chunks, lenses)
    pending, _ = cio.pending_jobs(chunks, args.verdict_dir, lenses)
    verdicts = {k: cio.final_verdict(panel, k, lenses, args.reference_min_agree)
                for k in sampled}
    unjudged = [k for k, v in verdicts.items() if v is None]
    if unjudged and not args.allow_incomplete:
        raise cio.CouncilDataError(
            f"audit incomplete: {len(unjudged)}/{len(sampled)} sampled items lack a verdict "
            f"on some lens ({len(pending)} pending jobs). Finish it (council_jobs.py) or "
            f"pass --allow-incomplete")
    if unjudged:
        warnings.append(f"INCOMPLETE audit: {len(unjudged)}/{len(sampled)} sampled items "
                        f"unjudged and excluded")

    rows, false_accepts, lens_rejects = [], [], {lens: 0 for lens in lenses}
    for s in manifest["strata"]:
        judged = [k for k, e in sampled.items() if e["stratum"] == s["stratum"] and verdicts[k]]
        rejects = [k for k in judged if not verdicts[k].accept]
        if s["N"] and not judged:
            raise cio.CouncilDataError(f"stratum {s['stratum']!r} has no audited item")
        rows.append({"stratum": s["stratum"], "N": s["N"], "n_drawn": s["n"],
                     "n_judged": len(judged), "reference_rejects": len(rejects),
                     **{k: v for k, v in cio.proportion(len(rejects), len(judged)).items()
                        if k in ("rate", "wilson95")}})
        for k in judged:
            for lens in cio.dissent(verdicts[k], lenses):
                lens_rejects[lens] += 1
        false_accepts += [{"sid": k[0], "rid": k[1], "stratum": s["stratum"],
                           "votes": f"{verdicts[k].n_valid}/{len(lenses)}",
                           "rejected_on": {lens: panel[lens][k].get("why", "")
                                           for lens in cio.dissent(verdicts[k], lenses)}}
                          for k in sorted(rejects)]
    est = cio.stratified_rate([(r["N"], r["n_judged"], r["reference_rejects"]) for r in rows])
    n_judged = sum(r["n_judged"] for r in rows)
    pooled = cio.proportion(sum(r["reference_rejects"] for r in rows), n_judged)
    return {
        "name": args.name, "command": command(args.argv),
        "seed": manifest["seed"], "draw_command": manifest["command"],
        "config": {"council": manifest["council"], "lenses": list(lenses),
                   "bulk_label": bulk["label"], "bulk_min_agree": bulk["min_agree"],
                   "reference_label": args.label,
                   "reference_min_agree": args.reference_min_agree,
                   "reference_verdict_dir": cio.rel(args.verdict_dir),
                   "reference_stamp": cio.read_stamp(args.verdict_dir),
                   "sample_dir": cio.rel(sample_dir), "strata_spec": manifest["strata_spec"]},
        "population": manifest["population"],
        "sample": {"n_drawn": len(sampled), "n_judged": n_judged,
                   "manifest_sha256": cio.sha256_file(manifest_path)},
        "reference_verdict_sha256": cio.verdict_hashes(args.verdict_dir, chunks, lenses),
        "warnings": warnings,
        "false_accept": {"rate": round(est["rate"], 4), "n_eff": round(est["n_eff"], 2),
                         "wilson95": [round(x, 4) for x in est["wilson95"]],
                         "estimator": "stratified sum_h W_h p_h; Wilson at Kish n_eff"},
        "false_accept_unweighted": pooled,
        "strata": rows,
        "reference_rejects_by_lens": lens_rejects,
        "false_accepts": false_accepts,
    }


def render_md(r: dict) -> str:
    c, fa = r["config"], r["false_accept"]
    lo, hi = fa["wilson95"]
    head = (f"Estimated false-accept rate of the {c['bulk_label']} bulk panel vs "
            f"{c['reference_label']}: {cio.pct(fa['rate'])} [{cio.pct(lo)}, {cio.pct(hi)}] "
            f"of its {r['population']['n_accepted']:,} accepts (stratified, "
            f"{r['sample']['n_judged']:,} audited, n_eff {fa['n_eff']:.1f}).")
    out = [f"# Council audit: {r['name']}", "", f"**{head}**", ""]
    for w in r["warnings"]:
        out += [f"> **WARNING.** {w}", ""]
    pop = r["population"]
    pooled = r["false_accept_unweighted"]
    out += ["Reproduce (draw, then estimate after the audit panel ran):", "", "```",
            r["draw_command"], r["command"], "```", "",
            "## Config", "", "| field | value |", "|---|---|",
            f"| council | {c['council']} (lenses: {', '.join(c['lenses'])}) |",
            f"| bulk panel | {c['bulk_label']}, accept at >= {c['bulk_min_agree']}/5 |",
            f"| reference panel | {c['reference_label']}, reject at < "
            f"{c['reference_min_agree']}/5 -- `{c['reference_verdict_dir']}` |",
            f"| population | {pop['n_items']:,} items: {pop['n_accepted']:,} accepted, "
            f"{pop['n_rejected']:,} rejected, {pop['n_incomplete']:,} incomplete |",
            f"| strata | {', '.join(c['strata_spec'])} |",
            f"| seed | {r['seed']} |", "",
            "## Estimate", "", "| estimator | value |", "|---|---|",
            f"| stratified (reported) | {cio.pct(fa['rate'])} [{cio.pct(lo)}, {cio.pct(hi)}], "
            f"n_eff {fa['n_eff']:.1f} |",
            f"| unweighted pooled (context only) | {cio.interval(pooled)} |", "",
            "## Strata", "",
            "| stratum | N (accepted) | drawn | judged | reference rejects | rate [Wilson 95%] |",
            "|---|---:|---:|---:|---:|---|"]
    for s in r["strata"]:
        rate = ("n/a" if s["rate"] is None else
                f"{cio.pct(s['rate'])} [{cio.pct(s['wilson95'][0])}, {cio.pct(s['wilson95'][1])}]")
        out.append(f"| {cio.md_cell(s['stratum'], 70)} | {s['N']:,} | {s['n_drawn']:,} | "
                   f"{s['n_judged']:,} | {s['reference_rejects']:,} | {rate} |")
    out += ["", "## Reference rejections by lens (audited items, unweighted)", "",
            "| lens | rejecting votes |", "|---|---:|"]
    out += [f"| {lens} | {n:,} |" for lens, n in r["reference_rejects_by_lens"].items()]
    out += ["", f"## Bulk accepts the reference rejected ({len(r['false_accepts']):,})", ""]
    if r["false_accepts"]:
        out += ["| sid | stratum | reference votes | rejected on |", "|---|---|---|---|"]
        for f in r["false_accepts"]:
            why = "; ".join(f"{k}: {v}" for k, v in f["rejected_on"].items())
            out.append(f"| {cio.md_cell(f['sid'], 40)} | {cio.md_cell(f['stratum'], 50)} | "
                       f"{f['votes']} | {cio.md_cell(why, 100)} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.argv = argv
    try:
        if args.cmd == "draw":
            manifest = draw(args)
            print(f"drew {len(manifest['sample']):,} of {manifest['population']['n_accepted']:,} "
                  f"accepted items in {len(manifest['strata'])} strata -> "
                  f"{cio.rel(args.out_dir)}")
            for s in manifest["strata"]:
                print(f"  {s['n']:>5} / {s['N']:<6} {s['stratum']}")
            if manifest["source"] and manifest["source"]["n_items_without_row"]:
                print(f"WARNING: {manifest['source']['n_items_without_row']} accepted items "
                      f"have no --source row (stratum value '(missing)')", file=sys.stderr)
            return 0
        cio.check_name(args.name)
        report = estimate(args)
    except cio.CouncilDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for w in report["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"council_audit_{args.name}.json"
    out_md = args.results_dir / f"council_audit_{args.name}.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report).splitlines()[2].strip("*"))
    print(f"wrote {cio.rel(out_json)} and {out_md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
