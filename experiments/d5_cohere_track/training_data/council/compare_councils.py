"""Compare two council panels that judged the same chunks (calibration).

A is the reference panel (opus/xhigh), B the candidate (sonnet/high). Both
verdict dirs must hold <lens>_c<N>.json judge files over the SAME chunk dir.
Reports, for B against A:

  * per lens: accept rates, leniency (B accept - A accept, pp), Cohen's kappa
  * final verdicts at >= min-agree/5: confusion matrix, reject rates, the
    leniency delta (A reject rate - B reject rate, pp; positive = B lenient),
    B's false-accept rate vs A (share of B accepts that A rejects) and B's
    miss rate (share of A rejects that B accepts), both with Wilson 95%
  * B at every threshold 1..5 against A at min-agree (calibration data only:
    a threshold chosen here must be validated on a separate audit sample,
    sample_audit.py, never on these chunks)
  * optional --by FIELD breakdowns over a chunk-item field (e.g. why_flagged)
  * every item whose final verdict flips, with each panel's reasons

Items are compared only where both panels are complete; missing coverage is
reported in the output, never silently dropped. Deterministic: no sampling,
no seed, byte-identical output on rerun. Writes
experiments/d5_cohere_track/results/council_calibration_<name>.{json,md}.

    python3 experiments/d5_cohere_track/training_data/council/compare_councils.py \\
        --council fused --chunk-dir data/generated/v2/council/fused_chunks \\
        --a-dir data/generated/v2/council/fused_votes --a-label opus/xhigh \\
        --b-dir data/generated/v2/council/fused_votes_sonnet --b-label sonnet/high \\
        --name fused_v2_sonnet_vs_opus
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import council_io as cio  # noqa: E402

sys.path.insert(0, str(cio.ROOT / "experiments" / "d5_cohere_track" / "nilechat_wordset"))
from merge_council import fleiss_kappa  # noqa: E402

MD_DISAGREEMENTS = 40


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--council", required=True, choices=sorted(cio.COUNCILS))
    parser.add_argument("--chunk-dir", required=True, type=pathlib.Path)
    parser.add_argument("--a-dir", required=True, type=pathlib.Path, help="reference panel")
    parser.add_argument("--a-label", required=True, help="e.g. opus/xhigh")
    parser.add_argument("--b-dir", required=True, type=pathlib.Path, help="candidate panel")
    parser.add_argument("--b-label", required=True, help="e.g. sonnet/high")
    parser.add_argument("--name", required=True, help="results/council_calibration_<name>.*")
    parser.add_argument("--min-agree", type=int, default=cio.MIN_AGREE)
    parser.add_argument("--chunks", help="comma-separated chunk subset (default: all)")
    parser.add_argument("--by", action="append", default=[],
                        help="chunk-item field to break final verdicts down by (repeatable)")
    parser.add_argument("--require-complete", action="store_true",
                        help="exit 2 instead of reporting when either panel is incomplete")
    parser.add_argument("--results-dir", type=pathlib.Path, default=cio.RESULTS_DIR)
    return parser.parse_args(argv)


def lens_stats(pa: list[bool], pb: list[bool]) -> dict:
    n = len(pa)
    counts = {"both_accept": sum(a and b for a, b in zip(pa, pb)),
              "a_only_accept": sum(a and not b for a, b in zip(pa, pb)),
              "b_only_accept": sum(b and not a for a, b in zip(pa, pb)),
              "both_reject": sum(not a and not b for a, b in zip(pa, pb))}
    a_rate = sum(pa) / n if n else None
    b_rate = sum(pb) / n if n else None
    return {"n": n,
            "a_accept_rate": None if a_rate is None else round(a_rate, 4),
            "b_accept_rate": None if b_rate is None else round(b_rate, 4),
            "leniency_pp": None if n == 0 else round(100 * (b_rate - a_rate), 2),
            "cohen_kappa": cio.cohen_kappa(pa, pb), "counts": counts}


def verdict_stats(a_acc: list[bool], b_acc: list[bool]) -> dict:
    n = len(a_acc)
    pairs = list(zip(a_acc, b_acc))
    confusion = {"a_accept_b_accept": sum(a and b for a, b in pairs),
                 "a_accept_b_reject": sum(a and not b for a, b in pairs),
                 "a_reject_b_accept": sum(b and not a for a, b in pairs),
                 "a_reject_b_reject": sum(not a and not b for a, b in pairs)}
    a_rej = (n - sum(a_acc)) / n if n else None
    b_rej = (n - sum(b_acc)) / n if n else None
    return {
        "n": n, "confusion": confusion,
        "a": {"accept_rate": None if n == 0 else round(1 - a_rej, 4),
              "reject_rate": None if n == 0 else round(a_rej, 4)},
        "b": {"accept_rate": None if n == 0 else round(1 - b_rej, 4),
              "reject_rate": None if n == 0 else round(b_rej, 4)},
        "leniency_pp": None if n == 0 else round(100 * (a_rej - b_rej), 2),
        "agreement": None if n == 0 else round(sum(a == b for a, b in pairs) / n, 4),
        "cohen_kappa": cio.cohen_kappa(a_acc, b_acc),
        "b_false_accept_vs_a": cio.proportion(confusion["a_reject_b_accept"], sum(b_acc)),
        "b_miss_rate_vs_a": cio.proportion(confusion["a_reject_b_accept"], n - sum(a_acc)),
    }


def compare(args: argparse.Namespace) -> dict:
    lenses = cio.COUNCILS[args.council]
    chunk_dir = args.chunk_dir.resolve()
    chunks = cio.select_chunks(cio.load_chunks(chunk_dir), args.chunks)
    items = [i for c in chunks.values() for i in c]
    keys = [cio.item_key(i) for i in items]
    panels = {"a": cio.load_panel(args.a_dir, chunks, lenses),
              "b": cio.load_panel(args.b_dir, chunks, lenses)}
    pending = {side: cio.pending_jobs(chunks, d, lenses)[0]
               for side, d in (("a", args.a_dir), ("b", args.b_dir))}
    finals = {side: {k: cio.final_verdict(panels[side], k, lenses, args.min_agree)
                     for k in keys} for side in panels}
    both = [k for k in keys if finals["a"][k] and finals["b"][k]]
    by_key = dict(zip(keys, items))
    chunk_of = {cio.item_key(i): c for c, rows in chunks.items() for i in rows}

    per_lens = {}
    for lens in lenses:
        shared = [k for k in keys if k in panels["a"][lens] and k in panels["b"][lens]]
        per_lens[lens] = lens_stats([panels["a"][lens][k]["v"] for k in shared],
                                    [panels["b"][lens][k]["v"] for k in shared])

    a_acc = [finals["a"][k].accept for k in both]
    final = verdict_stats(a_acc, [finals["b"][k].accept for k in both])
    sensitivity = []
    for k_min in range(1, len(lenses) + 1):
        stats = verdict_stats(a_acc, [finals["b"][k].n_valid >= k_min for k in both])
        sensitivity.append({"b_min_agree": k_min, "b_accept_rate": stats["b"]["accept_rate"],
                            "agreement": stats["agreement"], "cohen_kappa": stats["cohen_kappa"],
                            "b_false_accept_vs_a": stats["b_false_accept_vs_a"],
                            "b_miss_rate_vs_a": stats["b_miss_rate_vs_a"]})

    by = {}
    for field in args.by:
        groups: dict[str, list] = {}
        for k in both:
            groups.setdefault(str(by_key[k].get(field, "(missing)")), []).append(k)
        by[field] = {}
        for value in sorted(groups):
            g = groups[value]
            stats = verdict_stats([finals["a"][k].accept for k in g],
                                  [finals["b"][k].accept for k in g])
            by[field][value] = {"n": stats["n"], "a_reject_rate": stats["a"]["reject_rate"],
                                "b_reject_rate": stats["b"]["reject_rate"],
                                "leniency_pp": stats["leniency_pp"],
                                "b_miss_rate_vs_a": stats["b_miss_rate_vs_a"]}

    disagreements = []
    for k in both:
        va, vb = finals["a"][k], finals["b"][k]
        if va.accept == vb.accept:
            continue
        item = by_key[k]
        disagreements.append({
            "sid": k[0], "rid": k[1], "chunk": chunk_of[k],
            "text": item.get("sentence") or f"{item.get('word')} -> {item.get('form')}",
            "a_accept": va.accept, "b_accept": vb.accept,
            "a_votes": f"{va.n_valid}/{len(lenses)}", "b_votes": f"{vb.n_valid}/{len(lenses)}",
            "a_rejected_on": {lens: panels["a"][lens][k].get("why", "")
                              for lens in cio.dissent(va, lenses)},
            "b_rejected_on": {lens: panels["b"][lens][k].get("why", "")
                              for lens in cio.dissent(vb, lenses)},
        })

    # over the matched items only, so the two panels' kappas share a denominator
    kappa_rows = {side: [["valid" if v else "invalid" for v in finals[side][k].votes]
                         for k in both] for side in panels}
    stamps = {side: cio.read_stamp(d) for side, d in (("a", args.a_dir), ("b", args.b_dir))}
    warnings = []
    for side, label in (("a", args.a_label), ("b", args.b_label)):
        stamp = stamps[side]
        if stamp and f"{stamp.get('model')}/{stamp.get('effort')}" != label:
            warnings.append(f"panel {side.upper()} label {label!r} disagrees with its stamp "
                            f"{stamp.get('model')}/{stamp.get('effort')}")
        if stamp and stamp.get("chunk_dir") and \
                (cio.ROOT / stamp["chunk_dir"]).resolve() != chunk_dir:
            warnings.append(f"panel {side.upper()} was stamped for chunk dir "
                            f"{stamp['chunk_dir']}, not {cio.rel(chunk_dir)}")
    if args.a_dir.resolve() == args.b_dir.resolve():
        warnings.append("A and B are the same verdict dir (a sanity check, not a calibration)")
    if args.a_label == args.b_label:
        warnings.append("A and B carry the same label")

    complete = not pending["a"] and not pending["b"] and len(both) == len(keys)
    return {
        "name": args.name,
        "command": "python3 experiments/d5_cohere_track/training_data/council/"
                   "compare_councils.py " + shlex.join(sys.argv[1:] if args.argv is None
                                                        else args.argv),
        "seed": None, "deterministic": True,
        "config": {"council": args.council, "lenses": list(lenses),
                   "min_agree": args.min_agree, "chunk_dir": cio.rel(chunk_dir),
                   "chunks": list(chunks), "by": args.by,
                   "a": {"label": args.a_label, "dir": cio.rel(args.a_dir), "stamp": stamps["a"]},
                   "b": {"label": args.b_label, "dir": cio.rel(args.b_dir), "stamp": stamps["b"]}},
        "corpus": {"n_chunks": len(chunks), "n_items": len(keys),
                   "chunk_sha256": cio.chunk_hashes(chunk_dir, chunks)},
        "verdict_sha256": {"a": cio.verdict_hashes(args.a_dir, chunks, lenses),
                           "b": cio.verdict_hashes(args.b_dir, chunks, lenses)},
        "coverage": {"complete": complete,
                     "a_complete_items": sum(1 for k in keys if finals["a"][k]),
                     "b_complete_items": sum(1 for k in keys if finals["b"][k]),
                     "both_complete_items": len(both),
                     "a_pending_jobs": pending["a"], "b_pending_jobs": pending["b"]},
        "warnings": warnings,
        "final": final,
        "fleiss_kappa_across_lenses": {
            side: fleiss_kappa(kappa_rows[side], ["valid", "invalid"]) for side in panels},
        "lenses": per_lens,
        "b_threshold_sensitivity": sensitivity,
        "by": by,
        "disagreements": disagreements,
    }


def headline(r: dict) -> str:
    c, f = r["config"], r["final"]
    if f["n"] == 0:
        return "No item has complete verdicts from both panels; nothing to compare."
    fa = f["b_false_accept_vs_a"]
    fa_text = (f"{cio.pct(fa['rate'])} [{cio.pct(fa['wilson95'][0])}, "
               f"{cio.pct(fa['wilson95'][1])}] of {c['b']['label']}'s accepts are rejected "
               f"by {c['a']['label']}" if fa["rate"] is not None
               else f"{c['b']['label']} accepted nothing")
    return (f"{c['b']['label']} rejects {cio.pct(f['b']['reject_rate'])} of {f['n']:,} "
            f"{c['council']} items vs {c['a']['label']}'s {cio.pct(f['a']['reject_rate'])} "
            f"(leniency delta {f['leniency_pp']:+.1f} pp); {fa_text}.")


def reasons(rejected_on: dict[str, str]) -> str:
    return "; ".join(f"{lens}: {why}" for lens, why in rejected_on.items()) or "-"


def render_md(r: dict) -> str:
    c, f, cov = r["config"], r["final"], r["coverage"]
    a, b = c["a"]["label"], c["b"]["label"]
    out = [f"# Council calibration: {r['name']}", "", f"**{headline(r)}**", ""]
    if not cov["complete"]:
        out += [f"> **INCOMPLETE.** {a}: {cov['a_complete_items']:,}/{r['corpus']['n_items']:,} "
                f"items complete ({len(cov['a_pending_jobs'])} pending jobs); {b}: "
                f"{cov['b_complete_items']:,}/{r['corpus']['n_items']:,} "
                f"({len(cov['b_pending_jobs'])} pending jobs). Numbers below cover only the "
                f"{cov['both_complete_items']:,} items both panels finished.", ""]
    for w in r["warnings"]:
        out += [f"> **WARNING.** {w}", ""]
    out += ["Reproduce:", "", "```", r["command"], "```", "",
            "## Config", "", "| field | value |", "|---|---|",
            f"| council | {c['council']} (lenses: {', '.join(c['lenses'])}) |",
            f"| final verdict | ACCEPT at >= {c['min_agree']}/{len(c['lenses'])} lenses |",
            f"| reference A | {a} -- `{c['a']['dir']}` (stamp: "
            f"{'yes' if c['a']['stamp'] else 'none'}) |",
            f"| candidate B | {b} -- `{c['b']['dir']}` (stamp: "
            f"{'yes' if c['b']['stamp'] else 'none'}) |",
            f"| corpus | `{c['chunk_dir']}`: {r['corpus']['n_chunks']} chunks, "
            f"{r['corpus']['n_items']:,} items (sha256 per chunk in the JSON) |",
            "| seed | none: deterministic, no sampling |", "",
            f"## Final verdicts (>= {c['min_agree']}/{len(c['lenses'])})", "",
            f"| | {b} accept | {b} reject | total |", "|---|---:|---:|---:|"]
    cf = f["confusion"]
    out += [f"| **{a} accept** | {cf['a_accept_b_accept']:,} | {cf['a_accept_b_reject']:,} | "
            f"{cf['a_accept_b_accept'] + cf['a_accept_b_reject']:,} |",
            f"| **{a} reject** | {cf['a_reject_b_accept']:,} | {cf['a_reject_b_reject']:,} | "
            f"{cf['a_reject_b_accept'] + cf['a_reject_b_reject']:,} |",
            f"| total | {cf['a_accept_b_accept'] + cf['a_reject_b_accept']:,} | "
            f"{cf['a_accept_b_reject'] + cf['a_reject_b_reject']:,} | {f['n']:,} |", "",
            "| metric | value |", "|---|---|",
            f"| reject rate {a} | {cio.pct(f['a']['reject_rate'])} |",
            f"| reject rate {b} | {cio.pct(f['b']['reject_rate'])} |",
            f"| leniency delta (A reject - B reject) | "
            + ("n/a" if f["leniency_pp"] is None else f"{f['leniency_pp']:+.1f} pp") + " |",
            f"| agreement / Cohen's kappa | {cio.pct(f['agreement'])} / "
            f"{cio.fmt(f['cohen_kappa'])} |",
            f"| {b} false-accept vs {a} (B accepts that A rejects) | "
            f"{cio.interval(f['b_false_accept_vs_a'])} |",
            f"| {b} miss rate vs {a} (A rejects that B accepts) | "
            f"{cio.interval(f['b_miss_rate_vs_a'])} |",
            f"| Fleiss' kappa across lenses, {a} / {b} | "
            f"{cio.fmt(r['fleiss_kappa_across_lenses']['a'])} / "
            f"{cio.fmt(r['fleiss_kappa_across_lenses']['b'])} |", "",
            "## Per lens", "",
            f"| lens | n | {a} accept | {b} accept | leniency (pp) | Cohen's kappa | "
            f"only {a} rejects | only {b} rejects |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lens, s in r["lenses"].items():
        lp = "n/a" if s["leniency_pp"] is None else f"{s['leniency_pp']:+.1f}"
        out.append(f"| {lens} | {s['n']:,} | {cio.pct(s['a_accept_rate'])} | "
                   f"{cio.pct(s['b_accept_rate'])} | {lp} | {cio.fmt(s['cohen_kappa'])} | "
                   f"{s['counts']['b_only_accept']:,} | {s['counts']['a_only_accept']:,} |")
    out += ["", f"## {b} at other thresholds (vs {a} at >= {c['min_agree']})", "",
            "Calibration data only. A threshold picked from this table must be validated on a "
            "separate audit sample of the bulk run (`sample_audit.py`), not on these chunks.", "",
            f"| {b} min agree | {b} accept | agreement | kappa | {b} false-accept vs {a} | "
            f"{b} miss rate |", "|---:|---:|---:|---:|---|---|"]
    for s in r["b_threshold_sensitivity"]:
        out.append(f"| {s['b_min_agree']} | {cio.pct(s['b_accept_rate'])} | "
                   f"{cio.pct(s['agreement'])} | {cio.fmt(s['cohen_kappa'])} | "
                   f"{cio.interval(s['b_false_accept_vs_a'])} | "
                   f"{cio.interval(s['b_miss_rate_vs_a'])} |")
    for field, groups in r["by"].items():
        out += ["", f"## By `{field}`", "",
                f"| {field} | n | {a} reject | {b} reject | leniency (pp) | {b} miss rate |",
                "|---|---:|---:|---:|---:|---|"]
        for value, s in groups.items():
            lp = "n/a" if s["leniency_pp"] is None else f"{s['leniency_pp']:+.1f}"
            out.append(f"| {cio.md_cell(value, 60)} | {s['n']:,} | {cio.pct(s['a_reject_rate'])} "
                       f"| {cio.pct(s['b_reject_rate'])} | {lp} | "
                       f"{cio.interval(s['b_miss_rate_vs_a'])} |")
    dis = r["disagreements"]
    out += ["", f"## Final-verdict disagreements ({len(dis):,}; first "
                f"{min(len(dis), MD_DISAGREEMENTS)} shown, all in the JSON)", ""]
    if dis:
        out += [f"| sid | text | {a} | {b} | {a} rejected on | {b} rejected on |",
                "|---|---|---|---|---|---|"]
        for d in dis[:MD_DISAGREEMENTS]:
            out.append(f"| {cio.md_cell(d['sid'], 40)} | {cio.md_cell(d['text'], 70)} | "
                       f"{'accept' if d['a_accept'] else 'reject'} {d['a_votes']} | "
                       f"{'accept' if d['b_accept'] else 'reject'} {d['b_votes']} | "
                       f"{cio.md_cell(reasons(d['a_rejected_on']), 80)} | "
                       f"{cio.md_cell(reasons(d['b_rejected_on']), 80)} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.argv = argv
    try:
        cio.check_name(args.name)
        if args.min_agree not in range(1, len(cio.COUNCILS[args.council]) + 1):
            raise cio.CouncilDataError(f"--min-agree {args.min_agree} is not in 1..5")
        report = compare(args)
    except cio.CouncilDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    cov = report["coverage"]
    if not cov["complete"]:
        print(f"WARNING: INCOMPLETE -- A pending {len(cov['a_pending_jobs'])} jobs, "
              f"B pending {len(cov['b_pending_jobs'])}; comparing "
              f"{cov['both_complete_items']:,}/{report['corpus']['n_items']:,} items",
              file=sys.stderr)
        if args.require_complete:
            print("ERROR: --require-complete: nothing written", file=sys.stderr)
            return 2
    for w in report["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"council_calibration_{args.name}.json"
    out_md = args.results_dir / f"council_calibration_{args.name}.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(headline(report))
    print(f"wrote {cio.rel(out_json)} and {out_md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
