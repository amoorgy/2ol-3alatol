"""Print the Workflow args for the council jobs still pending in a verdict dir.

A job is one (chunk, lens) judge. It is pending when
<out-dir>/<lens>_<chunk>.json is missing, unparsable, malformed (a row without
string sid/rid, or a non-boolean v), holds contradicting duplicate verdicts,
or does not cover every item of its chunk (fewer rows, or rows for other
items). A council that dies at a session limit therefore resumes by re-running
this and passing the printed JSON as the args of council/fused_council.js or
council/pair_council.js. Complete files are never re-queued.

    python3 experiments/d5_cohere_track/training_data/council/council_jobs.py \\
        --council fused --chunk-dir data/generated/v2/council/fused_chunks \\
        --out-dir data/generated/v2/council/fused_votes_sonnet \\
        --model opus --effort low --stamp

stdout is exactly the args JSON: {council, model, effort, chunkDir, outDir,
sizes, jobs}. Diagnostics go to stderr. --stamp records the panel (council,
model, effort, chunk dir) in <out-dir>/panel.meta; once a dir is stamped, a
run naming another model or effort is refused, so one verdict dir never mixes
two panels, and a resume without --model/--effort takes them from the stamp.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import council_io as cio  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--council", required=True, choices=sorted(cio.COUNCILS))
    parser.add_argument("--chunk-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path,
                        help="the verdict dir the judges write to")
    parser.add_argument("--lenses", help="comma-separated subset (default: all five)")
    parser.add_argument("--chunks", help="comma-separated chunk subset, e.g. c0,c1")
    parser.add_argument("--model", help="sonnet for bulk fleets, opus for calibration/audits")
    parser.add_argument("--effort", help="low for bulk fleets (opus), xhigh for calibration/audits (opus)")
    parser.add_argument("--stamp", action="store_true",
                        help="write <out-dir>/panel.meta if absent (needs --model/--effort)")
    return parser.parse_args(argv)


def settle_panel(args: argparse.Namespace, chunk_dir: pathlib.Path,
                 out_dir: pathlib.Path) -> tuple[str | None, str | None]:
    """Reconcile --model/--effort with the stamp; raise on any disagreement."""
    stamp = cio.read_stamp(out_dir)
    model, effort = args.model, args.effort
    if stamp:
        mismatches = [f"{field} {given!r} != stamped {stamp.get(field)!r}"
                      for field, given in (("council", args.council), ("model", model),
                                           ("effort", effort))
                      if given is not None and given != stamp.get(field)]
        stamped_dir = stamp.get("chunk_dir")
        if stamped_dir and (cio.ROOT / stamped_dir).resolve() != chunk_dir:
            mismatches.append(f"chunk dir {cio.rel(chunk_dir)} != stamped {stamped_dir}")
        if mismatches:
            raise cio.CouncilDataError(
                f"{out_dir / cio.STAMP} belongs to another panel: " + "; ".join(mismatches)
                + ". Use a fresh --out-dir for a different panel.")
        model, effort = model or stamp.get("model"), effort or stamp.get("effort")
    elif args.stamp:
        if not (model and effort):
            raise cio.CouncilDataError("--stamp needs --model and --effort")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / cio.STAMP).write_text(json.dumps({
            "council": args.council, "model": model, "effort": effort,
            "chunk_dir": cio.rel(chunk_dir)}, indent=1) + "\n", encoding="utf-8")
        print(f"stamped {out_dir / cio.STAMP}: {args.council} {model}/{effort}",
              file=sys.stderr)
    return model, effort


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lenses = cio.resolve_lenses(args.council, args.lenses)
        chunk_dir, out_dir = args.chunk_dir.resolve(), args.out_dir.resolve()
        all_chunks = cio.load_chunks(chunk_dir)
        chunks = cio.select_chunks(all_chunks, args.chunks)
        model, effort = settle_panel(args, chunk_dir, out_dir)
    except cio.CouncilDataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pending, strays = cio.pending_jobs(chunks, out_dir, lenses)
    total = len(chunks) * len(lenses)
    reasons: dict[str, int] = {}
    for job in pending:
        kind = job["reason"].split(":")[0].split(" (")[0]
        reasons[kind] = reasons.get(kind, 0) + 1
    breakdown = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
    print(f"{args.council}: {len(pending)}/{total} jobs pending"
          + (f" ({breakdown})" if breakdown else " -- council complete")
          + f" in {cio.rel(out_dir)}", file=sys.stderr)
    for job in pending:
        if job["reason"] != "missing":
            print(f"  {job['lens']}_{job['chunk']}.json: {job['reason']}", file=sys.stderr)
    if strays:
        print(f"WARNING: {strays} verdict rows name items that are not in their chunk "
              f"(ignored; check the chunk dir was not re-chunked under this council)",
              file=sys.stderr)
    if not (model and effort):
        print("WARNING: no --model/--effort and no stamp; the workflow will refuse "
              "these args", file=sys.stderr)
    elif (model, effort) not in cio.POLICY_PANELS:
        print(f"WARNING: {model}/{effort} is not a policy panel "
              f"({', '.join('/'.join(p) for p in cio.POLICY_PANELS)})", file=sys.stderr)

    workflow_args = {
        "council": args.council, "model": model, "effort": effort,
        "chunkDir": str(chunk_dir), "outDir": str(out_dir),
        "sizes": {c: len(items) for c, items in all_chunks.items()},
        "jobs": [{"chunk": j["chunk"], "lens": j["lens"]} for j in pending],
    }
    print(json.dumps(workflow_args, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
