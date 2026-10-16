"""Freeze a Dataset B fusion plan from the v2 anchor and scenario banks.

The first N distinct anchors of data/stimulus/v2_anchor_bank.tsv (bank
order = priority order; a repeated word is planned once, first row wins),
each in K distinct ACCEPTED scenarios of data/stimulus/scenarios_v2.tsv,
drawn with a per-anchor rng seeded from (--seed, word), minus every job an
earlier audit already holds (--exclude-audit, repeatable; the finished v2
run is data/training/v2/fused_audit.jsonl).

    python3 experiments/d5_cohere_track/training_data/plan_fusion_v2.py \
        --anchors 5000 --scenarios-per-anchor 4 \
        --exclude-audit data/training/v2/fused_audit.jsonl \
        --out data/generated/v3/fusion_plan.jsonl

Writes, and never overwrites (a plan is frozen once a run starts on it):
  <out>              one job per line, canonical bytes (sorted keys)
  <out>.sha256       `shasum -a 256 -c` format; build_dataset_b.py --plan
                     refuses a plan whose bytes no longer match it
  <stem>.meta.json   config, seed, source files + sha256, counts, command

job_id = f"{word}#{scenario_id}". Ids 0-15 of the scenario bank are
fusion.SCENARIOS verbatim, so an old audit's (anchor, scenario_index) is the
same job id -- that is what makes --exclude-audit exact.

Stdlib only (plus franco_eval.scenario_bank, the bank's own loader).
"""

import argparse
import hashlib
import json
import pathlib
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.fusion_jobs import (completed_job_ids, draw_v2_jobs,
                                     read_anchor_bank, sidecar_path, write_plan)

ANCHOR_BANK = ROOT / "data" / "stimulus" / "v2_anchor_bank.tsv"
SCENARIO_BANK = ROOT / "data" / "stimulus" / "scenarios_v2.tsv"
# the pilot's SCENARIO_SEED, fixed a priori -- never tuned on any output
SEED = 29


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: pathlib.Path) -> str:
    """Repo-relative when inside the repo: the meta file must not leak
    one machine's absolute layout into a committed artifact."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def meta_path(plan: pathlib.Path) -> pathlib.Path:
    return plan.with_name(plan.stem + ".meta.json")


def load_accepted_scenarios(path: pathlib.Path) -> list[tuple[int, str]]:
    from franco_eval.scenario_bank import load_scenarios  # the bank's contract
    return load_scenarios(path, accepted_only=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--anchors", type=int, required=True,
                        help="first N distinct anchors of the bank (0 = all)")
    parser.add_argument("--scenarios-per-anchor", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-audit", type=pathlib.Path, action="append",
                        default=[], help="skip jobs already in this audit (repeatable)")
    parser.add_argument("--anchor-bank", type=pathlib.Path, default=ANCHOR_BANK)
    parser.add_argument("--scenario-bank", type=pathlib.Path, default=SCENARIO_BANK)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="plan JSONL to create (refused if it exists)")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.anchors < 0:
        print("--anchors must be >= 0 (0 = the whole bank)")
        return 2
    outputs = [args.out, sidecar_path(args.out), meta_path(args.out)]
    existing = [p for p in outputs if p.exists()]
    if existing:
        print(f"refusing to overwrite a frozen plan: {', '.join(map(str, existing))}")
        return 2
    missing = [p for p in args.exclude_audit if not p.exists()]
    if missing:
        print(f"--exclude-audit not found: {', '.join(map(str, missing))}")
        return 2
    try:
        rows = read_anchor_bank(args.anchor_bank)
        scenarios = load_accepted_scenarios(args.scenario_bank)
        excluded_by = [(path, completed_job_ids(path)) for path in args.exclude_audit]
        exclude = frozenset().union(*(ids for _p, ids in excluded_by))
        draw = draw_v2_jobs(rows, scenarios, n_anchors=args.anchors,
                            k=args.scenarios_per_anchor, seed=args.seed,
                            exclude=exclude)
    except (OSError, ValueError) as exc:  # PlanError is a ValueError
        print(f"cannot plan: {exc}")
        return 2
    if not draw.jobs:
        print(f"nothing to plan: {draw.stats}")
        return 2

    sha = write_plan(args.out, draw.jobs)
    meta = {
        "plan": shown(args.out), "sha256": sha, "jobs": len(draw.jobs),
        "anchors": draw.stats["anchors_used"],
        "scenarios_per_anchor": args.scenarios_per_anchor, "seed": args.seed,
        "rng": "random.Random(f'{seed}:{word}').sample(sorted accepted ids, k)",
        "anchor_bank": {"path": shown(args.anchor_bank),
                        "sha256": sha256_of(args.anchor_bank), "rows": len(rows)},
        "scenario_bank": {"path": shown(args.scenario_bank),
                          "sha256": sha256_of(args.scenario_bank),
                          "accepted": len(scenarios)},
        "exclude_audits": [{"path": shown(path), "sha256": sha256_of(path),
                            "jobs": len(ids)} for path, ids in excluded_by],
        "stats": draw.stats,
        "command": shlex.join(["python3", shown(pathlib.Path(__file__)), *argv]),
    }
    meta_path(args.out).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    stats = draw.stats
    print(f"plan  {args.out}\nsha256 {sha}\n"
          f"anchors {stats['anchors_used']} (requested {args.anchors or 'all'}, "
          f"{stats['duplicate_rows_skipped']} duplicate rows skipped) x "
          f"{args.scenarios_per_anchor} of {stats['scenarios_accepted']} accepted "
          f"scenarios, seed {args.seed}\n"
          f"jobs  {stats['jobs_drawn']} drawn - {stats['jobs_excluded']} already done "
          f"= {stats['jobs_planned']} planned")
    if stats["conflicting_duplicates"]:
        print(f"warning: repeated words with different arabic/gloss (first row kept): "
              f"{stats['conflicting_duplicates'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
