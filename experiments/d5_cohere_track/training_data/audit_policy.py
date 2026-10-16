"""Measure the v2 substitution policy against the committed a_v1 verdicts.

v2 respells ~158k sentences with no council per row, so its error rate has
to be ESTIMATED. The a_v1 rows the council fully judged (n_judges >= 4),
with their final outcome after the manual review, are a labelled set: apply
the v2 policy to each row and ask

  catch rate    of rows finally rejected, the share the policy would block
  false block   of rows finally kept, the share the policy would block
  residual      of rows the policy lets through, the share that were rejected
                -- the estimate for v2's unjudged rows

Two policy variants, because the full policy is scored partly in-sample:
its `bad_pairs` come from these same votes. `rules_only` drops them (and the
held-out/trap blocks, which do not bear on a_v1), leaving only the rulings
as code -- the honest estimate for substitution pairs no judge has seen.

    .venv/bin/python experiments/d5_cohere_track/training_data/audit_policy.py
"""

import csv
import dataclasses
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import tokens
from franco_eval.substitution_policy import (Policy, check, load_policy,
                                             variant_from_rule_path)

STIM = ROOT / "data" / "stimulus"
FAT = ROOT / "data" / "generated" / "training_a_full.jsonl"
KEPT = [ROOT / "data" / "training" / "a_v1" / f"{s}.jsonl" for s in ("train", "val")]
OUT = ROOT / "experiments" / "d5_cohere_track" / "results" / "substitution_policy_audit"


def judged_rows() -> list[tuple[dict, bool]]:
    """(fat row, finally_rejected) for every fully judged permutation."""
    with (STIM / "training_a_votes.tsv").open(encoding="utf-8") as handle:
        votes = {r["rendering_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    kept = {json.loads(line)["meta"]["rendering_id"]
            for path in KEPT for line in path.open(encoding="utf-8")}
    out = []
    with FAT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            vote = votes.get(row["rendering_id"])
            if (row["role"] != "permutation" or not row["substitutions"]
                    or vote is None or int(vote.get("n_judges", 0)) < 4):
                continue
            out.append((row, row["rendering_id"] not in kept))
    return out


def blocked(policy: Policy, row: dict) -> bool:
    last = max((o for _w, o, _c in tokens(row["text"])), default=-1)
    return any(check(policy, s["word"],
                     variant_from_rule_path(s["rendering"], s["rule_path"]),
                     sentence_final=s["start"] == last)
               for s in row["substitutions"])


def score(policy: Policy, rows: list[tuple[dict, bool]]) -> dict:
    rejected = [r for r, bad in rows if bad]
    kept = [r for r, bad in rows if not bad]
    caught = sum(blocked(policy, r) for r in rejected)
    false_blocks = sum(blocked(policy, r) for r in kept)
    passed_bad = len(rejected) - caught
    passed = passed_bad + len(kept) - false_blocks
    return {"judged": len(rows), "rejected": len(rejected), "kept": len(kept),
            "catch_rate": round(caught / max(len(rejected), 1), 4),
            "false_block_rate": round(false_blocks / max(len(kept), 1), 4),
            "residual_error": round(passed_bad / max(passed, 1), 4),
            "rows_passing": passed}


def main() -> int:
    full = load_policy(split=STIM / "nilechat" / "word_split.tsv",
                       stimuli=STIM / "nilechat" / "stimuli_v1.csv",
                       trap_votes=STIM / "nilechat" / "trap_votes.tsv",
                       training_votes=STIM / "training_a_votes.tsv")
    rules_only = dataclasses.replace(full, bad_pairs=frozenset(),
                                     protected_forms=frozenset(), trap_partners={})
    rows = judged_rows()
    report = {"full_policy_in_sample": score(full, rows),
              "rules_only_out_of_sample": score(rules_only, rows),
              "baseline_no_policy_error": round(
                  sum(bad for _r, bad in rows) / max(len(rows), 1), 4)}
    OUT.with_suffix(".json").write_text(json.dumps(report, indent=1),
                                        encoding="utf-8")
    lines = ["# v2 substitution policy vs. committed a_v1 verdicts", "",
             f"{len(rows):,} fully judged a_v1 permutation rows; baseline error "
             f"with no policy {report['baseline_no_policy_error']:.1%}.", "",
             "| policy | catch rate | false block | residual error | rows passing |",
             "|---|---|---|---|---|"]
    for name in ("rules_only_out_of_sample", "full_policy_in_sample"):
        s = report[name]
        lines.append(f"| {name} | {s['catch_rate']:.1%} | {s['false_block_rate']:.1%}"
                     f" | {s['residual_error']:.1%} | {s['rows_passing']:,} |")
    lines += ["", "`rules_only` is the honest estimate for unseen substitution "
              "pairs; `full` is in-sample (its bad_pairs come from these votes)."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
