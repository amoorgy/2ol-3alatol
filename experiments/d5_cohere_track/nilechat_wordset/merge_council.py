"""Merge council pass 2/3 judgement files into committed votes TSVs.

Deterministic: reads the per-judge JSON files the council agents wrote,
computes majorities and Fleiss' kappa, and writes
data/stimulus/nilechat/{variant_votes,trap_votes}.tsv. Downstream numbers
regenerate from these files, never from a live model -- that is what keeps an
LLM council compatible with hard rule 2.

    python3 experiments/d5_cohere_track/nilechat_wordset/merge_council.py

Generic mode, for any council using the Dataset-A judge-row contract
({sid, rid, v, why}, files <lens>_c<N>.json) -- e.g. a Sonnet bulk run or an
Opus audit panel from training_data/council/:

    python3 experiments/d5_cohere_track/nilechat_wordset/merge_council.py \\
        --dir data/generated/v2/council/fused_votes_sonnet --out <votes.tsv>

It refuses (exit 2, nothing written) unless the dir holds exactly
--expect-lenses lenses and every item has a vote from each of them: an item
missing a lens could otherwise pass 4/4 without ever facing the fifth.
--allow-incomplete overrides. It does not touch council_summary.json.
"""

import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "stimulus" / "nilechat"


def fleiss_kappa(rows: list[list[str]], categories: list[str]) -> float | None:
    """Fleiss' kappa over items x raters (categorical)."""
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return None
    n_raters = min(len(r) for r in rows)
    rows = [r[:n_raters] for r in rows]
    counts = [[r.count(c) for c in categories] for r in rows]
    n_items = len(rows)
    p_j = [sum(row[j] for row in counts) / (n_items * n_raters)
           for j in range(len(categories))]
    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in counts]
    p_bar = sum(p_i) / n_items
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return 1.0
    return round((p_bar - p_e) / (1 - p_e), 4)


def load_judgements(directory: pathlib.Path) -> dict[str, dict[tuple, dict]]:
    """lens -> {(key fields) -> judgement}; reports malformed files loudly."""
    out: dict[str, dict[tuple, dict]] = collections.defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        lens = path.stem.rsplit("_c", 1)[0]
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"MALFORMED {path}: {exc}")
            continue
        for row in rows:
            key = (row["w"], row.get("r") or row.get("t"))
            out[lens][key] = row
    return out


def merge_variants(directory: pathlib.Path, out: pathlib.Path) -> dict:
    judges = load_judgements(directory)
    lenses = sorted(judges)
    keys = sorted({k for lens in lenses for k in judges[lens]})
    rows, kappa_rows = [], []
    for key in keys:
        votes = [judges[lens].get(key) for lens in lenses]
        votes = [v for v in votes if v is not None]
        valid_votes = ["valid" if v.get("v") else "invalid" for v in votes]
        kappa_rows.append(valid_votes)
        n_invalid = valid_votes.count("invalid")
        majority_valid = valid_votes.count("valid") > n_invalid
        collisions = [v.get("c", "") for v in votes if v.get("c")]
        reasons = [v.get("why", "") for v in votes if not v.get("v") and v.get("why")]
        rows.append({
            "word_type": key[0], "rendering": key[1],
            "valid": "true" if majority_valid else "false",
            "votes_valid": valid_votes.count("valid"),
            "votes_invalid": n_invalid,
            "n_judges": len(votes),
            "collision": collections.Counter(collisions).most_common(1)[0][0]
            if collisions else "",
            "collision_count": "",
            "reason": " || ".join(reasons)[:300],
        })
    kappa = fleiss_kappa(kappa_rows, ["valid", "invalid"])
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    n_invalid = sum(1 for r in rows if r["valid"] == "false")
    return {"n": len(rows), "invalid": n_invalid, "fleiss_kappa": kappa,
            "lenses": lenses}


def merge_traps(directory: pathlib.Path, out: pathlib.Path,
                min_agree: int) -> dict:
    judges = load_judgements(directory)
    lenses = sorted(judges)
    keys = sorted({k for lens in lenses for k in judges[lens]})
    rows, kappa_rows = [], []
    for key in keys:
        votes = [judges[lens].get(key) for lens in lenses]
        votes = [v for v in votes if v is not None]
        verdicts = [v.get("verdict", "UNSURE") for v in votes]
        kappa_rows.append(verdicts)
        n_trap = verdicts.count("TRAP")
        arabics_w = collections.Counter(v.get("ar_w", "") for v in votes
                                        if v.get("ar_w"))
        arabics_t = collections.Counter(v.get("ar_t", "") for v in votes
                                        if v.get("ar_t"))
        rows.append({
            "word_type": key[0], "trap": key[1],
            "valid": "true" if n_trap >= min_agree else "false",
            "votes_trap": n_trap,
            "votes_same": verdicts.count("SAME"),
            "votes_unsure": verdicts.count("UNSURE"),
            "n_judges": len(votes),
            "arabic_word": arabics_w.most_common(1)[0][0] if arabics_w else "",
            "arabic_trap": arabics_t.most_common(1)[0][0] if arabics_t else "",
            "meaning_en_word": next((v.get("gl_w", "") for v in votes
                                if v.get("gl_w")), ""),
            "meaning_en_trap": next((v.get("gl_t", "") for v in votes
                                if v.get("gl_t")), ""),
            "reason": f"{n_trap}/{len(votes)} judges TRAP",
            "reviewer": "",
        })
    kappa = fleiss_kappa(kappa_rows, ["TRAP", "SAME", "UNSURE"])
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    n_valid = sum(1 for r in rows if r["valid"] == "true")
    return {"n": len(rows), "confirmed_traps": n_valid,
            "fleiss_kappa": kappa, "lenses": lenses, "min_agree": min_agree}


def load_training_judges(directory: pathlib.Path) -> dict[str, dict[tuple, dict]]:
    """lens -> {(sid, rid) -> judge row}; reports malformed files loudly."""
    judges: dict[str, dict[tuple, dict]] = collections.defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        lens = path.stem.rsplit("_c", 1)[0]
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"MALFORMED {path}: {exc}")
            continue
        for row in rows:
            judges[lens][(row["sid"], row["rid"])] = row
    return judges


def merge_training(directory: pathlib.Path, out: pathlib.Path,
                   min_agree: int = 4) -> dict:
    """Merge the Dataset-A verification council (pass id: training_a).

    Judge rows are keyed (sid, rid) and carry v (bool), bad (list of
    "word->rendering" strings naming the offending substitution) and why.
    ACCEPT requires >= min_agree of 5 judges valid -- training data wants
    precision, so anything weaker goes to manual review instead of train.
    """
    judges = load_training_judges(directory)
    lenses = sorted(judges)
    keys = sorted({k for lens in lenses for k in judges[lens]})
    rows_out, kappa_rows = [], []
    for key in keys:
        votes = [judges[lens].get(key) for lens in lenses]
        votes = [v for v in votes if v is not None]
        verdicts = ["valid" if v.get("v") else "invalid" for v in votes]
        kappa_rows.append(verdicts)
        n_valid = verdicts.count("valid")
        bad = collections.Counter(
            b for v in votes for b in (v.get("bad") or []))
        reasons = [v.get("why", "") for v in votes
                   if not v.get("v") and v.get("why")]
        rows_out.append({
            "sentence_id": key[0], "rendering_id": key[1],
            "valid": "true" if n_valid >= min_agree else "false",
            "votes_valid": n_valid,
            "votes_invalid": len(votes) - n_valid,
            "n_judges": len(votes),
            "bad_substitutions": " ".join(f"{b}:{n}" for b, n
                                          in bad.most_common(4)),
            "reason": " || ".join(reasons)[:300],
        })
    kappa = fleiss_kappa(kappa_rows, ["valid", "invalid"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)
    n_rejected = sum(1 for r in rows_out if r["valid"] == "false")
    return {"n": len(rows_out), "rejected": n_rejected,
            "fleiss_kappa": kappa, "lenses": lenses, "min_agree": min_agree}


def merge_generic(directory: pathlib.Path, out: pathlib.Path, min_agree: int,
                  expect_lenses: int, allow_incomplete: bool) -> tuple[int, str]:
    """(exit code, message) for --dir/--out; writes nothing when it refuses."""
    if not directory.is_dir() or not any(directory.glob("*.json")):
        return 2, f"ERROR: no verdict files (*.json) in {directory}"
    judges = load_training_judges(directory)
    lenses = sorted(judges)
    keys = {k for lens in lenses for k in judges[lens]}
    short = sorted(k for k in keys if sum(k in judges[lens] for lens in lenses) < len(lenses))
    problems = []
    non_bool = sum(not isinstance(row.get("v"), bool)
                   for lens in lenses for row in judges[lens].values())
    if non_bool:
        # merge_training reads v by truthiness, so "false" would count as valid
        problems.append(f"{non_bool} rows have a non-boolean v")
    if len(lenses) != expect_lenses:
        problems.append(f"found {len(lenses)} lenses {lenses}, expected {expect_lenses}")
    if short:
        problems.append(f"{len(short)} items lack a vote from some lens (e.g. {short[0]})")
    if problems and not allow_incomplete:
        return 2, ("ERROR: incomplete council in " + str(directory) + ": "
                   + "; ".join(problems)
                   + ". Finish it (training_data/council/council_jobs.py) or pass "
                     "--allow-incomplete. Nothing written.")
    result = merge_training(directory, out, min_agree)
    message = (f"{directory.name}: {result['n']} judged by {result['lenses']}, "
               f"{result['rejected']} rejected at <{min_agree}/{len(lenses)}, "
               f"kappa={result['fleiss_kappa']} -> {out}")
    if problems:
        message += "\nWARNING (--allow-incomplete): " + "; ".join(problems)
    return 0, message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p2-dir", type=pathlib.Path,
                        default=pathlib.Path("/tmp/council_p2"))
    parser.add_argument("--p3-dir", type=pathlib.Path,
                        default=pathlib.Path("/tmp/council_p3"))
    parser.add_argument("--min-agree", type=int, default=4,
                        help="judges (of 5) required to confirm a trap")
    parser.add_argument("--training", action="store_true",
                        help="merge the Dataset-A verification council "
                             "(/tmp/council_ta) instead of passes 2+3")
    parser.add_argument("--ta-dir", type=pathlib.Path,
                        default=pathlib.Path("/tmp/council_ta"))
    parser.add_argument("--fusion-v2", action="store_true",
                        help="merge the v2 fused-sentence council "
                             "(/tmp/council_fv2) into fusion_v2_votes.tsv")
    parser.add_argument("--fv2-dir", type=pathlib.Path,
                        default=ROOT / "data" / "generated" / "v2" / "council"
                        / "fused_votes")
    parser.add_argument("--pairs-v2", action="store_true",
                        help="merge the v2 substitution-pair council into "
                             "pairs_v2_votes.tsv (sentence_id = 'word->form')")
    parser.add_argument("--pv2-dir", type=pathlib.Path,
                        default=ROOT / "data" / "generated" / "v2" / "council"
                        / "pair_votes")
    parser.add_argument("--dir", type=pathlib.Path,
                        help="generic mode: merge this verdict dir (needs --out)")
    parser.add_argument("--out", type=pathlib.Path,
                        help="generic mode: the votes TSV to write")
    parser.add_argument("--expect-lenses", type=int, default=5,
                        help="generic mode: lenses the council must have")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="generic mode: merge even if lenses or votes are missing")
    args = parser.parse_args(argv)

    if (args.dir is None) != (args.out is None):
        parser.error("--dir and --out go together")
    if args.dir is not None:
        code, message = merge_generic(args.dir, args.out, args.min_agree,
                                      args.expect_lenses, args.allow_incomplete)
        print(message, file=sys.stderr if code else sys.stdout)
        return code

    if args.pairs_v2:
        # one verdict per distinct substitution pair; sid == rid == "word->form"
        result = merge_training(args.pv2_dir,
                                ROOT / "data" / "stimulus" / "pairs_v2_votes.tsv",
                                args.min_agree)
        print(f"pairs_v2: {result['n']} judged by {result['lenses']}, "
              f"{result['rejected']} rejected at <{args.min_agree}/5, "
              f"kappa={result['fleiss_kappa']}")
        summary = OUT_DIR / "council_summary.json"
        existing = json.loads(summary.read_text()) if summary.exists() else {}
        existing["pairs_v2"] = result
        summary.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        return 0

    if args.fusion_v2:
        # same judge-row contract as Dataset A (sid, rid, v, why); one fused
        # sentence per key, so rid == sid
        result = merge_training(args.fv2_dir,
                                ROOT / "data" / "stimulus" / "fusion_v2_votes.tsv",
                                args.min_agree)
        print(f"fusion_v2: {result['n']} judged by {result['lenses']}, "
              f"{result['rejected']} rejected at <{args.min_agree}/5, "
              f"kappa={result['fleiss_kappa']}")
        summary = OUT_DIR / "council_summary.json"
        existing = json.loads(summary.read_text()) if summary.exists() else {}
        existing["fusion_v2"] = result
        summary.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        return 0

    if args.training:
        result = merge_training(args.ta_dir,
                                ROOT / "data" / "stimulus"
                                / "training_a_votes.tsv",
                                args.min_agree)
        print(f"training_a: {result['n']} judged by {result['lenses']}, "
              f"{result['rejected']} rejected at <{args.min_agree}/5, "
              f"kappa={result['fleiss_kappa']}")
        summary = OUT_DIR / "council_summary.json"
        existing = json.loads(summary.read_text()) if summary.exists() else {}
        existing["training_a"] = result
        summary.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        return 0

    v = merge_variants(args.p2_dir, OUT_DIR / "variant_votes.tsv")
    print(f"variants: {v['n']} judged by {v['lenses']}, "
          f"{v['invalid']} invalid, kappa={v['fleiss_kappa']}")
    t = merge_traps(args.p3_dir, OUT_DIR / "trap_votes.tsv", args.min_agree)
    print(f"traps: {t['n']} judged by {t['lenses']}, "
          f"{t['confirmed_traps']} confirmed at >={args.min_agree}/5, "
          f"kappa={t['fleiss_kappa']}")
    (OUT_DIR / "council_summary.json").write_text(
        json.dumps({"pass2_variants": v, "pass3_traps": t}, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
