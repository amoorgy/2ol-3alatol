"""Build the Dataset B anchor bank: data/stimulus/v2_anchor_bank.tsv (tracked).

The words a scaled fusion run writes new sentences around, in PRIORITY order
so the first N rows are always a sensible run: the 284 train-side trap
anchors first (Arabic + gloss, MEANING rows), then bare respell-pool anchors
(NORMALIZATION rows only). The selection and ordering rules, and the evidence
for them, live in src/franco_eval/anchor_bank.py; this script only gathers
the inputs.

Inputs (no model calls, no RNG -- the order is a salted hash):
  data/stimulus/nilechat/{word_split,trap_votes}.tsv, stimuli_v1.csv,
  data/stimulus/training_a_votes.tsv          substitution policy + firewall
  data/generated/nilechat_wordfreq.tsv        respell pool, counts, bands
  data/training/v2/fused_sentences.jsonl      incidental-span measurement,
                                              pinned by sha256 (see below)

    python3 experiments/d5_cohere_track/training_data/build_anchor_bank.py
"""

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.anchor_bank import (BAND_LABELS, INCIDENTAL_MAX_RATE,
                                     band_of, bank_rows, family_ranks,
                                     incidental_spans, incidental_words,
                                     order_bare, select_bare, to_tsv,
                                     trap_anchors)
from franco_eval.substitution_policy import (REVIEW_ENGLISH, check,
                                             load_policy, needs_judge,
                                             variant_from_rule_path)

STIMULUS = ROOT / "data" / "stimulus"
SPLIT = STIMULUS / "nilechat" / "word_split.tsv"
TRAP_VOTES = STIMULUS / "nilechat" / "trap_votes.tsv"
STIMULI = STIMULUS / "nilechat" / "stimuli_v1.csv"
TRAINING_VOTES = STIMULUS / "training_a_votes.tsv"
FUSED = ROOT / "data" / "training" / "v2" / "fused_sentences.jsonl"
OUT = STIMULUS / "v2_anchor_bank.tsv"

# The incidental-span rates are measured on the finished v2 run (1,135 fused
# trap-anchor sentences). Pinned: if that file grows (a scaled run appends
# to it), the incidental set -- and with it every priority after row 284 --
# would silently move under whoever is resuming a run by anchor word.
# Re-measuring is a deliberate act: point --fused at the new file and
# update this hash in the same change.
FUSED_SHA256 = "f8abaf06164e3eef84764f36876e48b65e663162e13c9daae10d0c5d0d388187"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tsv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fused_pairs(path: pathlib.Path, expected_sha: str) -> list[tuple[str, str]]:
    """(text, anchor word) per fused sentence, after verifying the pin."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise SystemExit(
            f"{path} has sha256 {digest}, pinned {expected_sha}.\n"
            "The incidental-span measurement would change and move every bare "
            "anchor's priority. Re-measure deliberately: update FUSED_SHA256.")
    with path.open(encoding="utf-8") as handle:
        return [(r["text"], r["anchor"]["word"])
                for r in map(json.loads, handle) if r]


def respellable_words(pool: set[str], dataset_a, policy) -> set[str]:
    """Words with >= 1 mid-sentence variant the policy allows unjudged."""
    vocabulary = dataset_a.load_vocabulary()
    cache: dict = {}
    out = set()
    for word in sorted(pool):
        for form, _weight, meta in dataset_a.weighted_variants(word, vocabulary, cache):
            variant = variant_from_rule_path(form, meta["rule_path"])
            if (not check(policy, word, variant, sentence_final=False)
                    and not needs_judge(word, variant)):
                out.add(word)
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--fused", type=pathlib.Path, default=FUSED)
    parser.add_argument("--fused-sha256", default=FUSED_SHA256)
    parser.add_argument("--show", type=int, default=30,
                        help="bare anchors to print from the top of the bank")
    args = parser.parse_args()

    collect = _load("collect", HERE / "collect_training_sentences.py")
    dataset_a = _load("build_dataset_a", HERE / "build_dataset_a.py")
    table = collect.load_wordfreq(collect.WORDFREQ)
    counts = {w: c for w, (c, _d, _r) in table.items()}
    freq = {w: (c, d) for w, (c, d, _r) in table.items()}

    split = read_tsv(SPLIT)
    train_side = {r["word"] for r in split if r["side"] == "train"}
    heldout = {r["word"] for r in split if r["side"] == "heldout"}
    votes = read_tsv(TRAP_VOTES)
    trap_words = {w for r in votes for w in (r["word_type"], r["trap"])}
    policy = load_policy(split=SPLIT, stimuli=STIMULI, trap_votes=TRAP_VOTES,
                         training_votes=TRAINING_VOTES)

    traps = trap_anchors(votes, train_side)
    gated = collect.build_pool(table)
    pool = gated - collect.firewall(SPLIT)[0]

    pairs = fused_pairs(args.fused, args.fused_sha256)
    span_counts, n_fused = incidental_spans(pairs)
    incidental = incidental_words(span_counts, n_fused)
    respellable = respellable_words(pool, dataset_a, policy)

    exclusions = [
        ("held_out", heldout),
        ("protected_form", policy.protected_forms),
        ("trap_word", trap_words),
        ("english", REVIEW_ENGLISH | collect._english_stoplist()),
        ("not_respellable", pool - respellable),
        ("incidental", incidental),
    ]
    kept, reasons = select_bare(pool, exclusions)

    # families over every gated word AND every firewalled form, so a variant
    # of an excluded head (3shan <- 3ashan, ally <- elly) is deferred too
    universe = gated | {w for w in heldout | policy.protected_forms | trap_words
                        if w in counts}
    ranks = family_ranks({w: counts[w] for w in universe})
    bare = order_bare(kept, counts, ranks)
    rows = bank_rows(traps, bare, freq)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_tsv(rows), encoding="utf-8")
    report(rows, traps, pool, reasons, ranks, span_counts, n_fused, incidental,
           args)
    return 0


def report(rows, traps, pool, reasons, ranks, span_counts, n_fused,
           incidental, args) -> None:
    bare = [r for r in rows if r["source"] == "bare"]
    print(f"trap anchors: {len(traps)} distinct words (rows 1-{len(traps)})")
    print(f"respell pool: {len(pool):,} words; excluded (first reason wins):")
    for reason, n in reasons.items():
        print(f"  {reason:16s} {n:6,}")
    print(f"incidental: {len(incidental)} words at >= {INCIDENTAL_MAX_RATE:.0%} "
          f"of {n_fused:,} fused sentences; in pool, top: "
          + " ".join(f"{w}({span_counts[w] / n_fused:.0%})" for w in sorted(
              incidental & pool, key=lambda w: -span_counts[w])[:15]))
    per_band = collections.Counter(r["band"] for r in bare)
    heads = collections.Counter(r["band"] for r in bare if ranks.get(r["word"], 0) == 0)
    print(f"bare anchors: {len(bare):,} (rows {len(traps) + 1}-{len(rows)}); "
          f"family heads (tier 0): {sum(heads.values()):,}")
    for band in BAND_LABELS:
        print(f"  {band:8s} {per_band[band]:6,}  tier0 {heads[band]:6,}")
    print(f"trap-anchor bands: {dict(collections.Counter(band_of(r['count']) for r in rows[:len(traps)]))}")
    print(f"top {args.show} bare anchors: "
          + " ".join(r["word"] for r in bare[:args.show]))
    print(f"wrote {args.out} ({len(rows):,} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
