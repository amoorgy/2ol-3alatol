"""Assemble the NileChat stimulus file: base + licensed variants (+ traps).

Reads the 500-word candidate set and the corpus frequency table, generates
licensed respellings of each word, embeds every rendering in the word's carrier
sentence, and writes one committed CSV. Traps and council verdicts are merged
in when their files exist, so the same command regenerates the stimulus file at
every stage of the pipeline:

    .venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_variants.py

Everything is deterministic with no seed: variant enumeration is exhaustive and
ordered, and selection is a stratified top-k over phenomenon groups.

Provenance: the carrier sentences are machine-derived (Command R+), so under
hard rule 1 this file is PROBE STIMULUS, never a benchmark artifact. The
`provenance` column says so on every row.
"""

import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.licensing import SELF_LICENSED, BY_ID, enumerate_variants

CANDIDATES = ROOT / "data" / "generated" / "nilechat_candidates.json"
WORDFREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"
TRAP_VOTES = ROOT / "data" / "stimulus" / "nilechat" / "trap_votes.tsv"
VARIANT_VOTES = ROOT / "data" / "stimulus" / "nilechat" / "variant_votes.tsv"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

FIELDS = [
    "rendering_id", "item_id", "word_type", "rendering", "role",
    "licence", "corpus_count", "collision", "collision_count",
    "rule_path", "phenomena", "ortho_classes", "n_edits", "score",
    "trap_word", "trap_reason",
    "sentence", "target_start", "target_end",
    "provenance", "source_doc_id",
    "council_valid", "council_reason",
]

ATTEST_MIN = 20


def load_vocabulary(path: pathlib.Path) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            word, count, _rest = line.split("\t", 2)
            vocabulary[word] = int(count)
    return vocabulary


def licence_tier(variant, vocabulary: dict[str, int]) -> tuple[str, int]:
    """attested > rule_safe > self_licensed.

    "Attested" means attested in Command R+ output, not in human writing; the
    human-authored arm remains the check on any claim built from this.
    """
    count = vocabulary.get(variant.form, 0)
    if count >= ATTEST_MIN:
        return "attested", count
    if any(BY_ID[op_id].direction == SELF_LICENSED for op_id, _at in variant.path):
        return "self_licensed", count
    return "rule_safe", count


def pick_variants(variants, vocabulary, per_word: int):
    """Stratified top-k: one variant from each distinct phenomenon signature.

    Each rendering of a word then exercises a DIFFERENT phenomenon, which is
    worth more to E2 than five draws from the same class. Order inside a group
    is (attested first, then score, then form) -- deterministic, no RNG.
    """
    tier_rank = {"attested": 0, "rule_safe": 1, "self_licensed": 2}
    annotated = []
    for variant in variants:
        tier, count = licence_tier(variant, vocabulary)
        annotated.append((tier_rank[tier], -variant.score, variant.form,
                          tier, count, variant))
    annotated.sort()

    chosen, seen_signatures = [], set()
    for rank, _score, _form, tier, count, variant in annotated:
        if len(chosen) >= per_word:
            break
        signature = variant.classes
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen.append((tier, count, variant))
    for rank, _score, _form, tier, count, variant in annotated:
        if len(chosen) >= per_word:
            break
        if any(v is variant for _t, _c, v in chosen):
            continue
        chosen.append((tier, count, variant))
    return chosen


def render(sentence: str, start: int, end: int, rendering: str) -> tuple[str, int, int]:
    """Substitute the rendering at the target span; spans shift with length."""
    return (sentence[:start] + rendering + sentence[end:], start, start + len(rendering))


def load_votes(path: pathlib.Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {tuple(row[k] for k in key_fields): row for row in reader}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants-per-word", type=int, default=4,
                        help="4 variants + 1 base (+ 1 trap) = the star design")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--candidates", type=pathlib.Path, default=CANDIDATES)
    parser.add_argument("--wordfreq", type=pathlib.Path, default=WORDFREQ)
    parser.add_argument("--out", type=pathlib.Path, default=STIMULI)
    parser.add_argument("--trap-votes", type=pathlib.Path, default=TRAP_VOTES)
    parser.add_argument("--variant-votes", type=pathlib.Path, default=VARIANT_VOTES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    words = json.loads(args.candidates.read_text(encoding="utf-8"))["words"]
    vocabulary = load_vocabulary(args.wordfreq)
    variant_votes = load_votes(args.variant_votes, ("word_type", "rendering"))
    trap_votes = load_votes(args.trap_votes, ("word_type",))

    rows: list[dict] = []
    summary = collections.Counter()
    per_phenomenon = collections.Counter()

    for word in words:
        if not word["sentences"]:
            summary["skipped_no_sentence"] += 1
            continue
        carrier = word["sentences"][0]
        item_id = word["word_id"].replace("nw", "s")
        base_sentence, start, end = render(
            carrier["text"], carrier["target_char_start"],
            carrier["target_char_end"], word["word"])

        def row(rendering, role, **extra):
            sentence, s0, s1 = render(carrier["text"],
                                      carrier["target_char_start"],
                                      carrier["target_char_end"], rendering)
            return {
                "rendering_id": f"{item_id}_r{len([r for r in rows if r['item_id'] == item_id])}",
                "item_id": item_id, "word_type": word["word"],
                "rendering": rendering, "role": role,
                "licence": "", "corpus_count": vocabulary.get(rendering, 0),
                "collision": "", "collision_count": 0,
                "rule_path": "", "phenomena": "", "ortho_classes": "",
                "n_edits": 0, "score": "",
                "trap_word": "", "trap_reason": "",
                "sentence": sentence, "target_start": s0, "target_end": s1,
                "provenance": "nilechat_machine_derived",
                "source_doc_id": carrier["doc_id"],
                "council_valid": "", "council_reason": "",
                **extra,
            }

        rows.append(row(word["word"], "base", licence="base"))
        summary["base"] += 1

        # The vocabulary here powers the ATTESTATION LICENCE on the two
        # taa-marbuta ops (council pass 1: the result must be a corpus word to
        # fire at all). It does NOT flag collisions -- telling "same word
        # respelled" (3ashan/3shan) from "different word reached by a valid
        # rule" (men/meen) is precisely what the corpus cannot do, and that
        # judgement is council pass 2's. corpus_count carries the mechanical
        # fact; council_valid/collision carry the verdict.
        variants = enumerate_variants(word["word"], max_depth=args.max_depth,
                                      vocabulary=vocabulary)
        chosen = pick_variants(variants, vocabulary, args.variants_per_word)
        if len(chosen) < args.variants_per_word:
            summary["underfilled_words"] += 1
        for tier, count, variant in chosen:
            vote = variant_votes.get((word["word"], variant.form), {})
            rows.append(row(
                variant.form, "variant", licence=tier, corpus_count=count,
                rule_path="|".join(f"{op}@{at}" for op, at in variant.path),
                phenomena="+".join(variant.phenomena),
                ortho_classes="+".join(variant.classes),
                n_edits=variant.n_edits, score=f"{variant.score:.4f}",
                collision=vote.get("collision", ""),
                collision_count=int(vote["collision_count"]) if vote.get("collision_count") else 0,
                council_valid=vote.get("valid", ""),
                council_reason=vote.get("reason", ""),
            ))
            summary[f"variant_{tier}"] += 1
            for cls in variant.classes:
                per_phenomenon[cls] += 1
            if vote.get("valid") == "false":
                summary["council_invalid"] += 1

        trap = trap_votes.get((word["word"],))
        if trap:
            rows.append(row(
                trap["trap"], "trap", licence="trap",
                trap_word=trap["trap"], trap_reason=trap.get("reason", ""),
                ortho_classes="TRAP",
                council_valid=trap.get("valid", ""),
            ))
            summary["trap"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": 1,
        "provenance": "machine_derived (Command R+ transliteration); probe stimulus only",
        "n_rows": len(rows),
        "n_items": summary["base"],
        "counts": dict(summary),
        "by_ortho_class": dict(per_phenomenon.most_common()),
        "attest_min": ATTEST_MIN,
        "note_on_attested": "attested = attested in Command R+ output, not in "
                            "human writing; the human-authored arm is the check",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "variants_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(rows):,} rows -> {args.out}")
    for key in sorted(summary):
        print(f"  {key}: {summary[key]:,}")
    print(f"ortho classes: {dict(per_phenomenon.most_common(8))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
