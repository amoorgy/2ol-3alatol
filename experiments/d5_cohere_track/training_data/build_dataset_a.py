"""Dataset A: seeded spelling permutations of real NileChat sentences.

Three modes, one fat record file as the spine:

  (default)        generate: sentences JSONL -> fat records with substitutions
  --prep-council   chunk fat records into /tmp judge files for the 5-lens council
  --emit           fat records + council votes (+ user overrides) -> tracked
                   train/val JSONL + stats

Rules of the house carried over: seeded determinism (generation seed 13, split
seed 42 by SENTENCE so no sentence leaks across train/val), provenance on
every row, council verdicts flagged never deleted, rejected rows exported to a
tracked CSV for manual review.

    .venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_a.py
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval.diversity import select_diverse
from franco_eval.licensing import SELF_LICENSED, BY_ID, enumerate_variants
from franco_eval.respell import Span, eligible_spans, respell, sample_rendering

SENTENCES = ROOT / "data" / "generated" / "training_sentences.jsonl"
WORDFREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
FAT = ROOT / "data" / "generated" / "training_a_full.jsonl"
VOTES = ROOT / "data" / "stimulus" / "training_a_votes.tsv"
OUT_DIR = ROOT / "data" / "training" / "a_v1"
CHUNK_DIR = pathlib.Path("/tmp/council_ta_chunks")

GEN_SEED = 13
SPLIT_SEED = 42
ATTEST_MIN = 20
ATTESTED_BOOST = 2.0  # attested forms twice as likely as rule-safe, per score


def load_vocabulary() -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    with WORDFREQ.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            word, count, _rest = line.split("\t", 2)
            vocabulary[word] = int(count)
    return vocabulary


def weighted_variants(word: str, vocabulary: dict[str, int],
                      cache: dict) -> list[tuple[str, float, dict]]:
    """[(form, sampling weight, metadata)] -- weight = score x tier boost."""
    if word in cache:
        return cache[word]
    out = []
    for variant in enumerate_variants(word, vocabulary=vocabulary):
        attested = vocabulary.get(variant.form, 0) >= ATTEST_MIN
        weight = variant.score * (ATTESTED_BOOST if attested else 1.0)
        out.append((variant.form, weight, {
            "licence": "attested" if attested else "rule_safe",
            "corpus_count": vocabulary.get(variant.form, 0),
            "rule_path": "|".join(f"{op}@{at}" for op, at in variant.path),
            "ortho_classes": "+".join(variant.classes),
            "score": round(variant.score, 4),
        }))
    cache[word] = out
    return out


def load_votes() -> dict[str, dict]:
    votes: dict[str, dict] = {}
    if VOTES.exists():
        with VOTES.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                votes[row["rendering_id"]] = row
    return votes


def generate(args) -> int:
    vocabulary = load_vocabulary()
    cache: dict = {}
    rng = random.Random(GEN_SEED)
    split_rng = random.Random(SPLIT_SEED)

    # A re-run must not silently invalidate a council verdict already spent
    # on a rendering_id's OLD text: rendering_id is positional (sentence +
    # index), not content-derived, so overwriting it with different text
    # under the same id would leave the vote pointing at text nobody judged.
    # Any permutation row with a vote is therefore carried through
    # byte-identical; only its still-unjudged siblings get replaced.
    votes = load_votes()
    existing_by_sentence: dict[str, list[dict]] = collections.defaultdict(list)
    if FAT.exists():
        with FAT.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                existing_by_sentence[row["sentence_id"]].append(row)

    n_rows = 0
    n_sentences = 0
    per_licence = collections.Counter()
    FAT.parent.mkdir(parents=True, exist_ok=True)
    with SENTENCES.open(encoding="utf-8") as inp, \
            FAT.open("w", encoding="utf-8") as out:
        for line in inp:
            if args.limit and n_sentences >= args.limit:
                break
            record = json.loads(line)
            spans = [Span(s["word"], s["start"], s["end"])
                     for s in record["spans"]]
            variants_by_word = {
                span.word: [(f, w) for f, w, _m in
                            weighted_variants(span.word, vocabulary, cache)]
                for span in spans}
            usable = eligible_spans(spans, variants_by_word)
            if len(usable) < 2:
                continue  # nothing worth permuting
            n_sentences += 1
            split = "train" if split_rng.random() < 0.9 else "val"

            existing_rows = existing_by_sentence.get(record["sentence_id"], [])
            resolved = [r for r in existing_rows
                       if r["role"] == "permutation" and r["rendering_id"] in votes]
            resolved_choices = [{s["word"]: s["rendering"]
                                 for s in r["substitutions"]}
                                for r in resolved]
            next_index = 1 + max(
                [int(r["rendering_id"].rsplit("_p", 1)[1])
                 for r in existing_rows] or [0])

            base_choices: dict[str, str] = {}
            base_text, base_applied = record["text"], \
                respell(record["text"], spans, {})[1]

            # Sample a pool well beyond the still-needed count, then keep
            # the K that maximize minimum pairwise distance from each other,
            # the base, AND any already-resolved rendering for this sentence
            # (farthest-point/maximin selection) -- plain exact-text dedup
            # alone lets 1-word-different near-copies through, which teach a
            # fine-tuned model to parrot the carrier sentence.
            still_needed = max(args.renderings - len(resolved), 0)
            new_renderings: list[tuple[dict, str, tuple]] = []
            if still_needed:
                seen_texts = {record["text"], *(r["text"] for r in resolved)}
                pool: list[tuple[dict, str, tuple]] = []
                for _ in range(max(still_needed * 10, 30)):
                    choices = sample_rendering(spans, variants_by_word, rng,
                                               p_respell=args.p_respell)
                    text, applied = respell(record["text"], spans, choices)
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    pool.append((choices, text, applied))

                words = sorted({s.word for s in usable})
                anchors = [base_choices] + resolved_choices
                selected = select_diverse(anchors, [c for c, _, _ in pool],
                                          words, still_needed,
                                          min_distance_from_base=2)
                new_renderings = [pool[i] for i in selected]

            metadata_by_word = {
                span.word: {f: m for f, _w, m in
                            weighted_variants(span.word, vocabulary, cache)}
                for span in spans}

            out.write(json.dumps({
                "sentence_id": record["sentence_id"],
                "rendering_id": f"{record['sentence_id']}_p0",
                "role": "base",
                "text": base_text,
                "base_text": record["text"],
                "substitutions": [],
                "split": split,
                "provenance": record.get("provenance",
                                         "nilechat_machine_derived"),
                "source_doc_id": record["doc_id"],
                "council_valid": "",
            }, ensure_ascii=False) + "\n")
            n_rows += 1

            for row in resolved:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                for sub in row["substitutions"]:
                    per_licence[sub["licence"]] += 1

            for offset, (choices, text, applied) in enumerate(new_renderings):
                substitutions = []
                for a in applied:
                    if a.is_base:
                        continue
                    substitutions.append({
                        "word": a.word, "rendering": a.rendering,
                        "start": a.start, "end": a.end,
                        **metadata_by_word[a.word][a.rendering]})
                out.write(json.dumps({
                    "sentence_id": record["sentence_id"],
                    "rendering_id": f"{record['sentence_id']}_p{next_index + offset}",
                    "role": "permutation",
                    "text": text,
                    "base_text": record["text"],
                    "substitutions": substitutions,
                    "split": split,
                    "provenance": record.get("provenance",
                                             "nilechat_machine_derived"),
                    "source_doc_id": record["doc_id"],
                    "council_valid": "",
                }, ensure_ascii=False) + "\n")
                n_rows += 1
                for sub in substitutions:
                    per_licence[sub["licence"]] += 1

    print(f"{n_rows:,} rows from {n_sentences:,} sentences -> {FAT}")
    print(f"substitutions by licence: {dict(per_licence)}")
    return 0


def prep_council(args) -> int:
    """Chunk permutation rows into judge files. Base rows need no judging."""
    rows = [json.loads(line) for line in FAT.open(encoding="utf-8")]
    judgeable = [
        {"sid": r["sentence_id"], "rid": r["rendering_id"],
         "base": r["base_text"], "text": r["text"],
         "subs": [{"w": s["word"], "r": s["rendering"]}
                  for s in r["substitutions"]]}
        for r in rows if r["role"] == "permutation" and r["substitutions"]]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("*.json"):
        old.unlink()
    size = args.chunk_size
    for i in range(0, len(judgeable), size):
        (CHUNK_DIR / f"c{i // size}.json").write_text(
            json.dumps(judgeable[i:i + size], ensure_ascii=False),
            encoding="utf-8")
    n_chunks = (len(judgeable) + size - 1) // size
    print(f"{len(judgeable):,} permutation rows -> {n_chunks} chunks of "
          f"<= {size} in {CHUNK_DIR} (x5 lenses = {n_chunks * 5} judge calls)")
    return 0


def load_overrides(path: pathlib.Path) -> set[str]:
    """rendering_ids the user manually accepted in the review CSV."""
    if not path.exists():
        return set()
    accepted = set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            verdict = row.get("reviewer", "").strip().lower()
            first = verdict.split(" ", 1)[0].rstrip(":,") if verdict else ""
            if first in ("ok", "accept", "accepted", "yes"):
                accepted.add(row["rendering_id"])
    return accepted


def load_force_review(path: pathlib.Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def emit(args) -> int:
    votes = load_votes()
    overrides = load_overrides(OUT_DIR / "rejected_for_review.csv")
    force_review = load_force_review(args.force_review)

    rows = [json.loads(line) for line in FAT.open(encoding="utf-8")]
    kept, rejected, unjudged = [], [], 0
    for row in rows:
        vote = votes.get(row["rendering_id"])
        if row["role"] == "base":
            kept.append(row)
        elif not row["substitutions"]:
            # a permutation whose sampled choices all collapsed to base
            # spelling IS the base sentence: an exact duplicate, not data
            unjudged += 1
        elif vote is None:
            # The council stopped before reaching this chunk (or its panel is
            # incomplete). Unjudged is NOT disputed: these rows are excluded
            # from train/val AND from the manual-review CSV -- the CSV is for
            # rows judges actually flagged. A later council top-up can admit
            # them without touching anything emitted here.
            unjudged += 1
        elif int(vote.get("n_judges", 0)) < 4:
            unjudged += 1  # partial panel: cannot reach the >=4/5 bar fairly
        elif row["rendering_id"] in force_review \
                and row["rendering_id"] not in overrides:
            # calibration suspects: council said valid, but a cross-cohort
            # check impeached that cohort's leniency on this pattern -- the
            # human reviewer decides, not the lenient panel
            rejected.append((row, {**vote, "reason":
                             "calibration suspect: " + vote.get("reason", "")}))
        elif vote["valid"] == "true" or row["rendering_id"] in overrides:
            kept.append(row)
        else:
            rejected.append((row, vote))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    handles = {name: (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "val")}
    try:
        for row in kept:
            if args.format == "pairs":
                if row["role"] == "base":
                    continue  # a pair needs a permutation side
                payload = {"messages": [
                    {"role": "user", "content": row["text"]},
                    {"role": "assistant", "content": row["base_text"]},
                ]}
            else:
                payload = {"text": row["text"]}
            payload["meta"] = {"rendering_id": row["rendering_id"],
                               "provenance": row["provenance"]}
            handles[row["split"]].write(
                json.dumps(payload, ensure_ascii=False) + "\n")
            counts[row["split"]] += 1
    finally:
        for handle in handles.values():
            handle.close()

    review = OUT_DIR / "rejected_for_review.csv"
    if not review.exists() or args.rewrite_review:
        with review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rendering_id", "base_text", "text",
                             "substitutions", "votes_valid", "votes_invalid",
                             "reason", "reviewer"])
            for row, vote in rejected:
                writer.writerow([
                    row["rendering_id"], row["base_text"], row["text"],
                    " ".join(f"{s['word']}->{s['rendering']}"
                             for s in row["substitutions"]),
                    vote.get("votes_valid", ""), vote.get("votes_invalid", ""),
                    vote.get("reason", ""), ""])

    stats = {
        "total_rows": len(rows),
        "kept": len(kept), "rejected": len(rejected),
        "unjudged": unjudged,
        "overridden_by_reviewer": len(overrides),
        "forced_to_review": len(force_review),
        "train": counts["train"], "val": counts["val"],
        "format": args.format,
        "seeds": {"generation": GEN_SEED, "split": SPLIT_SEED},
        "provenance": "nilechat_machine_derived; council-verified; "
                      "machine corpus is measurement/training material, "
                      "never a benchmark artifact",
        "fat_sha256": hashlib.sha256(FAT.read_bytes()).hexdigest()[:16],
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1),
                                        encoding="utf-8")
    print(f"kept {len(kept):,} (train {counts['train']:,} / val "
          f"{counts['val']:,}), rejected {len(rejected):,} -> {review}, "
          f"unjudged {unjudged:,} (excluded, eligible for a council top-up)")
    print(f"wrote {OUT_DIR}/{{train,val}}.jsonl + stats.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--renderings", type=int, default=3,
                        help="sampled permutations per sentence (plus 1 base)")
    parser.add_argument("--p-respell", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0,
                        help="sentences to process (0 = all)")
    parser.add_argument("--prep-council", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--format", choices=("text", "pairs"), default="text")
    parser.add_argument("--force-review", type=pathlib.Path, default=None,
                        help="file of rendering_ids to pull from the kept set "
                             "into the review CSV (calibration suspects)")
    parser.add_argument("--rewrite-review", action="store_true",
                        help="overwrite rejected_for_review.csv (drops any "
                             "reviewer column edits -- off by default)")
    args = parser.parse_args()
    if args.prep_council:
        return prep_council(args)
    if args.emit:
        return emit(args)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
