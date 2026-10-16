"""Mine trap candidates: different words at minimal orthographic distance.

The instrument inversion that motivates this script: of all co-skeleton pairs
in NileChat, the ones classify_edge REJECTS (UNRELATED) are already handled --
the dangerous pairs are the ones that CONVERGE, because they are exactly what a
normalisation pipeline would silently merge (e3mel/a3mel, katab/ketab,
wa7ed/wa7da -- almost all hiding in VOWEL_DROP_MED).

String evidence cannot tell those from true spelling variants, so candidates
are ranked by distributional dissimilarity: PPMI context cosine, calibrated on
the human-authored pairs in data/benchmark/pairs.csv (hard rule 1's arm). The
ranking ORDERS the council's reading list; it never labels anything itself.

    .venv/bin/python experiments/d5_cohere_track/nilechat_wordset/mine_traps.py

Deterministic, no seed. Output is gitignored (machine-derived candidates); the
council's committed votes file is what enters version control.
"""

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import word_strings
from franco.mine import edit_distance
from franco.skeleton import skeleton
from franco_eval.context import (accumulate, build_model, calibrate,
                                 context_vocabulary)
from franco_eval.phenomena import (CLIPPING, DIGIT_POSITION, VOWEL_DROP,
                                   VOWEL_DROP_MED, VOWEL_SWAP, classify_edge)

CORPUS = ROOT / "data" / "external" / "nilechat_arabizi_egy" / "data"
SHARD = "train-{:05d}-of-00003.parquet"
WORDFREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
CANDIDATES = ROOT / "data" / "generated" / "nilechat_candidates.json"
OUT = ROOT / "data" / "generated" / "nilechat_trap_candidates.tsv"
PAIRS = ROOT / "data" / "benchmark" / "pairs.csv"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

# A converged pair whose classes all sit in the pure-spelling set is almost
# certainly one word (amreeki/amreeky, elly/elli). A trap needs at least one
# class with room for morphology to hide in.
MORPH_CLASSES = frozenset({VOWEL_SWAP, VOWEL_DROP_MED, VOWEL_DROP,
                           CLIPPING, DIGIT_POSITION})

FOREIGN = frozenset("pvxc")
PREFIXES = ("bey", "bet", "hay", "ha", "by", "bt", "ye", "te", "ne", "y",
            "t", "n", "a", "e")
CLITICS = ("", "a", "ah", "o", "oh", "ha", "hom", "ak", "ek", "ik", "kom", "y")


def load_pool(path, top_n):
    pool = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            word, count, doc_freq, cap_ratio, _sk, _n, _f = line.rstrip("\n").split("\t")
            count, doc_freq, cap_ratio = int(count), int(doc_freq), float(cap_ratio)
            if (len(word) < 3 or word.isdigit() or FOREIGN & set(word)
                    or "6" in word or "9" in word
                    or count < 200 or doc_freq < 100 or cap_ratio > 0.25):
                continue
            pool[word] = (count, doc_freq)
            if len(pool) >= top_n:
                break
    return pool


def deletion_index(words):
    """SymSpell-style: every string reachable by deleting <=2 chars maps back."""
    index = collections.defaultdict(set)
    for word in words:
        keys = {word}
        for i in range(len(word)):
            d1 = word[:i] + word[i + 1:]
            keys.add(d1)
            for j in range(len(d1)):
                keys.add(d1[:j] + d1[j + 1:])
        for key in keys:
            index[key].add(word)
    return index


def neighbours(word, index):
    out = set()
    keys = {word}
    for i in range(len(word)):
        d1 = word[:i] + word[i + 1:]
        keys.add(d1)
        for j in range(len(d1)):
            keys.add(d1[:j] + d1[j + 1:])
    for key in keys:
        out |= index.get(key, set())
    out.discard(word)
    return {n for n in out if edit_distance(word, n) <= 2}


def strip_prefix(word):
    for prefix in PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            return prefix, word[len(prefix):]
    return "", word


def devowel(text):
    return "".join(ch for ch in text if ch not in "aeiou")


def bucket(a, b, classes):
    """Morphology tag, first match wins. Tags guide stratification, not truth."""
    pa, sa = strip_prefix(a)
    pb, sb = strip_prefix(b)
    if sa == sb and pa != pb:
        # t/te, bt/bet are spelling variants of ONE prefix: reject when the
        # prefixes are identical after vowel-stripping. The exception is the
        # bare a/e alternation, which is 1sg vs imperative -- the e3mel/a3mel
        # case this experiment is named after -- unless the stem starts with l
        # (then a-/e- is the definite article al-/el-, one word).
        if devowel(pa) != devowel(pb):
            return "person_prefix"
        if {pa, pb} <= {"a", "e"} and not sa.startswith("l"):
            return "person_prefix"
    for x, y in ((a, b), (b, a)):
        for suffix in ("een", "in", "yn"):
            if x.endswith(suffix):
                stem = x[: -len(suffix)]
                if y in (stem, stem + "a") and set(classes) != {VOWEL_SWAP}:
                    return "number"
    for ca in CLITICS:
        for cb in CLITICS:
            if ca == cb:
                continue
            base_a = a[: len(a) - len(ca)] if (not ca or a.endswith(ca)) else None
            base_b = b[: len(b) - len(cb)] if (not cb or b.endswith(cb)) else None
            if base_a and base_a == base_b and len(base_a) >= 2:
                return "gender_clitic"
    if skeleton(a) == skeleton(b):
        return "template_vowel"
    return "minimal_pair"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trap-pool", type=int, default=6000)
    parser.add_argument("--per-word", type=int, default=8,
                        help="candidates kept per target word before ranking")
    parser.add_argument("--unrelated-quota", type=int, default=40,
                        help="UNRELATED pairs kept as easy controls, one per word")
    parser.add_argument("--min-count-pair", type=int, default=200)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    targets = [w["word"] for w in
               json.loads(CANDIDATES.read_text(encoding="utf-8"))["words"]]
    pool = load_pool(WORDFREQ, args.trap_pool)
    for word in targets:
        pool.setdefault(word, (0, 0))
    print(f"pool: {len(pool):,} words ({len(targets)} targets)", flush=True)

    index = deletion_index(pool)
    candidates = []
    for word in targets:
        for other in sorted(neighbours(word, index)):
            if pool[other][0] < args.min_count_pair:
                continue
            edge = classify_edge(word, other)
            if not edge.is_variant:
                tag = "unrelated_control"
            elif set(edge.classes) & MORPH_CLASSES:
                tag = bucket(word, other, edge.classes)
            else:
                continue  # pure spelling: almost certainly the same word
            candidates.append({
                "word_type": word, "trap": other,
                "edit_distance": edit_distance(word, other),
                "ortho_classes": "+".join(edge.classes),
                "converged": edge.converged,
                "bucket": tag,
                "count_trap": pool[other][0], "doc_freq_trap": pool[other][1],
            })
    print(f"candidates after class filter: {len(candidates):,}", flush=True)

    # --- context vectors over the corpus -------------------------------------
    import pyarrow.parquet as pq
    involved = ({c["word_type"] for c in candidates}
                | {c["trap"] for c in candidates})
    import csv as _csv
    with PAIRS.open(encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            involved |= {row["a"].lower(), row["b"].lower()}

    counts = {w: pool.get(w, (0, 0))[0] for w in pool}
    ctx_index = context_vocabulary(counts, skip_top=20, size=1500)

    def sentence_stream():
        docs = 0
        for shard in (0, 1, 2):
            pf = pq.ParquetFile(CORPUS / SHARD.format(shard))
            for batch in pf.iter_batches(batch_size=4000, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    docs += 1
                    if args.limit_docs and docs > args.limit_docs:
                        return
                    for line in text.split("\n"):
                        yield word_strings(line)

    print("building context vectors ...", flush=True)
    model = build_model(*accumulate(sentence_stream(), involved, ctx_index))
    print(f"  vectors for {len(model.vectors):,}/{len(involved):,} words", flush=True)

    calibration = calibrate(model, PAIRS)
    print(f"  calibration on human arm: AUC={calibration.auc} "
          f"threshold={calibration.threshold} "
          f"({calibration.n_same} SAME / {calibration.n_different} DIFFERENT, "
          f"{calibration.n_skipped} skipped)", flush=True)

    for c in candidates:
        c["ctx_cos"] = model.cosine(c["word_type"], c["trap"])

    # Rank: lowest cosine first (most different in context = most trap-like);
    # None (no vector) sinks to the bottom rather than pretending to be 0.
    candidates.sort(key=lambda c: (c["ctx_cos"] is None,
                                   c["ctx_cos"] if c["ctx_cos"] is not None else 1.0,
                                   c["word_type"], c["trap"]))

    # Converged candidates are the point of the exercise -- they are what a
    # normalisation pipeline would merge. UNRELATED pairs are already rejected
    # by the instrument, so they enter only as a small control quota: one per
    # word, ~40 overall, lowest cosine first.
    per_word = collections.Counter()
    kept = []
    for c in candidates:
        if c["bucket"] == "unrelated_control":
            continue
        if per_word[c["word_type"]] >= args.per_word:
            continue
        per_word[c["word_type"]] += 1
        c["ctx_rank"] = per_word[c["word_type"]]
        kept.append(c)
    controls = 0
    control_words = set()
    for c in candidates:
        if controls >= args.unrelated_quota:
            break
        if c["bucket"] != "unrelated_control" or c["word_type"] in control_words:
            continue
        control_words.add(c["word_type"])
        controls += 1
        c["ctx_rank"] = 0
        kept.append(c)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["word_type", "trap", "edit_distance", "ortho_classes", "converged",
              "bucket", "ctx_cos", "ctx_rank", "count_trap", "doc_freq_trap",
              "verdict", "reviewer"]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for c in kept:
            c["ctx_cos"] = "" if c["ctx_cos"] is None else f"{c['ctx_cos']:.4f}"
            c["verdict"] = ""
            c["reviewer"] = ""
            writer.writerow(c)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "context_calibration.json").write_text(json.dumps({
        "auc": calibration.auc, "threshold": calibration.threshold,
        "n_same": calibration.n_same, "n_different": calibration.n_different,
        "n_skipped": calibration.n_skipped, "usable": calibration.usable,
        "calibrated_on": "data/benchmark/pairs.csv (human-authored arm)",
        "skip_top": 20, "ctx_vocab": 1500, "window": 2,
    }, indent=1), encoding="utf-8")

    coverage = len({c["word_type"] for c in kept})
    print(f"kept {len(kept):,} candidates covering {coverage}/{len(targets)} words")
    print(f"buckets: {dict(collections.Counter(c['bucket'] for c in kept))}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
