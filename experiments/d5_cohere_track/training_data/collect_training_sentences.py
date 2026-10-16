"""Collect NileChat sentences with >=2 respellable pool words, for Dataset A.

Differs from the benchmark's carrier collector on purpose: the benchmark wants
one pristine sentence per probe word (one target, sentence "spoken for"); a
training set wants the opposite -- sentences dense in respellable words, each
carrying a spans list so the respeller can permute several words at once.

The contamination firewall is applied HERE, at the word-pool level:
  - the 500 benchmark words (data/generated/nilechat_candidates.json) and the
    human-arm words (data/benchmark/pairs.csv) are excluded from the pool, so
    they can never become spans (they may still appear un-respelled -- they
    include function words whose exclusion would delete the corpus);
  - the exact carrier sentences used by the benchmark stimuli are excluded
    entirely, keyed on (doc_id, line_index, sent_index).

Deterministic: corpus order, no RNG.

v2 (`--split`): the firewall switches from "all 500 benchmark words" to the
HELD-OUT side of data/stimulus/nilechat/word_split.tsv (held-out benchmark
words, their traps, the human arm) -- train-side benchmark words become
respellable. Benchmark carrier sentences stay blocked either way.
`--row-group-order hashed` walks the corpus's 1000-doc row groups in salted-
hash order, so the per-word cap fills from across the whole corpus instead
of from its first pages (with `--doc-stride 1` there is no stride to do it).

    .venv/bin/python experiments/d5_cohere_track/training_data/collect_training_sentences.py
    # v2: the whole corpus
    .venv/bin/python experiments/d5_cohere_track/training_data/collect_training_sentences.py \\
        --split data/stimulus/nilechat/word_split.tsv --doc-stride 1 --n-sentences 0 \\
        --row-group-order hashed --id-prefix n2_ --out data/generated/v2/sentences_nilechat.jsonl
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import is_clean, sentences, tokens, word_strings
from franco_eval.phenomena import EXCLUDED_DIGITS

CORPUS = ROOT / "data" / "external" / "nilechat_arabizi_egy" / "data"
SHARD = "train-{:05d}-of-00003.parquet"
WORDFREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
CANDIDATES = ROOT / "data" / "generated" / "nilechat_candidates.json"
HUMAN_ARM = ROOT / "data" / "benchmark" / "pairs.csv"
OUT = ROOT / "data" / "generated" / "training_sentences.jsonl"

FOREIGN_LETTERS = frozenset("pvxc")


def _english_stoplist() -> frozenset[str]:
    """Reuse the benchmark's curated stoplist instead of duplicating it.

    It lives in build_wordset.py (a sibling script, not a package), so it is
    loaded by file path -- the same pattern tests/test_registry.py uses for
    the tokenizer registry.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_wordset",
        pathlib.Path(__file__).resolve().parents[1]
        / "nilechat_wordset" / "build_wordset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ENGLISH_STOPLIST


def load_wordfreq(path: pathlib.Path) -> dict[str, tuple[int, int, float]]:
    """word -> (count, doc_freq, cap_ratio) from the committed-format TSV."""
    table: dict[str, tuple[int, int, float]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            word, count, doc_freq, cap_ratio, _rest = line.split("\t", 4)
            table[word] = (int(count), int(doc_freq), float(cap_ratio))
    return table


def build_pool(table, *, min_len=3, min_count=200, min_doc_freq=100,
               max_cap_ratio=0.25) -> set[str]:
    """The 36,939-word gate set, reproduced from the TSV (verified exact).

    Same gates as the benchmark's selection minus the has-a-feature gate:
    a training span only needs the variant engine to produce something,
    which the builder checks per word anyway.
    """
    stoplist = _english_stoplist()
    pool = set()
    for word, (count, doc_freq, cap_ratio) in table.items():
        if (len(word) < min_len or word.isdigit()
                or any(d in word for d in EXCLUDED_DIGITS)
                or FOREIGN_LETTERS & set(word)
                or word in stoplist
                or count < min_count or doc_freq < min_doc_freq
                or cap_ratio > max_cap_ratio):
            continue
        pool.add(word)
    return pool


def firewall(split: pathlib.Path | None = None
             ) -> tuple[set[str], set[tuple[str, int, int]]]:
    """(words never to respell, sentence keys never to use).

    Without `split`: every benchmark word and the human arm (a_v1). With it:
    the held-out side of the split only (v2) -- the split already carries
    the human arm and every held-out trap.
    """
    blocked_words: set[str] = set()
    blocked_sentences: set[tuple[str, int, int]] = set()
    if CANDIDATES.exists():
        data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        for entry in data["words"]:
            if split is None:
                blocked_words.add(entry["word"])
            for carrier in entry["sentences"]:
                blocked_sentences.add((carrier["doc_id"], carrier["line_index"],
                                       carrier["sent_index"]))
    if split is not None:
        with split.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["side"] == "heldout":
                    blocked_words.add(row["word"])
    elif HUMAN_ARM.exists():
        with HUMAN_ARM.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                blocked_words.add(row["a"].lower())
                blocked_words.add(row["b"].lower())
    return blocked_words, blocked_sentences


def iter_docs(order: str):
    """(doc_id, text) over all shards, row group by row group.

    "corpus" is file order (identical doc sequence to the old iter_batches
    walk, so a_v1 regenerates); "hashed" permutes the 1000-doc row groups by
    a salted hash -- a block shuffle at the cost of one pass.
    """
    import pyarrow.parquet as pq

    files = [pq.ParquetFile(CORPUS / SHARD.format(shard)) for shard in (0, 1, 2)]
    units = [(shard, group) for shard, pf in enumerate(files)
             for group in range(pf.metadata.num_row_groups)]
    if order == "hashed":
        units.sort(key=lambda u: hashlib.sha256(
            f"v2-rowgroups:{u[0]}:{u[1]}".encode()).hexdigest())
    for shard, group in units:
        table = files[shard].read_row_group(group, columns=["id", "text"])
        yield from zip(table.column("id").to_pylist(),
                       table.column("text").to_pylist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-sentences", type=int, default=3000)
    parser.add_argument("--min-spans", type=int, default=2)
    parser.add_argument("--max-per-word", type=int, default=8,
                        help="cap on sentences a single word may span in, so "
                             "high-frequency words do not dominate the set")
    parser.add_argument("--min-tokens", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--known-min", type=int, default=20)
    parser.add_argument("--known-ratio", type=float, default=0.7)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--max-per-doc", type=int, default=3,
                        help="sentences taken from any single document; a doc "
                             "is one topic, and 12 sentences about fertiliser "
                             "teach less than 3 sentences from 4 topics")
    parser.add_argument("--doc-stride", type=int, default=120,
                        help="consider only docs where sha256(doc_id) %% N == 0, "
                             "scattering the selection across the whole corpus "
                             "instead of draining its first pages; 1 = every doc")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument("--split", type=pathlib.Path, default=None,
                        help="v2 held-out split TSV: firewall its held-out side "
                             "instead of every benchmark word")
    parser.add_argument("--row-group-order", choices=("corpus", "hashed"),
                        default="corpus")
    parser.add_argument("--id-prefix", default="ts")
    args = parser.parse_args()
    limit = args.n_sentences or float("inf")  # 0 = the whole corpus

    table = load_wordfreq(WORDFREQ)
    pool = build_pool(table)
    blocked_words, blocked_sentences = firewall(args.split)
    pool -= blocked_words
    print(f"pool: {len(pool):,} respellable words "
          f"(firewall removed {len(blocked_words)} words, "
          f"{len(blocked_sentences)} benchmark carrier sentences)", flush=True)

    word_use: collections.Counter = collections.Counter()
    kept = 0
    docs = 0
    # 8-byte digests, not the texts: at whole-corpus scale the dedup set
    # would otherwise hold every kept sentence in memory twice
    seen: set[bytes] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for doc_id, text in iter_docs(args.row_group_order):
            docs += 1
            if (args.limit_docs and docs > args.limit_docs) or kept >= limit:
                break
            if docs % 50000 == 0:
                print(f"  {docs:,} docs, {kept:,} kept", flush=True)
            if args.doc_stride > 1 and int(hashlib.sha256(
                    doc_id.encode()).hexdigest()[:8], 16) % args.doc_stride:
                continue
            doc_kept = 0
            for line_index, sent_index, sentence in sentences(text):
                if kept >= limit or doc_kept >= args.max_per_doc:
                    break
                key = (doc_id, line_index, sent_index)
                if key in blocked_sentences or not is_clean(sentence):
                    continue
                words = word_strings(sentence)
                if not (args.min_tokens <= len(words) <= args.max_tokens):
                    continue
                known = sum(1 for w in words
                            if table.get(w, (0,))[0] >= args.known_min)
                if known / len(words) < args.known_ratio:
                    continue
                # tokenize once per sentence (the 36k-target hot path)
                toks = tokens(sentence)
                counts = collections.Counter(w for w, _o, _c in toks)
                spans = [
                    {"word": w, "start": offset, "end": offset + len(w)}
                    for w, offset, _cap in toks
                    if w in pool and counts[w] == 1 and offset != 0
                    and word_use[w] < args.max_per_word
                ]
                if len(spans) < args.min_spans:
                    continue
                digest = hashlib.blake2b(sentence.encode(), digest_size=8).digest()
                if digest in seen:
                    continue  # same text in another doc adds nothing
                seen.add(digest)
                for span in spans:
                    word_use[span["word"]] += 1
                kept += 1
                doc_kept += 1
                out.write(json.dumps({
                    "sentence_id": f"{args.id_prefix}{kept:05d}",
                    "doc_id": doc_id,
                    "line_index": line_index,
                    "sent_index": sent_index,
                    "text": sentence,
                    "n_tokens": len(words),
                    "spans": spans,
                    "provenance": "nilechat_machine_derived",
                }, ensure_ascii=False) + "\n")

    span_total = sum(word_use.values())
    print(f"kept {kept:,} sentences from {docs:,} docs; "
          f"{span_total:,} spans over {len(word_use):,} distinct words")
    print(f"most-used words: {word_use.most_common(8)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
