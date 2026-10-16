"""Select 500 Egyptian Arabizi word types from NileChat, each in a carrier sentence.

nilechat-arabizi-egy is machine-derived (Command R+ transliterating Egyptian
Arabic), so under hard rule 1 it is a MEASUREMENT CORPUS and a PROBE STIMULUS,
never a benchmark artifact. Every row this script writes carries provenance,
and every headline computed from it must also be shown on the human-authored
arm in data/benchmark/pairs.csv.

Two streaming passes over the shards:
  1  frequency table + the variance diagnostic + the 500-word selection
  2  carrier sentences for the selected words

Nothing here is stochastic; there is no seed to set or mis-set.

    .venv/bin/python experiments/d5_cohere_track/nilechat_wordset/build_wordset.py
"""

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import features, is_clean, sentences, tokens, word_strings
from franco.mine import edit_distance
from franco.skeleton import skeleton
from franco_eval.phenomena import EXCLUDED_DIGITS, classify_edge

CORPUS = ROOT / "data" / "external" / "nilechat_arabizi_egy" / "data"
SHARD = "train-{:05d}-of-00003.parquet"
OUT_FREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
OUT_CANDIDATES = ROOT / "data" / "generated" / "nilechat_candidates.json"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
HUMAN_ARM = ROOT / "data" / "benchmark" / "pairs.csv"

# Latin letters that never occur in this transliteration scheme. Their presence
# marks a loanword bleeding through (mcrofon, code, congress), not Arabizi.
FOREIGN_LETTERS = frozenset("pvxc")

# Arabizi function words that happen to be spelled like English ones. These are
# the corpus's most frequent types, so a careless stoplist would decapitate the
# word set: men=من (857k occurrences), we=و, ma=ما, aw=أو, kan=كان, sana=سنة.
PROTECTED = frozenset("""
men we be en ma aw ana kan sana man mesh msh el ela ala ana enta enti howa heya
ehna hom da dh dah di dee deh kol kul fe fi feh men mn 3an 3ala law lw bas bs
""".split())

_ENGLISH = """
about above after again against all also am an and any are as at back because
been before being below between both but by came can cannot could did does
doing down during each even every few first for from further get got had has
have having her here hers herself him himself his how however into its itself
just knew know less like made make many may me might more most much must my
myself never new next nor not now off once only or other others ought our ours
ourselves out over own please put quite rather really said same say says see
seen seem shall she should since some such take than that the their theirs them
themselves then there these they this those though through thus too under until
up upon us use used using very was way well went were what when where which
while who whom whose why will with within without would yes yet you your yours
yourself http https www com org net html page click here home news search email
contact login user file data code image video link site web online
""".split()
ENGLISH_STOPLIST = frozenset(_ENGLISH) - PROTECTED

QUOTAS = {"digit": 170, "diph": 80, "vlong": 70, "digraph": 70, "gem": 60, "finalyi": 50}


# --- pass 1: frequency ------------------------------------------------------

def count_shards(shards, limit_docs, prune_every, batch_size):
    """word -> [count, doc_freq, cap_count], plus corpus hygiene counters."""
    import pyarrow.parquet as pq

    table: dict[str, list[int]] = {}
    hygiene = collections.Counter()
    docs = 0
    prune_events = 0
    for shard in shards:
        path = CORPUS / SHARD.format(shard)
        if not path.exists():
            raise SystemExit(f"missing shard: {path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                       columns=["text"]):
            for text in batch.column("text").to_pylist():
                docs += 1
                hygiene["docs"] += 1
                if not is_clean(text):
                    hygiene["docs_contaminated"] += 1
                seen: set[str] = set()
                for word, _offset, capitalised in tokens(text):
                    row = table.get(word)
                    if row is None:
                        row = table[word] = [0, 0, 0]
                    row[0] += 1
                    row[2] += capitalised
                    if word not in seen:
                        seen.add(word)
                        row[1] += 1
                if prune_every and docs % prune_every == 0:
                    table = {w: r for w, r in table.items() if r[0] > 1}
                    prune_events += 1
                if limit_docs and docs >= limit_docs:
                    break
            if limit_docs and docs >= limit_docs:
                break
        if limit_docs and docs >= limit_docs:
            break
    hygiene["prune_events"] = prune_events
    return table, hygiene


# --- variance diagnostic ----------------------------------------------------

def variance_tiers(table, top_n):
    """How much real spelling variance a machine-transliterated corpus holds.

    Tier 1 groups by skeleton() alone and OVERSTATES the answer: the key is
    lossy enough that one cluster mixed قال / أول / يوليو / ليلى. Tier 2 keeps
    only forms within edit distance 2 of the cluster head that classify_edge
    can also explain orthographically. Tier 2 is the reportable number.

    Tier 3 (context-confirmed, via PPMI cosine) is computed separately by the
    trap-mining stage, which owns the context vectors.
    """
    ranked = sorted(table.items(), key=lambda kv: (-kv[1][0], kv[0]))
    top = [w for w, r in ranked[:top_n] if len(w) >= 3]

    raw: dict[str, list[str]] = collections.defaultdict(list)
    for word in top:
        raw[skeleton(word)].append(word)

    confirmed: dict[str, list[str]] = {}
    for key, forms in raw.items():
        if len(key) < 2 or len(forms) < 2:
            confirmed[key] = forms[:1] if forms else []
            continue
        forms = sorted(forms, key=lambda w: -table[w][0])
        head = forms[0]
        kept = [head] + [
            w for w in forms[1:]
            if edit_distance(w, head) <= 2 and classify_edge(w, head).is_variant
        ]
        confirmed[key] = kept

    def histogram(groups):
        sizes = collections.Counter(len(v) for v in groups.values() if v)
        total = sum(sizes.values())
        multi = sum(n for size, n in sizes.items() if size > 1)
        return {
            "n_skeletons": total,
            "n_singleton": sizes.get(1, 0),
            "n_multi": multi,
            "pct_multi": round(multi / total, 4) if total else 0.0,
            "size_histogram": {str(k): sizes[k] for k in sorted(sizes)},
        }

    return {
        "top_n": top_n,
        "n_types_considered": len(top),
        "tier1_raw_skeleton": histogram(raw),
        "tier2_confirmed": histogram(confirmed),
        "largest_raw_clusters": [
            {"skeleton": k, "forms": sorted(v, key=lambda w: -table[w][0])[:12]}
            for k, v in sorted(raw.items(), key=lambda kv: -len(kv[1]))[:10]
        ],
    }


# --- selection --------------------------------------------------------------

GATES = ("length", "numeric", "excluded_digit", "foreign_letters",
         "english_stop", "min_count", "min_doc_freq", "proper_noun", "no_feature")


def gate(word, row, args):
    """First failing gate, or None if the word is eligible."""
    count, doc_freq, cap_count = row
    if len(word) < args.min_len:
        return "length"
    if word.isdigit():
        return "numeric"
    if any(d in word for d in EXCLUDED_DIGITS):
        return "excluded_digit"
    if FOREIGN_LETTERS & set(word):
        return "foreign_letters"
    if word in ENGLISH_STOPLIST:
        return "english_stop"
    if count < args.min_count:
        return "min_count"
    if doc_freq < args.min_doc_freq:
        return "min_doc_freq"
    if count and cap_count / count > args.max_cap_ratio:
        return "proper_noun"
    if not features(word):
        return "no_feature"
    return None


def select(table, args, quotas):
    """Deterministic greedy fill by scarcest unfilled stratum.

    Straight top-N-by-frequency gives 33 gemination and 58 diphthong words --
    the same "too thin to defend" flaw the d5 README already lists as open. So
    each word goes to whichever of its strata is furthest from full, and one
    word per skeleton keeps near-duplicates out.
    """
    rejected = collections.Counter()
    pool = []
    for word, row in sorted(table.items(), key=lambda kv: (-kv[1][0], kv[0])):
        reason = gate(word, row, args)
        if reason:
            rejected[reason] += 1
        else:
            pool.append((word, row))

    filled = collections.Counter()
    chosen: list[tuple[str, list[int], str]] = []
    seen_skeletons: set[str] = set()
    for word, row in pool:
        if len(chosen) >= args.n_words:
            break
        key = skeleton(word)
        if key in seen_skeletons:
            continue
        open_strata = [f for f in features(word) if filled[f] < quotas.get(f, 0)]
        if not open_strata:
            continue
        stratum = min(open_strata, key=lambda f: (filled[f] / quotas[f], f))
        filled[stratum] += 1
        seen_skeletons.add(key)
        chosen.append((word, row, stratum))
    return chosen, rejected, filled, len(pool)


# --- pass 2: carrier sentences ---------------------------------------------

def collect_sentences(shards, wanted, table, args):
    """First-in-corpus-order sentences containing each target exactly once.

    Deterministic without a seed. The target must not be sentence-initial:
    substituting a variant there would change its capitalisation and confound
    the manipulation with a casing cue.
    """
    import pyarrow.parquet as pq

    found: dict[str, list[dict]] = {w: [] for w in wanted}
    remaining = set(wanted)
    # One sentence serves at most one target. Without this, common words all
    # draw the same early sentence and a single bad carrier damages several
    # items at once.
    used: set[tuple[str, int, int]] = set()
    docs = 0
    for shard in shards:
        for batch in pq.ParquetFile(CORPUS / SHARD.format(shard)).iter_batches(
                batch_size=args.batch_size, columns=["id", "text"]):
            ids = batch.column("id").to_pylist()
            for doc_id, text in zip(ids, batch.column("text").to_pylist()):
                docs += 1
                if args.limit_docs and docs > args.limit_docs:
                    return found
                if not remaining:
                    return found
                for line_index, sent_index, sentence in sentences(text):
                    key = (doc_id, line_index, sent_index)
                    if key in used or not is_clean(sentence):
                        continue
                    words = word_strings(sentence)
                    if not (args.min_tokens <= len(words) <= args.max_tokens):
                        continue
                    hits = remaining.intersection(words)
                    if not hits:
                        continue
                    known = sum(1 for w in words if table.get(w, [0])[0] >= args.known_min)
                    if known / len(words) < args.known_ratio:
                        continue
                    for word in sorted(hits):
                        if words.count(word) != 1:
                            continue
                        spans = [(t, o) for t, o, _c in tokens(sentence) if t == word]
                        if len(spans) != 1 or spans[0][1] == 0:
                            continue
                        offset = spans[0][1]
                        found[word].append({
                            "doc_id": doc_id, "line_index": line_index,
                            "sent_index": sent_index, "text": sentence,
                            "n_tokens": len(words),
                            "target_char_start": offset,
                            "target_char_end": offset + len(word),
                        })
                        used.add(key)
                        if len(found[word]) >= args.sentences_per_word:
                            remaining.discard(word)
                        break  # this sentence is now spoken for
    return found


def human_arm_words():
    if not HUMAN_ARM.exists():
        return set()
    import csv
    with HUMAN_ARM.open(encoding="utf-8") as handle:
        return {w for row in csv.DictReader(handle) for w in (row["a"], row["b"])}


# --- reporting --------------------------------------------------------------

def write_freq_table(table, path, min_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(((w, r) for w, r in table.items() if r[0] >= min_count),
                  key=lambda kv: (-kv[1][0], kv[0]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("word\tcount\tdoc_freq\tcap_ratio\tskeleton\tn_chars\tfeatures\n")
        for word, (count, doc_freq, cap_count) in rows:
            handle.write(f"{word}\t{count}\t{doc_freq}\t{cap_count / count:.4f}\t"
                         f"{skeleton(word)}\t{len(word)}\t{','.join(features(word))}\n")
    return len(rows)


def diagnostic_markdown(report):
    hyg, var = report["hygiene"], report["variance"]
    t1, t2 = var["tier1_raw_skeleton"], var["tier2_confirmed"]
    sel = report["selection"]
    lines = [
        "# NileChat word-set diagnostic",
        "",
        "**Provenance: machine-derived.** `nilechat-arabizi-egy` is Egyptian Arabic",
        "transliterated into Arabizi by Command R+ (CC-BY-NC-4.0). Under hard rule 1",
        "it is a measurement corpus and probe stimulus only, never a benchmark",
        "artifact, and every headline computed from it must also be shown on the",
        "human-authored arm in `data/benchmark/pairs.csv`.",
        "",
        f"Shards: {report['config']['shards']} | documents scanned: {hyg['docs']:,} | "
        f"word types kept: {report['n_types']:,}",
        "",
        "## Corpus hygiene",
        "",
        "| property | value |",
        "|---|---|",
        f"| documents with Arabic script, a prime digraph, or a URL | "
        f"{hyg['docs_contaminated']:,} ({hyg['docs_contaminated'] / max(hyg['docs'], 1):.1%}) |",
        f"| frequency-table prune events | {hyg['prune_events']} |",
        "",
        "A word can lose at most one count per prune event, so the worst-case",
        f"undercount is {hyg['prune_events']} against a `--min-count` of "
        f"{report['config']['min_count']}.",
        "",
        "## Spelling variance — is there any?",
        "",
        "The corpus was written by one model applying one convention, so the",
        "premise that humans spell a word five ways may not survive contact with",
        "it. It partly does: Command R+ was inconsistent *across documents*.",
        "",
        "| tier | skeletons | with >1 spelling | share |",
        "|---|---|---|---|",
        f"| 1 — raw `skeleton()` | {t1['n_skeletons']:,} | {t1['n_multi']:,} | {t1['pct_multi']:.1%} |",
        f"| 2 — confirmed | {t2['n_skeletons']:,} | {t2['n_multi']:,} | {t2['pct_multi']:.1%} |",
        "",
        "**Tier 2 is the reportable number.** Tier 1 overstates it because",
        "`skeleton()` deletes vowels, so it merges words that merely share a",
        "consonant frame. Tier 2 keeps only forms within edit distance 2 of the",
        "cluster head that `classify_edge` can also explain orthographically.",
        "",
        "Neither tier licenses mining the SAME arm from this corpus: orthographic",
        "explainability is not word identity. `men`~`meen` (من / مين) reads as a",
        "clean VOWEL_LENGTH edit and is two different words. Variants are",
        "therefore *generated* from a known base word, which makes same-word",
        "identity true by construction.",
        "",
        "## Selection",
        "",
        f"{sel['n_pool']:,} of {report['n_types']:,} types passed every gate; "
        f"{sel['n_chosen']} were selected under stratum quotas.",
        "",
        "| gate | rejected |",
        "|---|---|",
    ]
    for name in GATES:
        lines.append(f"| {name} | {sel['rejected'].get(name, 0):,} |")
    lines += ["", "| stratum | quota | filled |", "|---|---|---|"]
    for name, quota in report["config"]["quotas"].items():
        lines.append(f"| {name} | {quota} | {sel['filled'].get(name, 0)} |")
    lines += [
        "",
        "Quotas exist because the top 500 by raw frequency yield too few",
        "gemination and diphthong words to report per-phenomenon — the same",
        "\"too thin to defend\" flaw the d5 README already lists as open.",
        "",
        f"{sel['n_in_human_arm']} of the selected words also appear in the",
        "human-authored arm; that overlap is recorded, not excluded, because",
        "cross-arm comparison needs it.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", default="0,1,2")
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--min-doc-freq", type=int, default=100)
    parser.add_argument("--max-cap-ratio", type=float, default=0.25)
    parser.add_argument("--n-words", type=int, default=500)
    parser.add_argument("--quota", default=None,
                        help="digit=170,diph=80,... (default matches --n-words 500)")
    parser.add_argument("--diag-n", type=int, default=2000)
    parser.add_argument("--sentences-per-word", type=int, default=3)
    parser.add_argument("--min-tokens", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--known-min", type=int, default=20)
    parser.add_argument("--known-ratio", type=float, default=0.7)
    parser.add_argument("--min-count-table", type=int, default=20)
    parser.add_argument("--prune-every", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--limit-docs", type=int, default=0,
                        help=">0 for a fast smoke run")
    parser.add_argument("--out-freq", type=pathlib.Path, default=OUT_FREQ)
    parser.add_argument("--out-candidates", type=pathlib.Path, default=OUT_CANDIDATES)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    shards = [int(s) for s in args.shards.split(",") if s.strip() != ""]
    quotas = dict(QUOTAS)
    if args.quota:
        quotas = {k: int(v) for k, v in
                  (part.split("=") for part in args.quota.split(","))}
    if sum(quotas.values()) < args.n_words:
        raise SystemExit(f"quotas sum to {sum(quotas.values())} < --n-words {args.n_words}")

    print(f"pass 1: counting shards {shards} ...", flush=True)
    table, hygiene = count_shards(shards, args.limit_docs, args.prune_every,
                                  args.batch_size)
    print(f"  {hygiene['docs']:,} docs, {len(table):,} word types", flush=True)

    n_written = write_freq_table(table, args.out_freq, args.min_count_table)
    print(f"  wrote {n_written:,} rows -> {args.out_freq}", flush=True)

    print("variance diagnostic ...", flush=True)
    variance = variance_tiers(table, args.diag_n)
    print(f"  tier1 {variance['tier1_raw_skeleton']['pct_multi']:.1%} multi-form, "
          f"tier2 {variance['tier2_confirmed']['pct_multi']:.1%}", flush=True)

    chosen, rejected, filled, n_pool = select(table, args, quotas)
    print(f"  pool {n_pool:,} -> selected {len(chosen)}", flush=True)
    if len(chosen) < args.n_words:
        print(f"  WARNING: wanted {args.n_words}, got {len(chosen)}; "
              f"relax a gate or lower a quota", flush=True)

    human = human_arm_words()
    print("pass 2: carrier sentences ...", flush=True)
    carriers = collect_sentences(shards, {w for w, _r, _s in chosen}, table, args)
    n_with = sum(1 for w, _r, _s in chosen if carriers.get(w))

    words = []
    for index, (word, row, stratum) in enumerate(chosen, start=1):
        count, doc_freq, cap_count = row
        words.append({
            "word_id": f"nw{index:04d}", "word": word, "count": count,
            "doc_freq": doc_freq, "cap_ratio": round(cap_count / count, 4),
            "skeleton": skeleton(word), "features": list(features(word)),
            "stratum": stratum, "in_human_arm": word in human,
            "sentences": carriers.get(word, []),
        })

    report = {
        "schema_version": 1,
        "provenance": "machine_derived",
        "corpus": "UBC-NLP/Arabizi-Egypt (nilechat-arabizi-egy), CC-BY-NC-4.0",
        "config": {"shards": shards, "min_count": args.min_count,
                   "min_doc_freq": args.min_doc_freq,
                   "max_cap_ratio": args.max_cap_ratio, "quotas": quotas,
                   "n_words": args.n_words, "limit_docs": args.limit_docs},
        "hygiene": dict(hygiene),
        "n_types": len(table),
        "variance": variance,
        "selection": {
            "n_pool": n_pool, "n_chosen": len(chosen),
            "n_with_sentences": n_with,
            "n_in_human_arm": sum(1 for w in words if w["in_human_arm"]),
            "rejected": dict(rejected), "filled": dict(filled),
        },
    }

    args.out_candidates.parent.mkdir(parents=True, exist_ok=True)
    args.out_candidates.write_text(
        json.dumps({"config": report["config"], "provenance": "machine_derived",
                    "words": words}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "wordset_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "wordset_diagnostic.md").write_text(
        diagnostic_markdown(report), encoding="utf-8")

    print(f"  {n_with}/{len(chosen)} words have >=1 carrier sentence")
    print(f"wrote {args.out_candidates}")
    print(f"wrote {args.out_dir / 'wordset_diagnostic.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
