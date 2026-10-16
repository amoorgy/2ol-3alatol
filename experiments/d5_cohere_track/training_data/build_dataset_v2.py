"""v2 master records: respell the whole-corpus sentences and the fused ones.

Reads   data/generated/v2/sentences_nilechat.jsonl  (collect --split, whole corpus)
        data/training/v2/fused_sentences.jsonl      (build_dataset_b --split output,
                                                    committed: not regenerable; optional)
Writes  data/generated/v2/master.jsonl        one line per rendering
        data/generated/v2/master_stats.json

Same respeller as a_v1 (licensed variants, attested forms x2, farthest-point
selection), with three v2 changes:
  - every candidate substitution passes franco_eval.substitution_policy at
    its position -- the a_v1 review's rulings as code, because no row here is
    individually judged. Every denial is counted by reason.
  - 2 renderings per sentence, not 3: changes.md 2026-08-24 addendum 4 found
    carrier repetition, not rendering similarity, drives memorisation.
  - RNG seeded PER SENTENCE and split by a hash of the sentence id, so adding
    fused sentences later never reshuffles the NileChat renderings.

Fused sentences keep their base row even with nothing respellable (the
meaning rows need it) and may be respelled at a single span -- the anchor
alone is exactly the variant a meaning row asks about.

    .venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_v2.py
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import tokens
from franco_eval.diversity import select_diverse
from franco_eval.fusion import Anchor, check_sentence
from franco_eval.respell import Span, eligible_spans, respell, sample_rendering
from franco_eval.substitution_policy import (check, load_policy,
                                             variant_from_rule_path)

V2 = ROOT / "data" / "generated" / "v2"
NILECHAT = V2 / "sentences_nilechat.jsonl"
# the TRACKED copy: fused sentences are cloud-model output and cannot be
# regenerated, so the committed file (with its audit) is the source
FUSED = ROOT / "data" / "training" / "v2" / "fused_sentences.jsonl"
MASTER = V2 / "master.jsonl"
STATS = V2 / "master_stats.json"
SPLIT = ROOT / "data" / "stimulus" / "nilechat" / "word_split.tsv"
STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"
TRAP_VOTES = ROOT / "data" / "stimulus" / "nilechat" / "trap_votes.tsv"
TRAINING_VOTES = ROOT / "data" / "stimulus" / "training_a_votes.tsv"

GEN_SEED = 13
SPLIT_SALT = "v2-split-42"
POLICY_VERSION = "v2.0"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VariantSource:
    """Policy-filtered weighted variants, cached per (word, sentence_final)."""

    def __init__(self, dataset_a, vocabulary: dict[str, int], policy) -> None:
        self._a = dataset_a
        self._vocabulary = vocabulary
        self._policy = policy
        self._raw: dict = {}
        self._allowed: dict[tuple[str, bool], list[tuple[str, float]]] = {}
        self.meta: dict[str, dict[str, dict]] = {}
        self.denied: collections.Counter = collections.Counter()

    def allowed(self, word: str, final: bool) -> list[tuple[str, float]]:
        key = (word, final)
        if key not in self._allowed:
            weighted = self._a.weighted_variants(word, self._vocabulary, self._raw)
            self.meta[word] = {f: m for f, _w, m in weighted}
            keep = []
            for form, weight, meta in weighted:
                reason = check(self._policy, word,
                               variant_from_rule_path(form, meta["rule_path"]),
                               sentence_final=final)
                if reason:
                    self.denied[reason] += 1
                else:
                    keep.append((form, weight))
            self._allowed[key] = keep
        return self._allowed[key]


def split_of(sentence_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{sentence_id}".encode()).hexdigest()
    return "val" if int(digest[:8], 16) / 16 ** 8 < 0.1 else "train"


def fused_spans(record: dict, pool: set[str]) -> list[Span]:
    """build_dataset_b marks only the anchor; add every other pool word."""
    toks = tokens(record["text"])
    counts = collections.Counter(w for w, _o, _c in toks)
    anchor = record["anchor"]["word"]
    return [Span(w, o, o + len(w)) for w, o, _c in toks
            if (w == anchor or w in pool) and counts[w] == 1 and o != 0]


def renderings_for(record: dict, spans: list[Span], source: VariantSource,
                   k: int, p_respell: float, min_distance: int) -> list:
    last = max((o for _w, o, _c in tokens(record["text"])), default=-1)
    variants = {s.word: source.allowed(s.word, s.start == last) for s in spans}
    words = sorted({s.word for s in eligible_spans(spans, variants)})
    rng = random.Random(f"{GEN_SEED}:{record['sentence_id']}")
    seen = {record["text"]}
    pool = []
    for _ in range(max(k * 10, 30)):
        choices = sample_rendering(spans, variants, rng, p_respell=p_respell)
        text, applied = respell(record["text"], spans, choices)
        if text not in seen:
            seen.add(text)
            pool.append((choices, text, applied))
    picked = select_diverse([{}], [c for c, _t, _a in pool], words, k,
                            min_distance_from_base=min_distance)
    return words, [pool[i] for i in picked]


def master_rows(record: dict, source_name: str, renderings: list,
                source: VariantSource) -> list[dict]:
    common = {
        "sentence_id": record["sentence_id"],
        "base_text": record["text"],
        "split": split_of(record["sentence_id"]),
        "provenance": record["provenance"],
        "source": source_name,
        "source_doc_id": record["doc_id"],
        "anchor": record.get("anchor"),
        "policy_version": POLICY_VERSION,
    }
    rows = [{**common, "rendering_id": f"{record['sentence_id']}_p0",
             "role": "base", "text": record["text"], "substitutions": []}]
    for index, (_choices, text, applied) in enumerate(renderings, start=1):
        subs = [{"word": a.word, "rendering": a.rendering,
                 "start": a.start, "end": a.end,
                 **source.meta[a.word][a.rendering]}
                for a in applied if not a.is_base]
        rows.append({**common, "rendering_id": f"{record['sentence_id']}_p{index}",
                     "role": "permutation", "text": text, "substitutions": subs})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--renderings", type=int, default=2)
    parser.add_argument("--p-respell", type=float, default=0.55)
    parser.add_argument("--fused", type=pathlib.Path, action="append", default=None,
                        help="fused sentences JSONL, repeatable (default: the tracked "
                             "v2 file; expansion rounds add theirs -- first id wins)")
    parser.add_argument("--limit", type=int, default=0,
                        help="NileChat sentences to process (0 = all)")
    args = parser.parse_args()

    dataset_a = _load("build_dataset_a", HERE / "build_dataset_a.py")
    collect = _load("collect", HERE / "collect_training_sentences.py")
    policy = load_policy(split=SPLIT, stimuli=STIMULI, trap_votes=TRAP_VOTES,
                         training_votes=TRAINING_VOTES)
    source = VariantSource(dataset_a, dataset_a.load_vocabulary(), policy)
    pool = collect.build_pool(collect.load_wordfreq(collect.WORDFREQ))
    pool -= collect.firewall(SPLIT)[0]

    stats = collections.Counter()
    licences = collections.Counter()
    inputs = [(NILECHAT, "nilechat")] + [(p, "fusion") for p in (args.fused or [FUSED])
                                          if p.exists()]
    # 14 anchor words sit in two confirmed trap pairs, so the v2 fusion plan
    # holds them twice; where both draws hit one scenario the sentence id
    # repeats. The plan is left as it was (re-planning would shift every
    # later scenario draw); the first row per id is kept here.
    seen_ids: set[str] = set()
    with MASTER.open("w", encoding="utf-8") as out:
        for path, source_name in inputs:
            fused = source_name == "fusion"
            with path.open(encoding="utf-8") as handle:
                for n, line in enumerate(handle, start=1):
                    if not fused and args.limit and n > args.limit:
                        break
                    record = json.loads(line)
                    if record["sentence_id"] in seen_ids:
                        stats[f"{source_name}_duplicate_id_dropped"] += 1
                        continue
                    seen_ids.add(record["sentence_id"])
                    # re-gate with TODAY's checks: gates tightened after a
                    # sentence was fused (non-Egyptian digits, 2026-09-25)
                    # must still apply to it
                    if fused and not check_sentence(
                            record["text"], Anchor(record["anchor"]["word"])).ok:
                        stats["fusion_failed_current_gates"] += 1
                        continue
                    spans = (fused_spans(record, pool) if fused else
                             [Span(s["word"], s["start"], s["end"])
                              for s in record["spans"]])
                    words, renderings = renderings_for(
                        record, spans, source, args.renderings, args.p_respell,
                        min_distance=1 if fused else 2)
                    if not fused and len(words) < 2:
                        stats[f"{source_name}_skipped_lt2_usable"] += 1
                        continue
                    for row in master_rows(record, source_name, renderings, source):
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats[f"{source_name}_{row['role']}"] += 1
                        for sub in row["substitutions"]:
                            licences[sub["licence"]] += 1
                    if n % 20000 == 0:
                        print(f"  {source_name}: {n:,} sentences", flush=True)

    summary = {"counts": dict(stats), "substitutions_by_licence": dict(licences),
               "policy_version": POLICY_VERSION,
               "policy_denials_by_reason": dict(source.denied),
               "seeds": {"generation": GEN_SEED, "split_salt": SPLIT_SALT},
               "renderings_per_sentence": args.renderings}
    STATS.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                     encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"wrote {MASTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
