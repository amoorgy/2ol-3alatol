"""Level 3 — does a MODEL understand the same word across spellings?

For every probe item, the model sees the carrier sentence with the target word
rendered several ways and answers "what does <word> mean here? one Arabic
word". Scoring is agreement across renderings -- no gold needed:

  variants SHOULD agree      (disagreement = under-robust / brittle)
  the trap SHOULD disagree   (agreement    = over-normalised)

The two rates are never collapsed into one score. A same-rendering repeat is
asked as the generation-noise ceiling, and a 30-word Arabic-script probe is the
capability floor: a model that cannot give the Arabic meaning of an
Egyptian word at all must not have its Arabizi numbers read as an
orthography effect.

Pure stdlib. Local Ollama only. Cache makes every run incremental, and the
pilot subset is hash-stratified so the 150 are always a subset of the 500.

    .venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level3_meaning.py \
        --models "aya-expanse:8b,gemma4:e2b,qwen3.5:9b,llama3.1:8b" --limit 150
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

from franco_eval.arabic_norm import first_arabic_word
from franco_eval.meaning_metrics import ItemMeanings, score
from franco_eval.model_clients import GenConfig, JsonlCache, chat_with_retries
from franco_eval.script_id import is_arabic_answer

STIMULI = ROOT / "data" / "stimulus" / "nilechat" / "stimuli_v1.csv"
CONTROL = ROOT / "data" / "stimulus" / "arabic_control_v1.tsv"
CACHE = ROOT / "data" / "generated" / "level3_meaning_cache.jsonl"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

SAMPLE_SEED = 13
FLOOR = 0.5  # capability floor: below this on the Arabic-script probe -> FAIL

PROMPT_EN = """Here is a sentence in Egyptian Arabic written in Latin letters and digits (Franco-Arabic / Arabizi):

{sentence}

In this sentence, what does the word "{rendering}" mean?
Answer with exactly one Arabic word. No explanation, no Latin letters, no quotes."""

PROMPT_ARZ = """دي جملة مكتوبة بالفرانكو (عربي مصري بحروف لاتينية وأرقام):

{sentence}

في الجملة دي، كلمة "{rendering}" يعني إيه؟
جاوب بكلمة عربية واحدة بس، من غير أي شرح ولا حروف لاتينية."""

CONTROL_PROMPT = """ما معنى الكلمة المصرية "{word}"؟
جاوب بكلمة عربية فصحى واحدة فقط، بدون شرح."""


def prompt_for(lang: str, sentence: str, rendering: str) -> str:
    template = PROMPT_EN if lang == "en" else PROMPT_ARZ
    return template.format(sentence=sentence, rendering=rendering)


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_items(path: pathlib.Path) -> dict[str, dict]:
    """item_id -> {base row, variant rows, trap row} from the stimulus file."""
    items: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["council_valid"] == "false":
                continue  # kept in the file, never scored
            entry = items.setdefault(row["item_id"],
                                     {"base": None, "variants": [], "trap": None})
            if row["role"] == "base":
                entry["base"] = row
            elif row["role"] == "variant":
                entry["variants"].append(row)
            elif row["role"] == "trap" and entry["trap"] is None:
                entry["trap"] = row
    return {k: v for k, v in items.items() if v["base"] is not None}


def pick(item_ids: list[str], stratum_of: dict[str, str], limit: int) -> set[str]:
    """Hash-stratified nested sampling: the 150 are always a subset of the 500.

    Plain slicing would silently change the "same" pilot whenever the input
    order changed; ranking by a seeded hash inside each stratum cannot.
    """
    if not limit or limit >= len(item_ids):
        return set(item_ids)
    def rank(item_id: str) -> int:
        key = f"{SAMPLE_SEED}|{stratum_of.get(item_id, '')}|{item_id}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for item_id in item_ids:
        by_stratum[stratum_of.get(item_id, "")].append(item_id)
    for members in by_stratum.values():
        members.sort(key=rank)
    chosen: set[str] = set()
    while len(chosen) < limit:
        progressed = False
        for members in by_stratum.values():
            if len(chosen) >= limit:
                break
            for item_id in members:
                if item_id not in chosen:
                    chosen.add(item_id)
                    progressed = True
                    break
        if not progressed:
            break
    return chosen


def parse(raw: str) -> str | None:
    """Script gate first, then meaning extraction. None is a reported category."""
    if not is_arabic_answer(raw):
        return None
    return first_arabic_word(raw)


def build_calls(items: dict[str, dict], chosen: set[str], lang: str) -> list[dict]:
    calls = []
    for item_id in sorted(chosen):
        entry = items[item_id]
        base = entry["base"]
        def call(row, kind, repeat=0):
            text = prompt_for(lang, row["sentence"], row["rendering"])
            return {"item_id": item_id, "kind": kind,
                    "rendering_id": row["rendering_id"] + ("~2" if repeat else ""),
                    "prompt": text, "prompt_sha": prompt_sha(text),
                    "classes": row["ortho_classes"]}
        calls.append(call(base, "base"))
        calls.append(call(base, "repeat", repeat=1))  # the stability ceiling
        for row in entry["variants"]:
            calls.append(call(row, "variant"))
        if entry["trap"]:
            calls.append(call(entry["trap"], "trap"))
    return calls


def run_control(model: str, cache: JsonlCache, timeout: int) -> dict:
    """30 Arabic-script words with accept-sets: the capability floor."""
    rows = []
    with CONTROL.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row)
    n_correct = 0
    n_scored = 0
    for row in rows:
        text = CONTROL_PROMPT.format(word=row["word"])
        record = {"model": model, "rendering_id": f"control_{row['word']}",
                  "prompt_sha": prompt_sha(text)}
        cached = cache.get(record)
        if cached is None:
            raw = chat_with_retries(model, text, timeout=timeout)
            record["raw"] = raw
            cache.put(record)
        else:
            raw = cached["raw"]
        meaning = parse(raw)
        accepts = {a.strip() for a in row["accepts"].split("|")}
        from franco_eval.arabic_norm import normalize_arabic
        accepts = {normalize_arabic(a) for a in accepts} - {None}
        if meaning is not None:
            n_scored += 1
            if meaning in accepts:
                n_correct += 1
    accuracy = round(n_correct / len(rows), 4) if rows else None
    return {"n_words": len(rows), "n_scored": n_scored,
            "accuracy": accuracy,
            "capability": ("PASS" if accuracy is not None and accuracy >= FLOOR
                           else "FAIL")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", required=True,
                        help="comma-separated Ollama tags")
    parser.add_argument("--stimuli", type=pathlib.Path, default=STIMULI)
    parser.add_argument("--limit", type=int, default=None,
                        help="items per model; hash-stratified, nested")
    parser.add_argument("--prompt-lang", choices=("en", "arz"), default="en")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--cache-stats", action="store_true",
                        help="report cache coverage and exit; no inference")
    parser.add_argument("--dry-run", action="store_true",
                        help="count the calls that WOULD be made and exit")
    parser.add_argument("--control-only", action="store_true",
                        help="run only the Arabic-script capability floor")
    parser.add_argument("--skip-control", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    items = load_items(args.stimuli)
    stratum_of = {}
    with args.stimuli.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "base":
                first = row["ortho_classes"] or ""
                stratum_of[row["item_id"]] = first
    chosen = pick(sorted(items), stratum_of, args.limit)
    calls = build_calls(items, chosen, args.prompt_lang)
    cache = JsonlCache(args.cache, ("model", "rendering_id", "prompt_sha"))

    print(f"items: {len(chosen)}/{len(items)}  calls/model: {len(calls)}  "
          f"prompt-lang: {args.prompt_lang}")
    if args.dry_run or args.cache_stats:
        for model in models:
            wanted = [{"model": model, "rendering_id": c["rendering_id"],
                       "prompt_sha": c["prompt_sha"]} for c in calls]
            hit, total = cache.coverage(wanted)
            print(f"  {model:<18} cached {hit}/{total}  to-run {total - hit}")
        return 0

    results: dict[str, dict] = {}
    for model in models:
        print(f"\n=== {model} ===")
        control = None
        if not args.skip_control and CONTROL.exists():
            control = run_control(model, cache, args.timeout)
            print(f"  capability floor: {control['accuracy']} "
                  f"[{control['capability']}]")
            if args.control_only:
                results[model] = {"control": control}
                continue

        answers: dict[str, str | None] = {}
        failed = 0
        for index, call in enumerate(calls, start=1):
            record = {"model": model, "rendering_id": call["rendering_id"],
                      "prompt_sha": call["prompt_sha"]}
            cached = cache.get(record)
            if cached is not None:
                answers[call["rendering_id"]] = parse(cached["raw"])
                continue
            try:
                raw = chat_with_retries(model, call["prompt"],
                                        timeout=args.timeout)
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"  fail {call['rendering_id']}: {exc}")
                if failed > 20:
                    print(f"  giving up on {model}")
                    break
                continue
            record["raw"] = raw
            cache.put(record)
            answers[call["rendering_id"]] = parse(raw)
            if index % 100 == 0:
                print(f"  {index}/{len(calls)}", flush=True)
        if failed > 20:
            continue

        meanings: list[ItemMeanings] = []
        for item_id in sorted(chosen):
            entry = items[item_id]
            base_id = entry["base"]["rendering_id"]
            variants = tuple(answers.get(r["rendering_id"])
                             for r in entry["variants"])
            classes = tuple(tuple(c for c in r["ortho_classes"].split("+") if c)
                            for r in entry["variants"])
            meanings.append(ItemMeanings(
                item_id=item_id, word_type=entry["base"]["word_type"],
                base=answers.get(base_id),
                variants=variants, variant_classes=classes,
                trap=(answers.get(entry["trap"]["rendering_id"])
                      if entry["trap"] else None),
                has_trap=entry["trap"] is not None,
                base_repeat=answers.get(base_id + "~2"),
                has_repeat=(base_id + "~2") in answers,
            ))
        results[model] = {"control": control, **score(meanings)}
        r = results[model]
        print(f"  agreement={r['variant_pairwise_agreement']}  "
              f"under_robust={r['under_robust_rate']}  "
              f"over_norm={r['over_norm_rate']}  "
              f"ceiling={r['ceiling_agreement']}  "
              f"non_answer={r['non_answer_rate']}")

    if not results:
        print("no results; is 'ollama serve' running?")
        return 1

    payload = {
        "schema_version": 1,
        "provenance": "nilechat arm: machine-derived probe stimulus (hard rule 1)",
        "prompt_lang": args.prompt_lang,
        "n_items": len(chosen),
        "limit": args.limit,
        "sample_seed": SAMPLE_SEED,
        "gen": vars(GenConfig()) if hasattr(GenConfig(), "__dict__") else {
            "temperature": 0.0, "num_predict": 24, "seed": 13},
        "models": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.prompt_lang == "en" else f"_{args.prompt_lang}"
    (args.out_dir / f"level3_meaning{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# Level 3 — in-context word-meaning consistency",
        "",
        "**Provenance: machine-derived probe stimulus** (Command R+ Arabizi). "
        "Headlines must also be shown on the human-authored arm.",
        "",
        "Variants of one word SHOULD agree; the trap SHOULD disagree. The two "
        "failure rates are reported side by side and never combined.",
        "",
        "| model | floor | agreement | under-robust | over-norm | ceiling | non-answer |",
        "|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for model, r in sorted(results.items(),
                           key=lambda kv: -(kv[1].get("variant_pairwise_agreement") or 0)):
        floor = r["control"]["capability"] if r.get("control") else "n/a"
        lines.append(f"| {model} | {floor} | {fmt(r.get('variant_pairwise_agreement'))} "
                     f"| {fmt(r.get('under_robust_rate'))} "
                     f"| {fmt(r.get('over_norm_rate'))} "
                     f"| {fmt(r.get('ceiling_agreement'))} "
                     f"| {fmt(r.get('non_answer_rate'))} |")
    (args.out_dir / f"level3_meaning{suffix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {(args.out_dir / ('level3_meaning' + suffix + '.md')).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
