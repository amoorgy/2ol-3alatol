"""Level 2 — can the MODEL tell a real spelling variant from a look-alike?

Asks each model, for every labelled pair, whether the two spellings are the same
Egyptian word or two different words. Both sections are scored, because a model
that answers SAME to everything scores perfectly on Section 1 and zero on
Section 2 — only the pair of numbers is informative.

Runs against a local Ollama server. Responses are cached per (model, pair,
order) so runs resume instead of restarting.

    .venv/bin/python experiments/d5_cohere_track/e1_variant_probe/run_level2_inference.py \
        --models qwen3.5:0.8b --both-orders
"""

import argparse
import collections
import csv
import json
import pathlib
import re
import statistics
import time
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

PAIRS = ROOT / "data" / "benchmark" / "pairs.csv"
# Results live next to the experiment, derived from this script's own
# location so they move with it. Top-level results/ is D1's showcase.
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
CACHE = ROOT / "data" / "generated" / "level2_cache.jsonl"
OLLAMA = "http://localhost:11434/api/chat"

PROMPT = """You are shown two words written in Egyptian Arabic using Latin \
letters and digits (Franco-Arabic / Arabizi).

Word 1: {a}
Word 2: {b}

Are these two different spellings of the SAME Egyptian word, or are they two \
DIFFERENT words with different meanings?

Answer with exactly one word: SAME or DIFFERENT."""

ANSWER_RE = re.compile(r"\b(SAME|DIFFERENT)\b", re.IGNORECASE)


def ask_with_retries(model: str, prompt: str, timeout: int, tries: int = 3) -> str:
    """Retry transient Ollama failures.

    Swapping a 16GB model out for an 18GB one can make a request fail with a
    5xx; abandoning the model on the first failure loses a whole run for no
    reason, which is what happened to qwen3.8:27b-mlx.
    """
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return ask(model, prompt, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    raise last if last else RuntimeError("unreachable")


def ask(model: str, prompt: str, timeout: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 12, "seed": 13},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return (data.get("message") or {}).get("content", "") or ""


def parse(raw: str) -> str | None:
    m = ANSWER_RE.search(raw or "")
    return m.group(1).upper() if m else None


def load_cache() -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[(rec["model"], rec["pair_id"], rec["order"])] = rec["raw"]
    return out


def append_cache(rec: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def score(rows: list[dict]) -> dict:
    """Confusion matrix over both sections, plus the answer-SAME bias."""
    same_rows = [r for r in rows if r["label"] == "SAME"]
    diff_rows = [r for r in rows if r["label"] == "DIFFERENT"]
    parsed = [r for r in rows if r["answer"]]
    def rate(rows_: list[dict], target: str) -> float | None:
        """None, not zero, when a section has no parsed answers — absent data is
        not a score of zero."""
        vals = [1.0 if r["answer"] == target else 0.0 for r in rows_ if r["answer"]]
        return statistics.fmean(vals) if vals else None

    recall_same = rate(same_rows, "SAME")
    recall_diff = rate(diff_rows, "DIFFERENT")
    say_same = statistics.fmean(
        [1.0 if r["answer"] == "SAME" else 0.0 for r in parsed]
    ) if parsed else 0.0
    by_sub: dict[str, list[float]] = collections.defaultdict(list)
    for r in rows:
        if not r["answer"]:
            continue
        correct = 1.0 if r["answer"] == r["label"] else 0.0
        by_sub[f"{r['label'].lower()}/{r['subcategory']}"].append(correct)

    # Order consistency: does swapping the two words change the verdict?
    per_pair: dict[str, set[str]] = collections.defaultdict(set)
    for r in rows:
        if r["answer"]:
            per_pair[r["pair_id"]].add(r["answer"])
    both = [v for v in per_pair.values() if len(v) >= 1]
    flips = [1.0 if len(v) > 1 else 0.0 for v in per_pair.values()]
    order_flip = statistics.fmean(flips) if flips and len(rows) > len(per_pair) else None
    return {
        "n": len(rows),
        "n_parsed": len(parsed),
        "unparseable_rate": round(1 - len(parsed) / max(1, len(rows)), 4),
        "accuracy_on_SAME": None if recall_same is None else round(recall_same, 4),
        "accuracy_on_DIFFERENT": None if recall_diff is None else round(recall_diff, 4),
        "balanced_accuracy": (None if recall_same is None or recall_diff is None
                              else round((recall_same + recall_diff) / 2, 4)),
        "answers_SAME_rate": round(say_same, 4),
        "order_flip_rate": None if order_flip is None else round(order_flip, 4),
        "by_subcategory": {k: {"acc": round(statistics.fmean(v), 4), "n": len(v)}
                           for k, v in sorted(by_sub.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True,
                    help="comma-separated Ollama model tags")
    ap.add_argument("--both-orders", action="store_true",
                    help="also ask with the two words swapped, to expose order bias")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    with PAIRS.open(encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    if args.limit:
        # Stratify so a small sample still contains both sections, and rank by
        # a seeded hash of the pair_id rather than slicing: [:per] made the
        # "same" subset silently depend on input order, and hash-ranking also
        # nests -- a --limit 30 sample is always inside the --limit 60 one.
        import hashlib
        by_label: dict[str, list[dict]] = collections.defaultdict(list)
        for p_ in pairs:
            by_label[p_["label"]].append(p_)
        per = max(1, args.limit // max(1, len(by_label)))
        rank = lambda row: int(hashlib.sha256(  # noqa: E731
            f"13|{row['label']}|{row['pair_id']}".encode()).hexdigest()[:8], 16)
        pairs = [row for label in sorted(by_label)
                 for row in sorted(by_label[label], key=rank)[:per]]
    orders = ("ab", "ba") if args.both_orders else ("ab",)
    cache = load_cache()

    results: dict[str, dict] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows: list[dict] = []
        asked = cached = failed = 0
        for p in pairs:
            for order in orders:
                a, b = (p["a"], p["b"]) if order == "ab" else (p["b"], p["a"])
                key = (model, p["pair_id"], order)
                if key in cache:
                    raw, cached = cache[key], cached + 1
                else:
                    try:
                        raw = ask_with_retries(
                            model, PROMPT.format(a=a, b=b), args.timeout)
                        asked += 1
                    except (urllib.error.URLError, TimeoutError, OSError) as exc:
                        failed += 1
                        raw = ""
                        if failed <= 3:
                            print(f"    {p['pair_id']} failed after retries "
                                  f"({type(exc).__name__})")
                    if raw:
                        append_cache({"model": model, "pair_id": p["pair_id"],
                                      "order": order, "raw": raw})
                        cache[key] = raw
                rows.append({**p, "order": order, "raw": raw, "answer": parse(raw)})
            if failed > 20:
                print(f"    giving up on {model}: {failed} failures")
                break
        if not rows:
            print(f"  {model}: no responses")
            continue
        results[model] = score(rows)
        r = results[model]
        fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  {model:<34} SAME={fmt(r['accuracy_on_SAME'])}  "
              f"DIFFERENT={fmt(r['accuracy_on_DIFFERENT'])}  "
              f"balanced={fmt(r['balanced_accuracy'])}  "
              f"says-SAME={fmt(r['answers_SAME_rate'])}  "
              f"order-flip={fmt(r['order_flip_rate'])}  "
              f"(new={asked} cached={cached} failed={failed})")

    if not results:
        print("no results; is `ollama serve` running?")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "level2_inference.json").write_text(
        json.dumps({"n_pairs": len(pairs), "orders": list(orders),
                    "models": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = [
        "# Level 2 — can the model tell a real variant from a look-alike?",
        "",
        f"{len(pairs)} labelled pairs, orders asked: {', '.join(orders)}.",
        "",
        "A model answering SAME to everything scores 1.000 on the SAME column and",
        "0.000 on DIFFERENT, so read the two together. `says-SAME` exposes that bias",
        "directly.",
        "",
        f"Base rate of SAME in the test set: {sum(p['label'] == 'SAME' for p in pairs) / len(pairs):.3f}. "
        "A `says-SAME` rate above that is a bias toward calling things the same word.",
        "",
        "| model | acc. on SAME | acc. on DIFFERENT | balanced | says-SAME | order-flip | unparseable |",
        "|---|---|---|---|---|---|---|",
    ]
    cell = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    for m, r in sorted(results.items(),
                       key=lambda kv: -(kv[1]["balanced_accuracy"] or -1)):
        lines.append(f"| {m} | {cell(r['accuracy_on_SAME'])} "
                     f"| {cell(r['accuracy_on_DIFFERENT'])} "
                     f"| **{cell(r['balanced_accuracy'])}** "
                     f"| {cell(r['answers_SAME_rate'])} "
                     f"| {cell(r['order_flip_rate'])} "
                     f"| {r['unparseable_rate']:.3f} |")
    (OUT_DIR / "level2_inference.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    print(f"\nwrote {(OUT_DIR / 'level2_inference.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
