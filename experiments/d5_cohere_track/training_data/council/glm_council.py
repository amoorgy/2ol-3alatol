"""Run a five-lens council with GLM (Z.AI coding plan) instead of Claude judges.

User decision 2026-09-26: judging moves to glm-5.3 on the flat Z.AI plan so
councils stop costing Claude tokens. The CRITERIA do not change: the lens
texts and the framing are read from the committed council scripts
(fused_council.js / pair_council.js) -- everything before their agent-only
"1. Read the chunk file" step -- and only the delivery differs: items are
inlined in batches and GLM replies with a JSON array.

Output is the same verdict-file format as the Claude councils
(<out-dir>/<lens>_<chunk>.json, rows {sid, rid, v, why}), so council_jobs,
merge_council, compare_councils and sample_audit work unchanged. Resumable:
a complete verdict file is skipped. A spent Z.AI quota sleeps and retries
the same file; a fatal account error exits 1.

Caveat, measured before use (compare_councils against Opus x-high): glm-5.3
is also the FUSOR, so on fused sentences it judges its own output.

    .venv/bin/python experiments/d5_cohere_track/training_data/council/glm_council.py \\
        --council fused --chunk-dir data/generated/v2/council/fused_chunks \\
        --out-dir data/generated/v2/council/fused_votes_glm
"""

import argparse
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco_eval import zai_client  # noqa: E402
from franco_eval.model_clients import FatalAPIError, QuotaExhausted  # noqa: E402

SCRIPTS = {"fused": HERE / "fused_council.js", "pairs": HERE / "pair_council.js"}
MODEL = "glm-5.3"
BATCH_RETRIES = 3


def load_council(council: str) -> tuple[str, dict[str, str]]:
    """(framing template with a {lens} slot, lens -> criterion text)."""
    source = SCRIPTS[council].read_text(encoding="utf-8")
    block = re.search(r"^const LENSES = \{\n(.*?)^\}", source, re.S | re.M).group(1)
    lenses = {}
    for key, text in re.findall(r"^\s+(\w+): '((?:[^'\\]|\\.)*)',?\s*$", block, re.M):
        lenses[key] = text.replace("\\'", "'").replace("\\\\", "\\")
    template = re.search(r"^const prompt = \(chunk, lens, a\) => `(.*?)`", source, re.S | re.M).group(1)
    framing = template.split("\n\n1. Read the chunk file")[0]
    return framing.replace("${LENSES[lens]}", "{lens}"), lenses


def build_prompt(framing: str, lens_text: str, items: list[dict]) -> str:
    """Items are keyed by a small index `i` (1..N), not by sid: GLM garbled
    near-identical sids (fs_f2dan_00 -> fs_f2dan_0) until a batch failed all
    retries, while small integers are copied reliably."""
    rows = [{"i": n, **{k: v for k, v in item.items() if k not in ("sid", "rid")}}
            for n, item in enumerate(items, start=1)]
    keys = ", ".join(rows[0])
    payload = json.dumps(rows, ensure_ascii=False, indent=0)
    return (framing.replace("{lens}", lens_text)
            + f"\n\nItems (a JSON array; keys: {keys}):\n{payload}\n\n"
            "For EVERY item decide v = true (passes your lens) or v = false. \"why\" is "
            "required when v is false (at most 20 words) and may be \"\" when true.\n"
            "Reply with ONLY a JSON array: one object per item, with exactly the "
            "keys i, v, why (i is the item's number, copied unchanged). No other text.")


def parse_verdicts(reply: str, items: list[dict]) -> list[dict]:
    """Rows for `items`, in item order; ValueError on anything malformed."""
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in reply")
    rows = json.loads(reply[start:end + 1])
    if not isinstance(rows, list):
        raise ValueError("reply is not a JSON array")
    by_index: dict[int, dict] = {}
    for row in rows:
        try:
            by_index[int(row["i"])] = row
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"verdict without a usable index: {str(row)[:80]}") from None
    if set(by_index) != set(range(1, len(items) + 1)) or len(rows) != len(items):
        raise ValueError(f"expected indices 1..{len(items)}, got {sorted(by_index)[:5]}... ({len(rows)} rows)")
    out = []
    for n, item in enumerate(items, start=1):
        row = by_index[n]
        v = row.get("v")
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        if not isinstance(v, bool):
            raise ValueError(f"non-boolean v for item {n}: {v!r}")
        out.append({"sid": item["sid"], "rid": item["rid"], "v": v,
                    "why": str(row.get("why") or "")[:300]})
    return out


def judge_chunk(items: list[dict], framing: str, lens_text: str, ask, *,
                batch_size: int, workers: int) -> list[dict]:
    """All verdicts for one chunk and lens; a malformed batch is re-asked."""
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def run(batch: list[dict]) -> list[dict]:
        last: Exception | None = None
        for _ in range(BATCH_RETRIES):
            try:
                return parse_verdicts(ask(build_prompt(framing, lens_text, batch)), batch)
            except ValueError as exc:  # malformed reply: ask again
                last = exc
        raise ValueError(f"batch starting {batch[0]['sid']}: {last}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, batches))
    return [row for batch in results for row in batch]


def is_complete(path: pathlib.Path, items: list[dict]) -> bool:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return ([r.get("sid") for r in rows] == [i["sid"] for i in items]
            and all(isinstance(r.get("v"), bool) for r in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--council", required=True, choices=sorted(SCRIPTS))
    parser.add_argument("--chunk-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-minutes", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    framing, lenses = load_council(args.council)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def ask(prompt: str) -> str:
        return zai_client.chat_with_retries(MODEL, prompt, timeout=args.timeout,
                                            temperature=0.0, num_predict=1200)

    chunks = sorted(args.chunk_dir.glob("c*.json"), key=lambda p: int(p.stem[1:]))
    jobs = [(c, lens) for c in chunks for lens in sorted(lenses)]
    done = failed = 0
    for chunk_path, lens in jobs:
        items = json.loads(chunk_path.read_text(encoding="utf-8"))
        out = args.out_dir / f"{lens}_{chunk_path.stem}.json"
        if is_complete(out, items):
            done += 1
            continue
        while True:
            try:
                rows = judge_chunk(items, framing, lenses[lens], ask,
                                   batch_size=args.batch_size, workers=args.workers)
                break
            except QuotaExhausted as exc:
                print(f"quota: {exc}; sleeping {args.wait_minutes} min", flush=True)
                time.sleep(args.wait_minutes * 60)
            except FatalAPIError as exc:
                print(f"FATAL: {exc}", flush=True)
                return 1
            except ValueError as exc:  # a batch failed every re-ask
                rows = None
                print(f"[skip] {out.name}: {exc} -- left pending for the next pass", flush=True)
                break
        if rows is None:
            failed += 1
            continue
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        done += 1
        rejected = sum(1 for r in rows if not r["v"])
        print(f"[{done}/{len(jobs)}] {out.name}: {len(rows)} judged, {rejected} false", flush=True)
    if failed:
        print(f"INCOMPLETE: {done}/{len(jobs)} verdict files; {failed} left pending -- run again")
        return 3
    print(f"council complete: {done}/{len(jobs)} verdict files in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
