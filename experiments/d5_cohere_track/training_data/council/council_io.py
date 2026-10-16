"""Shared I/O and statistics for the five-lens council tools. stdlib only.

A council has two directories. The chunk dir holds c<N>.json files, each a
JSON array of items keyed (sid, rid). The verdict dir holds
<lens>_c<N>.json files, one per judge, each a JSON array of
{sid, rid, v, why} with one row per item of the chunk. An item's final verdict
is ACCEPT when >= MIN_AGREE of the five lenses vote v = true -- the same rule
merge_council.merge_training applies.

Model and effort are not in the verdict files. council_jobs.py --stamp puts
them in <verdict dir>/panel.meta, which no *.json glob picks up.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass
from statistics import NormalDist

ROOT = pathlib.Path(__file__).resolve().parents[4]
RESULTS_DIR = ROOT / "experiments" / "d5_cohere_track" / "results"

# sorted, like merge_council's sorted(judges); must match the LENSES keys of
# fused_council.js / pair_council.js (a test checks this)
COUNCILS: dict[str, tuple[str, ...]] = {
    "fused": ("anchor", "coherence", "degeneracy", "fluency", "register"),
    "pairs": ("coherence", "identity", "plausibility", "register", "skeptic"),
}
MIN_AGREE = 4
STAMP = "panel.meta"
# User rulings 2026-09-26: bulk judges were Sonnet medium until calibration
# showed it misses 96% of Opus x-high's pair rejections (kappa 0.055); Opus on
# LOW effort agrees far better (kappa 0.451 on fused c0) at ~60k tokens/judge,
# so Opus low is the bulk panel (user bar: >=4/5). Opus x-high stays for
# calibration and audits; sonnet/medium stays listed for the c18-c22 panel.
POLICY_PANELS = {("opus", "low"): "bulk", ("opus", "xhigh"): "calibration/audit",
                 ("sonnet", "medium"): "bulk (retired: pairs c18-c22 only)"}
Z95 = NormalDist().inv_cdf(0.975)

_CHUNK_NAME = re.compile(r"c(\d+)")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

Key = tuple[str, str]


class CouncilDataError(ValueError):
    """Chunk or verdict data that breaks the council contract."""


# ------------------------------------------------------------------ chunks

def item_key(item: dict) -> Key:
    return (item["sid"], item["rid"])


def load_chunks(chunk_dir: pathlib.Path) -> dict[str, list[dict]]:
    """c<N> -> items, in numeric chunk order. Other files are ignored."""
    chunk_dir = pathlib.Path(chunk_dir)
    if not chunk_dir.is_dir():
        raise CouncilDataError(f"chunk dir not found: {chunk_dir}")
    named = sorted((int(m.group(1)), p) for p in chunk_dir.glob("c*.json")
                   if (m := _CHUNK_NAME.fullmatch(p.stem)))
    if not named:
        raise CouncilDataError(f"no c<N>.json chunks in {chunk_dir}")
    chunks: dict[str, list[dict]] = {}
    seen: dict[Key, str] = {}
    for _, path in named:
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CouncilDataError(f"{path}: unparsable chunk ({exc})") from exc
        if not isinstance(items, list) or not all(
                isinstance(i, dict) and isinstance(i.get("sid"), str)
                and isinstance(i.get("rid"), str) for i in items):
            raise CouncilDataError(f"{path}: not a JSON array of items with sid/rid")
        for item in items:
            key = item_key(item)
            if key in seen:
                raise CouncilDataError(
                    f"item {key} is in both {seen[key]} and {path.stem}")
            seen[key] = path.stem
        chunks[path.stem] = items
    return chunks


def select_chunks(chunks: dict[str, list[dict]], names: str | None) -> dict[str, list[dict]]:
    if not names:
        return chunks
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in wanted if n not in chunks]
    if unknown:
        raise CouncilDataError(f"unknown chunks: {', '.join(unknown)}")
    return {n: chunks[n] for n in chunks if n in wanted}


def resolve_lenses(council: str, override: str | None = None) -> tuple[str, ...]:
    if council not in COUNCILS:
        raise CouncilDataError(f"unknown council {council!r}; expected one of {sorted(COUNCILS)}")
    lenses = COUNCILS[council]
    if not override:
        return lenses
    wanted = tuple(x.strip() for x in override.split(",") if x.strip())
    unknown = [x for x in wanted if x not in lenses]
    if unknown:
        raise CouncilDataError(f"lenses {unknown} are not in the {council} council {lenses}")
    return tuple(x for x in lenses if x in wanted)


# ---------------------------------------------------------------- verdicts

def verdict_path(verdict_dir: pathlib.Path, lens: str, chunk: str) -> pathlib.Path:
    return pathlib.Path(verdict_dir) / f"{lens}_{chunk}.json"


def read_rows(path: pathlib.Path) -> tuple[list[dict] | None, str | None]:
    """(rows, None) for a well-formed judge file, else (None, reason)."""
    if not path.exists():
        return None, "missing"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"unparsable ({exc.__class__.__name__})"
    if not isinstance(rows, list):
        return None, "malformed: not a JSON array"
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("sid"), str)
                or not isinstance(row.get("rid"), str)):
            return None, "malformed: a row without string sid/rid"
        if not isinstance(row.get("v"), bool):
            # "false" is truthy: merged as-is it would count as an accept
            return None, "malformed: non-boolean v"
    return rows, None


def index_rows(rows: list[dict], wanted: set[Key]) -> tuple[dict[Key, dict], set[Key], int]:
    """(key -> row for wanted keys, keys with contradicting rows, stray rows)."""
    indexed: dict[Key, dict] = {}
    conflicts: set[Key] = set()
    strays = 0
    for row in rows:
        key = item_key(row)
        if key not in wanted:
            strays += 1
        elif key in indexed and indexed[key]["v"] != row["v"]:
            conflicts.add(key)
        else:
            indexed.setdefault(key, row)
    for key in conflicts:
        indexed.pop(key, None)
    return indexed, conflicts, strays


@dataclass(frozen=True)
class JobStatus:
    complete: bool
    reason: str
    strays: int = 0


def job_status(path: pathlib.Path, items: list[dict]) -> JobStatus:
    """A job is complete when its file gives every chunk item one boolean v."""
    rows, problem = read_rows(path)
    if problem:
        return JobStatus(False, problem)
    wanted = [item_key(i) for i in items]
    indexed, conflicts, strays = index_rows(rows, set(wanted))
    if conflicts:
        return JobStatus(False, f"conflicting duplicate verdicts for {len(conflicts)} items",
                         strays)
    missing = sum(1 for k in wanted if k not in indexed)
    if missing:
        if len(rows) < len(items):
            return JobStatus(False, f"short: {len(rows)}/{len(items)} rows", strays)
        return JobStatus(False, f"mismatch: {missing} of {len(items)} chunk items have "
                                f"no verdict", strays)
    return JobStatus(True, "", strays)


def pending_jobs(chunks: dict[str, list[dict]], verdict_dir: pathlib.Path,
                 lenses: tuple[str, ...]) -> tuple[list[dict], int]:
    """([{chunk, lens, reason}] in chunk-then-lens order, total stray rows)."""
    pending, strays = [], 0
    for chunk, items in chunks.items():
        for lens in lenses:
            status = job_status(verdict_path(verdict_dir, lens, chunk), items)
            strays += status.strays
            if not status.complete:
                pending.append({"chunk": chunk, "lens": lens, "reason": status.reason})
    return pending, strays


Panel = dict[str, dict[Key, dict]]


def load_panel(verdict_dir: pathlib.Path, chunks: dict[str, list[dict]],
               lenses: tuple[str, ...]) -> Panel:
    """lens -> {key -> row}, from every well-formed file (partial files count).

    Only chunk items are kept; contradicting duplicates are dropped, so an item
    with no clean vote on some lens has no final verdict.
    """
    panel: Panel = {lens: {} for lens in lenses}
    for chunk, items in chunks.items():
        wanted = {item_key(i) for i in items}
        for lens in lenses:
            rows, problem = read_rows(verdict_path(verdict_dir, lens, chunk))
            if problem:
                continue
            indexed, _, _ = index_rows(rows, wanted)
            panel[lens].update(indexed)
    return panel


@dataclass(frozen=True)
class Verdict:
    votes: tuple[bool, ...]      # in lens order
    n_valid: int
    accept: bool


def final_verdict(panel: Panel, key: Key, lenses: tuple[str, ...],
                  min_agree: int = MIN_AGREE) -> Verdict | None:
    """None unless every lens voted on the item."""
    rows = [panel[lens].get(key) for lens in lenses]
    if any(r is None for r in rows):
        return None
    votes = tuple(r["v"] for r in rows)
    n_valid = sum(votes)
    return Verdict(votes, n_valid, n_valid >= min_agree)


def dissent(verdict: Verdict, lenses: tuple[str, ...]) -> list[str]:
    return [lens for lens, vote in zip(lenses, verdict.votes) if not vote]


# ---------------------------------------------------------------- statistics

def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two raters on binary labels.

    None when undefined: no items, or both raters constant on the same label
    (chance agreement 1). Reporting 1.0 there would claim agreement beyond
    chance that the data cannot show.
    """
    if len(a) != len(b):
        raise ValueError("raters judged different numbers of items")
    if not a:
        return None
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    if p_e >= 1.0:
        return None
    return round((p_o - p_e) / (1 - p_e), 4)


def wilson_interval(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval; n may be fractional (an effective sample size)."""
    if n <= 0:
        raise ValueError("Wilson interval needs n > 0")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # At 0 or n successes the bound is exactly 0 or 1; computing it leaves
    # libm-dependent residue (2.8e-17), so results would differ by platform.
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return lo, hi


def stratified_rate(strata: list[tuple[int, int, int]]) -> dict:
    """Stratified proportion over (N_h, n_h, x_h) with a Wilson interval.

    rate = sum_h W_h x_h / n_h with W_h = N_h / N. The interval is Wilson at
    Kish's effective sample size n_eff = 1 / sum_h (W_h^2 / n_h), which equals
    n under proportional allocation and shrinks when small strata are
    oversampled. No finite-population correction (conservative).
    """
    total = sum(N for N, _, _ in strata)
    if total <= 0:
        raise ValueError("empty population")
    for N, n, x in strata:
        if N > 0 and n <= 0:
            raise ValueError("every non-empty stratum needs at least one sampled item")
        if not 0 <= x <= n:
            raise ValueError(f"stratum count {x} outside [0, {n}]")
    live = [(N / total, n, x) for N, n, x in strata if N > 0]
    rate = sum(w * x / n for w, n, x in live)
    n_eff = 1 / sum(w * w / n for w, n, _ in live)
    lo, hi = wilson_interval(rate * n_eff, n_eff)
    return {"rate": rate, "n_eff": n_eff, "wilson95": (lo, hi)}


def proportion(k: int, n: int) -> dict:
    """{k, n, rate, wilson95} with 4-decimal rounding; rate None when n == 0."""
    if n == 0:
        return {"k": k, "n": 0, "rate": None, "wilson95": None}
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


# ---------------------------------------------------------------- records

def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def chunk_hashes(chunk_dir: pathlib.Path, chunks: dict[str, list[dict]]) -> dict[str, str]:
    return {c: sha256_file(pathlib.Path(chunk_dir) / f"{c}.json") for c in chunks}


def verdict_hashes(verdict_dir: pathlib.Path, chunks: dict[str, list[dict]],
                   lenses: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for chunk in chunks:
        for lens in lenses:
            path = verdict_path(verdict_dir, lens, chunk)
            if path.exists():
                out[path.name] = sha256_file(path)
    return out


def rel(path: pathlib.Path) -> str:
    """Repo-relative path when inside the repo, so reports are portable."""
    resolved = pathlib.Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_stamp(verdict_dir: pathlib.Path) -> dict | None:
    path = pathlib.Path(verdict_dir) / STAMP
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CouncilDataError(f"{path}: unparsable stamp ({exc})") from exc


def check_name(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise CouncilDataError(f"--name {name!r}: use letters, digits, '_', '-', '.'")
    return name


def pct(x: float | None, places: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{places}f}%"


def fmt(x: float | None, places: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{places}f}"


def interval(p: dict) -> str:
    if p["rate"] is None:
        return "n/a (n = 0)"
    lo, hi = p["wilson95"]
    return f"{p['k']}/{p['n']} = {pct(p['rate'])} [{pct(lo)}, {pct(hi)}]"


def md_cell(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    return text.replace("|", "\\|")
