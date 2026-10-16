"""Job planning and output safety for long Dataset B fusion runs.

A fusion job is (anchor, scenario index). v2 writes several contexts per
trap anchor -- every scenario is one more sentence carrying the "different
word" signal -- so jobs are planned up front, deterministically, and a run
can stop and resume without redoing or duplicating a job.

Output safety exists because of a real incident: the b2 pilot reused an
output filename, and append mode silently stacked a 20-anchor run onto 505
stale rows from an abandoned overnight attempt. A non-empty output is now
refused unless the caller explicitly resumes.

The v2 expansion (2026-09-25) adds a FROZEN plan -- a JSONL of jobs drawn
once from the anchor bank x scenario bank and pinned by its sha256 -- plus
what a multi-day unattended run needs around it: a checkpoint written
after every batch, a lock so two runners never share one audit, repair of
a half-written last line after a kill, and a RunGuard that turns a spent
API quota into a clean pause instead of thousands of degraded jobs.

Stdlib only.
"""

import argparse
import collections
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import random
import shlex
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from franco_eval.fusion import SCENARIOS, Anchor
from franco_eval.model_clients import (FatalAPIError, QuotaExhausted,
                                       RequestRejected, is_transient)

ANCHOR_BANK_HEADER = ("word", "arabic", "gloss_en", "context_type", "source",
                      "count", "doc_freq", "band", "priority")


class StaleOutputError(RuntimeError):
    pass


class PlanError(ValueError):
    """A bank or plan file breaks its contract."""


class ResumeRefused(RuntimeError):
    """A resume would mix two different runs into one audit."""


class LockHeld(RuntimeError):
    """Another runner already writes this audit."""


class RunStopped(RuntimeError):
    """Raised instead of a new model call once a run has decided to stop."""


class EmptyReply(RuntimeError):
    """A model answered with no content (the reasoning-budget failure)."""


# --------------------------------------------------------------- pilot / v2

def plan_jobs(anchors: Sequence[Anchor], k: int, *,
              seed: int) -> list[tuple[Anchor, int]]:
    """[(anchor, scenario_index)] -- k distinct scenarios per anchor.

    k=1 draws with randrange per anchor, exactly as the pilot did, so a
    pilot anchor keeps the scenario its cached drafts were written for.
    """
    if k > len(SCENARIOS):
        raise ValueError(f"k={k} exceeds the {len(SCENARIOS)} scenarios")
    rng = random.Random(seed)
    jobs: list[tuple[Anchor, int]] = []
    for anchor in anchors:
        if k == 1:
            jobs.append((anchor, rng.randrange(len(SCENARIOS))))
        else:
            jobs.extend((anchor, s) for s in rng.sample(range(len(SCENARIOS)), k))
    return jobs


def ensure_fresh(paths: Sequence[pathlib.Path], *, resume: bool) -> None:
    """Refuse to append into a non-empty output unless resuming."""
    if resume:
        return
    stale = [p for p in paths if p.exists() and p.stat().st_size > 0]
    if stale:
        raise StaleOutputError(
            f"refusing to append into non-empty {', '.join(map(str, stale))}; "
            "pass --resume to continue that run, or choose a fresh path")


def _audit_rows(audit: pathlib.Path) -> Iterator[dict]:
    with audit.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def completed_keys(audit: pathlib.Path) -> set[tuple[str, int]]:
    """(anchor word, scenario index) of every job the audit already holds."""
    if not audit.exists():
        return set()
    return {(row["anchor"], row["scenario_index"]) for row in _audit_rows(audit)}


# ------------------------------------------------------------- frozen plan

def job_id(word: str, scenario_id: int) -> str:
    return f"{word}#{scenario_id}"


def completed_job_ids(audit: pathlib.Path) -> set[str]:
    """job ids in an audit. Rows written before plans existed carry no
    job_id; their scenario_index is the scenario id (ids 0-15 of the
    scenario bank are fusion.SCENARIOS verbatim, in order)."""
    if not audit.exists():
        return set()
    return {row.get("job_id") or job_id(row["anchor"], row["scenario_index"])
            for row in _audit_rows(audit)}


@dataclass(frozen=True)
class PlannedJob:
    job_id: str
    anchor: Anchor
    scenario_id: int
    scenario: str
    source: str = ""
    band: str = ""

    def to_json(self) -> dict:
        return {"job_id": self.job_id, "scenario_id": self.scenario_id,
                "scenario": self.scenario, "source": self.source,
                "band": self.band,
                "anchor": {"word": self.anchor.word, "arabic": self.anchor.arabic,
                           "gloss_en": self.anchor.gloss_en,
                           "context_type": self.anchor.context_type}}


@dataclass(frozen=True)
class PlanDraw:
    jobs: tuple[PlannedJob, ...]
    stats: dict


@dataclass(frozen=True)
class Plan:
    path: pathlib.Path
    sha256: str
    jobs: tuple[PlannedJob, ...]
    sidecar_checked: bool


def read_anchor_bank(path: pathlib.Path) -> list[dict[str, str]]:
    """Rows of data/stimulus/v2_anchor_bank.tsv in file (= priority) order.

    Raw TSV (the bank writer refuses tabs/newlines inside a field), so it
    is read with QUOTE_NONE, like scenario_bank.read_bank.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = tuple(next(reader, ()))
        if header != ANCHOR_BANK_HEADER:
            raise PlanError(f"{path}: header must be {ANCHOR_BANK_HEADER}, got {header}")
        rows = []
        for line_no, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) != len(ANCHOR_BANK_HEADER):
                raise PlanError(f"{path}:{line_no}: {len(fields)} fields, "
                                f"want {len(ANCHOR_BANK_HEADER)}")
            row = dict(zip(ANCHOR_BANK_HEADER, fields))
            if not row["word"] or row["word"] != "".join(row["word"].split()):
                raise PlanError(f"{path}:{line_no}: bad anchor word {row['word']!r}")
            rows.append(row)
    return rows


def check_pinned_scenarios(scenarios: Sequence[tuple[int, str]]) -> None:
    """Ids 0-15 must be fusion.SCENARIOS verbatim: the v2 audit's
    scenario_index, the exclusion keys and the shared cache all rely on it."""
    text = dict(scenarios)
    for index, expected in enumerate(SCENARIOS):
        if text.get(index) != expected:
            raise PlanError(f"scenario {index} is not fusion.SCENARIOS[{index}] "
                            f"verbatim (or not accepted): {text.get(index)!r}")


def _unique_anchor_rows(rows: Sequence[dict], limit: int) -> tuple[list[dict], int, list[str]]:
    """First `limit` distinct words (0 = all); the first row of a word wins."""
    kept: dict[str, dict] = {}
    skipped, conflicts = 0, []
    for row in rows:
        word = row["word"]
        if word in kept:
            skipped += 1
            first = kept[word]
            if ((first.get("arabic", ""), first.get("gloss_en", ""))
                    != (row.get("arabic", ""), row.get("gloss_en", ""))
                    and word not in conflicts):
                conflicts.append(word)
            continue
        if limit and len(kept) >= limit:
            continue  # keep scanning only to count duplicates of kept words
        kept[word] = row
    return list(kept.values()), skipped, conflicts


def draw_v2_jobs(anchor_rows: Sequence[dict], scenarios: Sequence[tuple[int, str]],
                 *, n_anchors: int, k: int, seed: int,
                 exclude: frozenset[str] | set[str] = frozenset()) -> PlanDraw:
    """K distinct accepted scenarios per anchor, anchor-major in bank order.

    Each anchor has its OWN rng, seeded from (seed, word): raising
    --anchors, or a row inserted above an anchor, never reshuffles the
    scenarios already drawn for it. Excluded job ids (done in an earlier
    audit) are dropped after the draw, never redrawn -- exclusion is a pure
    filter, so it cannot perturb any other job.
    """
    check_pinned_scenarios(scenarios)
    ids = sorted(sid for sid, _text in scenarios)
    text = dict(scenarios)
    if not 1 <= k <= len(ids):
        raise PlanError(f"--scenarios-per-anchor {k} outside [1, {len(ids)}] "
                        f"(the accepted scenarios)")
    rows, skipped, conflicts = _unique_anchor_rows(anchor_rows, n_anchors)
    jobs: list[PlannedJob] = []
    drawn = excluded = 0
    for row in rows:
        anchor = Anchor(row["word"], row.get("arabic", ""), row.get("gloss_en", ""),
                        row.get("context_type") or "general")
        rng = random.Random(f"{seed}:{anchor.word}")
        for sid in rng.sample(ids, k):
            drawn += 1
            jid = job_id(anchor.word, sid)
            if jid in exclude:
                excluded += 1
                continue
            jobs.append(PlannedJob(jid, anchor, sid, text[sid],
                                   row.get("source", ""), row.get("band", "")))
    stats = {"anchors_requested": n_anchors, "anchors_used": len(rows),
             "duplicate_rows_skipped": skipped, "conflicting_duplicates": conflicts,
             "scenarios_accepted": len(ids), "jobs_drawn": drawn,
             "jobs_excluded": excluded, "jobs_planned": len(jobs)}
    return PlanDraw(tuple(jobs), stats)


def plan_bytes(jobs: Sequence[PlannedJob]) -> bytes:
    """Canonical serialization: same jobs, same bytes, same sha256."""
    return "".join(json.dumps(job.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
                   for job in jobs).encode("utf-8")


def sidecar_path(plan: pathlib.Path) -> pathlib.Path:
    return plan.with_name(plan.name + ".sha256")


def write_plan(path: pathlib.Path, jobs: Sequence[PlannedJob]) -> str:
    """Write plan + `shasum -a 256 -c`-compatible sidecar; never overwrite."""
    data = plan_bytes(jobs)
    sha = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:  # FileExistsError: a plan is frozen
        handle.write(data)
    sidecar_path(path).write_text(f"{sha}  {path.name}\n", encoding="utf-8")
    return sha


def _job_from_json(row: dict, where: str) -> PlannedJob:
    try:
        anchor = row["anchor"]
        job = PlannedJob(row["job_id"], Anchor(anchor["word"], anchor["arabic"],
                                               anchor["gloss_en"], anchor["context_type"]),
                         int(row["scenario_id"]), row["scenario"],
                         row.get("source", ""), row.get("band", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"{where}: malformed job ({exc!r})") from exc
    if job.job_id != job_id(job.anchor.word, job.scenario_id) or not job.scenario:
        raise PlanError(f"{where}: job_id {job.job_id!r} does not match its "
                        f"anchor/scenario")
    return job


def load_plan(path: pathlib.Path) -> Plan:
    """Parse and verify a frozen plan. An edit after planning (sidecar
    sha mismatch), a duplicate job id or a malformed row is refused."""
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    sidecar = sidecar_path(path)
    if sidecar.exists():
        recorded = sidecar.read_text(encoding="utf-8").split()[0]
        if recorded != sha:
            raise PlanError(f"{path} was edited after planning: sha256 {sha} "
                            f"!= {recorded} in {sidecar.name}")
    jobs, seen = [], set()
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise PlanError(f"{path}:{line_no}: not JSON") from exc
        job = _job_from_json(row, f"{path}:{line_no}")
        if job.job_id in seen:
            raise PlanError(f"{path}:{line_no}: duplicate job_id {job.job_id!r}")
        seen.add(job.job_id)
        jobs.append(job)
    return Plan(path, sha, tuple(jobs), sidecar.exists())


# --------------------------------------------------- checkpoint and resume

def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def checkpoint_path(audit: pathlib.Path) -> pathlib.Path:
    return audit.with_name(audit.stem + ".checkpoint.json")


def lock_path(audit: pathlib.Path) -> pathlib.Path:
    return audit.with_name(audit.stem + ".lock")


def read_checkpoint(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_checkpoint(path: pathlib.Path, payload: dict) -> None:
    """Atomic: a kill mid-write leaves the previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def repair_tail(path: pathlib.Path) -> int:
    """Cut a half-written last line (a kill mid-append); bytes dropped.
    Only a runner holding the lock may call this on its own outputs."""
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    with path.open("r+b") as handle:
        handle.truncate(keep)
    return len(data) - keep


@contextlib.contextmanager
def run_lock(path: pathlib.Path) -> Iterator[None]:
    """Exclusive, non-blocking flock; released when the process exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockHeld(f"another runner holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def check_resume(checkpoint: dict, *, plan_sha: str, out: pathlib.Path,
                 teachers: Sequence[str], fusor: str, done: int | None) -> None:
    """A resume must continue THE SAME run: same frozen plan, same output,
    same teacher pool and fusor (matched conditions across the dataset),
    and an audit that has not lost rows since the last checkpoint
    (done=None skips that last check, for the call made before the audit
    can be read)."""
    problems = []
    if checkpoint["plan"]["sha256"] != plan_sha:
        problems.append(f"plan sha256 {plan_sha} != checkpoint's "
                        f"{checkpoint['plan']['sha256']}")
    if pathlib.Path(checkpoint["out"]) != out:
        problems.append(f"--out {out} != checkpoint's {checkpoint['out']}")
    if list(checkpoint["teachers"]) != list(teachers):
        problems.append(f"--teachers {list(teachers)} != checkpoint's "
                        f"{checkpoint['teachers']}")
    if checkpoint["fusor"] != fusor:
        problems.append(f"--fusor {fusor} != checkpoint's {checkpoint['fusor']}")
    if done is not None and done < checkpoint.get("jobs_done", 0):
        problems.append(f"audit holds {done} jobs but the checkpoint recorded "
                        f"{checkpoint['jobs_done']}: rows were lost")
    if problems:
        raise ResumeRefused("refusing to resume: " + "; ".join(problems))


# ---------------------------------------------------------------- RunGuard

def failure_kind(exc: BaseException) -> str:
    if isinstance(exc, QuotaExhausted):
        return "quota"
    if isinstance(exc, FatalAPIError):
        return "fatal"
    if isinstance(exc, RequestRejected):
        return "rejected"
    if isinstance(exc, EmptyReply):
        return "empty"
    return "transient" if is_transient(exc) else "error"


class RunGuard:
    """Shared stop signal and per-model failure tallies for one run.

    Stops on the first of: a spent quota (-> pause and resume later), a
    fatal account/config error, or `max_streak` consecutive failed calls
    of one model on DISTINCT prompts (a dead Ollama, a model that only
    times out or only returns empty content). That last one is a PAUSE too
    (user ruling 2026-09-26): deepseek-v4-flash went empty for a few minutes
    twice in the pilot and each stop cost hours; run_fusion_v2.sh waits
    briefly and resumes, and gives up only after repeated outages with no
    progress. Rejections -- a content
    filter on one anchor -- never count: an anchor-major plan sends the
    same word several times in a row. Re-asking an already-failed prompt
    (the fusor phase re-drafts every job) is not a new failure either.
    """

    def __init__(self, max_streak: int = 5) -> None:
        self.max_streak = max_streak
        self.stop_status: str | None = None
        self.last_error: dict | None = None
        self.failures: dict[str, collections.Counter] = {}
        self._streak: collections.Counter = collections.Counter()
        self._failed_keys: dict[str, set] = collections.defaultdict(set)
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.stop_status:
            raise RunStopped(self.stop_status)

    def success(self, model: str) -> None:
        with self._lock:
            self._streak[model] = 0

    def failure(self, model: str, exc: BaseException, *, key: str) -> None:
        kind = failure_kind(exc)
        with self._lock:
            self.failures.setdefault(model, collections.Counter())[kind] += 1
            self.last_error = {"model": model, "kind": kind,
                               "type": type(exc).__name__,
                               "message": str(exc)[:500], "at": utc_now()}
            if kind in ("transient", "empty", "error") and key not in self._failed_keys[model]:
                self._failed_keys[model].add(key)
                self._streak[model] += 1
            if self.stop_status:
                return  # the first reason is the one the checkpoint reports
            if kind == "quota":
                self.stop_status = f"paused_quota:{model}"
            elif kind == "fatal":
                self.stop_status = f"failed_fatal:{model}"
            elif self._streak[model] >= self.max_streak:
                self.stop_status = f"paused_model_down:{model}"


# ------------------------------------------------------ checkpoint record

# --plan exit codes; 75 is EX_TEMPFAIL (quota pause) and 76 a model-outage
# pause -- the two codes run_fusion_v2.sh resumes on, with different waits
EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 1, 2, 3
EXIT_PAUSED, EXIT_MODEL_DOWN, EXIT_INTERRUPTED = 75, 76, 130
MAX_FAILED_JOBS_LISTED = 200
MAX_HISTORY = 100


def status_exit_code(status: str) -> int:
    if status == "done":
        return EXIT_DONE
    if status.startswith("paused_quota:"):
        return EXIT_PAUSED
    if status.startswith("paused_model_down:"):
        return EXIT_MODEL_DOWN
    if status == "incomplete":
        return EXIT_INCOMPLETE
    if status == "interrupted":
        return EXIT_INTERRUPTED
    return EXIT_FAILED  # failed_fatal:*, failed:*


def merge_latency(previous: dict, calls: dict[str, list[float]]) -> dict:
    merged = {m: dict(v) for m, v in previous.items()}
    for model, durations in calls.items():
        entry = merged.setdefault(model, {"calls": 0, "total_s": 0.0})
        entry["calls"] += len(durations)
        entry["total_s"] = round(entry["total_s"] + sum(durations), 3)
    for entry in merged.values():
        entry["mean_s"] = round(entry["total_s"] / entry["calls"], 3) if entry["calls"] else None
    return merged


@dataclass
class PlanRun:
    """One --plan invocation: what its checkpoint reports."""

    args: argparse.Namespace
    plan: Plan
    teachers: list[str]
    previous: dict | None
    done: set[str]
    totals: collections.Counter
    latency: dict[str, list[float]] = field(default_factory=dict)
    outcomes: collections.Counter = field(default_factory=collections.Counter)
    failed_jobs: dict[str, str] = field(default_factory=dict)
    crash: dict | None = None
    batches: int = 0
    attempted: int = 0
    started: float = field(default_factory=time.monotonic)
    invocation_started_at: str = field(default_factory=utc_now)

    def fail(self, job: PlannedJob, reason: str) -> None:
        self.outcomes[reason] += 1
        self.failed_jobs[job.job_id] = reason

    def payload(self, status: str, guard: RunGuard, *, final: bool = False) -> dict:
        prev = self.previous or {}
        wall = time.monotonic() - self.started
        remaining = len(self.plan.jobs) - len(self.done)
        per_job = wall / self.attempted if self.attempted else None
        last_error = self.crash or guard.last_error
        history = list(prev.get("history", []))
        if final:
            history.append({"invocation": prev.get("invocations", 0) + 1,
                            "started_at": self.invocation_started_at,
                            "ended_at": utc_now(), "status": status,
                            "jobs_done": len(self.done), "last_error": last_error})
        return {
            "status": status,
            "exit_code": None if status == "running" else status_exit_code(status),
            "plan": {"path": str(self.plan.path.resolve()), "sha256": self.plan.sha256,
                     "jobs": len(self.plan.jobs),
                     "sidecar_checked": self.plan.sidecar_checked},
            "out": str(self.args.out.resolve()), "audit": str(self.args.audit.resolve()),
            "fusor": self.args.fusor, "teachers": self.teachers,
            "batch_size": self.args.batch_size, "workers": self.args.workers,
            "timeout": self.args.timeout,
            "jobs_total": len(self.plan.jobs), "jobs_done": len(self.done),
            "jobs_remaining": remaining,
            "totals": dict(self.totals),
            "outcomes_this_invocation": dict(self.outcomes),
            "failed_jobs": dict(list(self.failed_jobs.items())[:MAX_FAILED_JOBS_LISTED]),
            "failed_jobs_count": len(self.failed_jobs),
            "batches_this_invocation": self.batches,
            "throughput": {"jobs_attempted": self.attempted, "wall_s": round(wall, 1),
                           "s_per_job": round(per_job, 2) if per_job else None,
                           "eta_h": round(remaining * per_job / 3600, 2) if per_job else None},
            "latency": merge_latency(prev.get("latency", {}), self.latency),
            "model_failures": {m: dict(c) for m, c in sorted(guard.failures.items())},
            "last_error": last_error,
            "started_at": prev.get("started_at") or self.invocation_started_at,
            "invocation_started_at": self.invocation_started_at,
            "updated_at": utc_now(),
            "invocations": prev.get("invocations", 0) + 1,
            "quota_pauses": prev.get("quota_pauses", 0)
            + (1 if final and status.startswith("paused_quota:") else 0),
            "history": history[-MAX_HISTORY:],
            "command": shlex.join(sys.argv),
        }
