"""Dataset B: FusioN-synthesized Egyptian sentences around fixed anchors.

Per anchor word: N teacher models draft one sentence each (tau=0.3), a fusor
merges the drafts (Egyptian-Arabizi fusion prompt), mechanical gates verify
the anchor survived byte-identical and the language did not drift, and the
difflib contribution audit records what the fusor actually did.

Output uses the SAME schema as data/generated/training_sentences.jsonl, so
accepted sentences flow through build_dataset_a.py's respeller unchanged --
prose is generated here, orthography stays ruled there.

The fusor is a CONFIG FIELD, not a decision: the 3-arm pilot
(qwen3.8:27b-mlx / command-r:35b / zai:glm-5.3) runs this script three
times against the same teacher pool. Cloud/API fusors produce
committed-audited (not regenerable) data; that trade is recorded in every
row's metadata.

Teachers/fusors are Ollama by default; prefix with "zai:", "deepseek:", or
"openrouter:" to route through those API clients instead. Added
2026-08-25 to test whether a teacher pool with genuinely large members
(not just a large fusor) changes fusion's outcome against an external
reference -- see results/fusion_pilot.md's "teacher ceiling" finding for
why this matters: a frontier-scale fusor still lost to deepseek-v4-pro
almost every time when built from a teacher pool capped at 30B, because
fusion can only recombine what its drafts contain.

    .venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_b.py \
        --smoke 3 --fusor "qwen3.8:27b-mlx" \
        --teachers "llama3.1:8b,gemma4:e2b,openrouter:nvidia/nemotron-3.5-lightning:free,zai:glm-5.3-flash,deepseek:deepseek-v4-flash,openrouter:minimax/minimax-m3:free"

v2 (final dataset): anchors are the TRAIN side of the held-out split only,
each written in K distinct scenarios (every scenario is one more context for
the "different word" signal); jobs are planned up front so an interrupted
run resumes with --resume instead of appending duplicates:

    .venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_b.py \
        --split data/stimulus/nilechat/word_split.tsv --scenarios-per-anchor 4 \
        --fusor "zai:glm-5.3" \
        --teachers "llama3.1:8b,gemma4:e2b,nemotron-3-nano:30b,deepseek:deepseek-v4-flash,zai:glm-5.3-flash" \
        --out data/generated/v2/fused_sentences.jsonl --audit data/generated/v2/fused_audit.jsonl

--plan mode (v2 expansion, tens of thousands of jobs): the jobs come from a
FROZEN plan written by plan_fusion_v2.py and are never re-planned. They run
in batches (teacher-major prefetch within a batch, then the fusor, then
assembly), and a checkpoint JSON beside the audit is rewritten after every
batch. A spent API quota (the glm teacher and the glm fusor share ONE Z.AI
coding-plan quota) stops the run after the current job with status
"paused_quota:<model>" and exit code 75; a model outage (5 failed calls in
a row) pauses as "paused_model_down:<model>" with exit code 76;
run_fusion_v2.sh sleeps and resumes on both. --resume skips every job id already in the audit and refuses a
different plan, output, teacher pool or fusor:

    .venv/bin/python experiments/d5_cohere_track/training_data/build_dataset_b.py \
        --plan data/generated/v3/fusion_plan.jsonl --batch-size 200 \
        --fusor "zai:glm-5.3" \
        --teachers "llama3.1:8b,gemma4:e2b,nemotron-3-nano:30b,deepseek:deepseek-v4-flash,zai:glm-5.3-flash" \
        --out data/generated/v3/fused_sentences.jsonl --audit data/generated/v3/fused_audit.jsonl

Exit codes (--plan): 0 done, 75 paused on quota (resume later), 3 pass
finished but some jobs failed (resume retries them), 2 refused to start,
1 stopped (fatal API error, a model that keeps failing, or a crash),
130 interrupted (Ctrl-C / SIGTERM).
"""

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from franco.corpus import tokens, word_strings
from franco_eval.fusion import (SCENARIOS, Anchor, check_sentence,
                                contribution, fusion_prompt, teacher_prompt)
from franco_eval import deepseek_client, openrouter_client, zai_client
from franco_eval.fusion_jobs import (EXIT_REFUSED, EmptyReply, LockHeld, Plan,
                                     PlanError, PlannedJob, PlanRun,
                                     ResumeRefused, RunGuard, RunStopped,
                                     StaleOutputError, check_resume,
                                     checkpoint_path, completed_job_ids,
                                     completed_keys, ensure_fresh, load_plan,
                                     lock_path, plan_jobs, read_checkpoint,
                                     repair_tail, run_lock, status_exit_code,
                                     utc_now, write_checkpoint)
from franco_eval.model_clients import (FatalAPIError, GenConfig, JsonlCache,
                                       QuotaExhausted, chat_with_retries)

# One scenario per anchor, shared by every teacher AND the fusor for that
# anchor (see fusion.SCENARIOS) -- seeded so a re-run picks the same
# scenarios for the same anchors.
SCENARIO_SEED = 29

TRAPS = ROOT / "data" / "stimulus" / "nilechat" / "trap_votes.tsv"
WORDFREQ = ROOT / "data" / "generated" / "nilechat_wordfreq.tsv"
OUT = ROOT / "data" / "generated" / "training_b_sentences.jsonl"
AUDIT = ROOT / "data" / "generated" / "training_b_audit.jsonl"
CACHE = ROOT / "data" / "generated" / "fusion_cache.jsonl"

# num_ctx pinned well above the longest prompt (teacher/fusor prompts are a
# few hundred tokens) -- some Ollama packages default to a huge window
# (command-r:35b ships 131072), and allocating that KV cache for an 80-token
# completion pushed every call past the timeout.
TEACHER_CFG = GenConfig(temperature=0.3, num_predict=80, seed=13, num_ctx=4096)
FUSOR_CFG = GenConfig(temperature=0.3, num_predict=80, seed=13, num_ctx=4096)

# a stop in progress must reach the batch loop, never be eaten as "one
# teacher failed" -- that would write jobs the stop degraded
STOP_ERRORS = (RunStopped, QuotaExhausted, FatalAPIError)
# consecutive failed calls (distinct prompts) that mark a model as down
MAX_STREAK = 5


def load_train_side(split: pathlib.Path | None) -> set[str] | None:
    """Train-side words of the held-out split, or None (pilot: no split)."""
    if split is None:
        return None
    with split.open(encoding="utf-8") as handle:
        return {row["word"] for row in csv.DictReader(handle, delimiter="\t")
                if row["side"] == "train"}


def anchors_from_traps(limit: int,
                       train_side: set[str] | None = None) -> list[Anchor]:
    """Council-confirmed traps carry Arabic + glosses -- the richest anchors.

    Each confirmed pair yields TWO anchors (each side used correctly in its
    own sentence): the anti-over-normalization training signal. With a
    split, only pairs on its train side -- a pair never straddles it.
    """
    out: list[Anchor] = []
    if not TRAPS.exists():
        return out
    with TRAPS.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["valid"] != "true":
                continue
            if train_side is not None and row["word_type"] not in train_side:
                continue
            out.append(Anchor(row["word_type"], row["arabic_word"],
                              row.get("meaning_en_word", ""), "trap_contrast"))
            out.append(Anchor(row["trap"], row["arabic_trap"],
                              row.get("meaning_en_trap", ""), "trap_contrast"))
            if len(out) >= limit:
                break
    return out[:limit]


def clean_reply(raw: str) -> str:
    """Models decorate; keep the first plausible sentence line."""
    for line in raw.strip().splitlines():
        line = line.strip().strip('"“”').strip()
        if len(line.split()) >= 4:
            return line
    return raw.strip().strip('"').strip()


def call_model(model: str, prompt: str, *, timeout: int, cfg: GenConfig) -> str:
    """One model call with the client's own retry classification.

    "zai:<model>" / "deepseek:<model>" / "openrouter:<model>" route through
    their own API clients instead of Ollama -- non-Ollama teachers/fusors,
    used to bring frontier-tier voices into an otherwise-local pool.
    """
    if model.startswith("zai:"):
        return zai_client.chat_with_retries(
            model.removeprefix("zai:"), prompt, timeout=timeout,
            temperature=cfg.temperature, num_predict=cfg.num_predict)
    if model.startswith("deepseek:"):
        return deepseek_client.chat_with_retries(
            model.removeprefix("deepseek:"), prompt, timeout=timeout,
            temperature=cfg.temperature, num_predict=cfg.num_predict)
    if model.startswith("openrouter:"):
        return openrouter_client.chat_with_retries(
            model.removeprefix("openrouter:"), prompt, timeout=timeout,
            temperature=cfg.temperature, num_predict=cfg.num_predict)
    return chat_with_retries(model, prompt, timeout=timeout, cfg=cfg)


def make_asker(cache: JsonlCache, timeout: int, latency: dict, *,
               call=None, guard: RunGuard | None = None):
    """ask(model, role, anchor_word, prompt, cfg) -> raw reply, cached.

    Uncached call durations are recorded per model in `latency` -- the
    throughput number the user's no-unattended-runs rule asks for.

    The cache key (model, f"{role}:{anchor_word}", prompt_sha) is FROZEN:
    fusion_cache.jsonl holds paid-for answers under exactly this key.

    With a guard (--plan mode) every failure is tallied, an empty reply is
    a failure, and once the guard has stopped no NEW call is made (cached
    answers are still served, so completed work is salvaged).
    """
    lock = threading.Lock()

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def ask(model: str, role: str, anchor_word: str, prompt: str,
            cfg: GenConfig) -> str:
        record = {"model": model, "rendering_id": f"{role}:{anchor_word}",
                  "prompt_sha": sha(prompt)}
        with lock:
            cached = cache.get(record)
        if cached is not None:
            return cached["raw"]
        if guard is not None:
            guard.check()
        started = time.monotonic()
        try:
            raw = (call or call_model)(model, prompt, timeout=timeout, cfg=cfg)
            if guard is not None and not raw:
                raise EmptyReply(f"{model} returned an empty reply")
        except Exception as exc:
            if guard is not None:
                guard.failure(model, exc, key=f"{record['rendering_id']}:{record['prompt_sha']}")
            raise
        if guard is not None:
            guard.success(model)
        record["raw"] = raw
        with lock:
            latency.setdefault(model, []).append(time.monotonic() - started)
            cache.put(record)
        return raw

    return ask


def draft_all(ask, teachers: list[str], anchor: Anchor,
              scenario: str, *, reraise: tuple = ()) -> list[tuple[str, str]]:
    """[(teacher, draft)] for every teacher that answered.

    Pairs, not a bare draft list: when a teacher fails, the survivors'
    indices shift, and the old `teachers[index]` credited the wrong model.
    `reraise` (--plan mode: STOP_ERRORS) are never swallowed.
    """
    drafted = []
    for teacher in teachers:
        try:
            drafted.append((teacher, clean_reply(
                ask(teacher, "teacher", anchor.word,
                    teacher_prompt(anchor, scenario), TEACHER_CFG))))
        except reraise:
            raise
        except Exception as exc:  # one teacher down must not kill the anchor
            print(f"  teacher {teacher} failed on {anchor.word}: {exc}")
    return drafted


def is_api(model: str) -> bool:
    return model.startswith(("zai:", "deepseek:", "openrouter:"))


def _run_all(model: str, calls, workers: int) -> None:
    """Run zero-arg calls: threaded for an API model, in order for Ollama.

    Failures are swallowed here on purpose -- this only warms the cache;
    the assembly loop retries and records every failure it still hits.
    """
    # a heartbeat every 50 calls: a long silent phase is indistinguishable
    # from a hung one (an earlier overnight attempt was called off for it)
    progress = {"done": 0, "started": time.monotonic()}
    lock = threading.Lock()

    def safe(call):
        try:
            call()
        except Exception:  # retried and recorded by the assembly loop
            pass
        with lock:
            progress["done"] += 1
            if progress["done"] % 50 == 0:
                print(f"    {model}: {progress['done']}/{len(calls)} "
                      f"({time.monotonic() - progress['started']:.0f}s)", flush=True)

    if is_api(model) and workers > 1:
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            list(pool.map(safe, calls))
        finally:  # Ctrl-C must not wait for every queued call to run
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        for call in calls:
            safe(call)


def prefetch(ask, teachers: list[str], fusor: str,
             todo: list[tuple[Anchor, int]], workers: int) -> None:
    """Fill the cache teacher-by-teacher, then the fusor, before assembly.

    Job-major order (all five teachers per anchor) made Ollama swap models
    on every call: the three local teachers (24 + 7.2 + 4.9 GB) do not fit
    in 32 GB together, and a cold nemotron load is ~60s against ~2.5s warm
    (measured 2026-09-23). Teacher-major order loads each model once.
    """
    prefetch_items(ask, teachers, fusor, [(a, SCENARIOS[s]) for a, s in todo],
                   workers)


def prefetch_items(ask, teachers: list[str], fusor: str,
                   items: list[tuple[Anchor, str]], workers: int) -> None:
    """prefetch() over (anchor, scenario text) -- a --plan batch carries
    its scenario text, not an index into fusion.SCENARIOS."""
    for teacher in teachers:
        started = time.monotonic()
        _run_all(teacher, [
            (lambda a=a, s=s: ask(teacher, "teacher", a.word,
                                  teacher_prompt(a, s), TEACHER_CFG))
            for a, s in items], workers)
        print(f"  prefetch {teacher:40s} {time.monotonic() - started:7.1f}s",
              flush=True)

    def fuse(a: Anchor, s: str) -> None:
        drafted = draft_all(ask, teachers, a, s)  # cache hits now
        if len(drafted) >= 2:
            ask(fusor, "fusor", a.word,
                fusion_prompt(a, [d for _t, d in drafted], s), FUSOR_CFG)

    started = time.monotonic()
    _run_all(fusor, [(lambda a=a, s=s: fuse(a, s)) for a, s in items], workers)
    print(f"  prefetch {fusor + ' (fusor)':40s} {time.monotonic() - started:7.1f}s",
          flush=True)


def sentence_row(fused: str, anchor: Anchor, scenario_index: int,
                 fusor: str) -> dict:
    """Dataset A's sentence schema, so the respeller takes it unchanged."""
    # spans for the respeller: the anchor word is a span; other pool
    # coverage is decided by the builder exactly as for corpus text
    spans = [{"word": w, "start": offset, "end": offset + len(w)}
             for w, offset, _c in tokens(fused)
             if w == anchor.word and offset != 0]
    return {
        # stable across resumes: one sentence per (anchor, scenario)
        "sentence_id": f"fs_{anchor.word}_{scenario_index:02d}",
        "doc_id": f"fusion:{fusor}",
        "line_index": 0, "sent_index": 0,
        "text": fused,
        "n_tokens": len(word_strings(fused)),
        "spans": spans,
        "provenance": "fusion_synthetic",
        "scenario_index": scenario_index,
        "anchor": {"word": anchor.word, "arabic": anchor.arabic,
                   "gloss_en": anchor.gloss_en,
                   "context_type": anchor.context_type},
    }


def best_of_n(drafted: list[tuple[str, str]], anchor: Anchor
              ) -> tuple[str | None, str | None]:
    """Best-of-N baseline for the arena benchmark: the first raw teacher
    draft (no fusion) that passes the same gates the fused output has to
    pass. No extra model call -- this is exactly what "select, don't fuse"
    means."""
    return next(((t, d) for t, d in drafted if check_sentence(d, anchor).ok),
                (None, None))


def audit_row(anchor: Anchor, scenario: str, scenario_index: int, fusor: str,
              drafted: list[tuple[str, str]], fused: str, report,
              bon: tuple[str | None, str | None]) -> dict:
    drafts = [d for _t, d in drafted]
    bon_teacher, bon_pick = bon
    return {
        "anchor": anchor.word, "arabic": anchor.arabic,
        "context_type": anchor.context_type, "scenario": scenario,
        "scenario_index": scenario_index,
        "fusor": fusor, "teachers": [t for t, _d in drafted],
        "drafts": drafts, "fused": fused,
        "gates_ok": report.ok, "gate_reasons": list(report.reasons),
        "contribution": contribution(drafts, fused),
        "bon_pick": bon_pick, "bon_teacher": bon_teacher,
    }


def report_throughput(latency: dict, wall: float, n_jobs: int,
                      remaining: int) -> None:
    """Wall time over every attempted job, failures included -- a report
    that only timed successful jobs once printed 0.0s/job for a run whose
    uncached calls all failed."""
    if not n_jobs:
        return
    per_job = wall / n_jobs
    print(f"\nthroughput: {n_jobs} jobs in {wall / 60:.1f} min = "
          f"{per_job:.1f}s per job; projected {remaining * per_job / 3600:.1f}h "
          f"for the {remaining} jobs still to run (small runs overestimate: "
          f"one cold model load each is amortised over few jobs)")
    for model, calls in sorted(latency.items()):
        print(f"  {model:45s} {len(calls):4d} calls, "
              f"{sum(calls) / len(calls):6.1f}s mean")


def parse_teachers(spec: str) -> list[str]:
    return [t.strip() for t in spec.split(",") if t.strip()]


def run_legacy(args: argparse.Namespace) -> int:
    """The pilot / --split path, unchanged: jobs planned on the fly."""
    args.out = args.out or OUT
    args.audit = args.audit or AUDIT
    try:
        ensure_fresh([args.out, args.audit], resume=args.resume)
    except StaleOutputError as exc:
        print(exc)
        return 2
    teachers = parse_teachers(args.teachers)
    train_side = load_train_side(args.split)
    anchors = anchors_from_traps(args.smoke or 10 ** 9, train_side)
    if not anchors:
        print("no confirmed-trap anchors found; is trap_votes.tsv present?")
        return 1
    jobs = plan_jobs(anchors, args.scenarios_per_anchor, seed=SCENARIO_SEED)
    done = completed_keys(args.audit) if args.resume else set()
    todo = [(a, s) for a, s in jobs if (a.word, s) not in done]
    # a smoke run projects the time for the FULL job list, not just itself
    full_jobs = len(plan_jobs(anchors_from_traps(10 ** 9, train_side),
                              args.scenarios_per_anchor, seed=SCENARIO_SEED))
    print(f"{len(jobs)} jobs ({len(anchors)} anchors x "
          f"{args.scenarios_per_anchor} scenarios); {len(done)} already done, "
          f"{len(todo)} to run", flush=True)

    latency: dict[str, list[float]] = {}
    ask = make_asker(JsonlCache(CACHE, ("model", "rendering_id", "prompt_sha")),
                     args.timeout, latency)
    run_started = time.monotonic()
    prefetch(ask, teachers, args.fusor, todo, args.workers)
    kept = 0
    outcomes = collections.Counter()
    # an anchor word in two trap pairs is planned twice; two draws of one
    # scenario would write one sentence id twice (13 did in the v2 run)
    written = ({json.loads(line)["sentence_id"] for line in args.out.open(encoding="utf-8")}
               if args.out.exists() else set())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as out, \
            args.audit.open("a", encoding="utf-8") as audit:
        for index, (anchor, scenario_index) in enumerate(todo, start=1):
            scenario = SCENARIOS[scenario_index]
            drafted = draft_all(ask, teachers, anchor, scenario)
            if len(drafted) < 2:
                outcomes["too_few_drafts"] += 1
                continue
            drafts = [d for _t, d in drafted]
            bon = best_of_n(drafted, anchor)

            try:
                fused = clean_reply(ask(args.fusor, "fusor", anchor.word,
                                        fusion_prompt(anchor, drafts, scenario),
                                        FUSOR_CFG))
            except Exception as exc:  # recorded as an outcome, run continues
                print(f"  fusor failed on {anchor.word}: {exc}")
                outcomes["fusor_error"] += 1
                continue

            report = check_sentence(fused, anchor)
            audit.write(json.dumps(audit_row(anchor, scenario, scenario_index,
                                             args.fusor, drafted, fused, report, bon),
                                   ensure_ascii=False) + "\n")
            audit.flush()
            if not report.ok:
                outcomes["gated_out"] += 1
                print(f"  [{index}/{len(todo)}] {anchor.word}#{scenario_index}: "
                      f"REJECTED {report.reasons}", flush=True)
                continue
            row = sentence_row(fused, anchor, scenario_index, args.fusor)
            if row["sentence_id"] in written:
                outcomes["duplicate_id"] += 1
                continue
            written.add(row["sentence_id"])
            kept += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  [{index}/{len(todo)}] {anchor.word}#{scenario_index}: "
                  f"OK  {fused[:70]}", flush=True)

    print(f"\nkept {kept}/{len(todo)}  outcomes: {dict(outcomes)}")
    report_throughput(latency, time.monotonic() - run_started, len(todo),
                      remaining=max(full_jobs - len(done) - len(todo), 0))
    print(f"sentences -> {args.out}\naudit    -> {args.audit}")
    return 0


# ------------------------------------------------------------------ --plan

def _read_outputs(audit: pathlib.Path, out: pathlib.Path
                  ) -> tuple[collections.Counter, set[str]]:
    """Whole-run totals (exact, from the files) and the sentence ids."""
    totals = collections.Counter(audited=0, gated_out=0, kept=0)
    if audit.exists():
        for line in audit.read_text(encoding="utf-8").splitlines():
            if line.strip():
                totals["audited"] += 1
                totals["gated_out"] += not json.loads(line)["gates_ok"]
    written = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                written.add(json.loads(line)["sentence_id"])
    totals["kept"] = len(written)
    return totals, written


def _prepare(args, plan: Plan, teachers: list[str], previous: dict | None) -> set[str]:
    """Validate a fresh start or a resume; repair a killed run's last line.
    Returns the job ids already done. Raises StaleOutputError/ResumeRefused."""
    identity = {"plan_sha": plan.sha256, "out": args.out.resolve(),
                "teachers": teachers, "fusor": args.fusor}
    if previous is None:
        # no checkpoint: these outputs are not this plan's, whatever --resume says
        ensure_fresh([args.out, args.audit], resume=False)
    else:
        if not args.resume:
            raise StaleOutputError(
                f"{checkpoint_path(args.audit)} exists (status "
                f"{previous.get('status')}); pass --resume to continue that run, "
                "or choose a fresh --audit")
        check_resume(previous, done=None, **identity)
        for path in (args.audit, args.out):
            dropped = repair_tail(path)
            if dropped:
                print(f"repaired {path}: dropped a {dropped}-byte partial last line")
    done = completed_job_ids(args.audit)
    if previous is not None:
        check_resume(previous, done=len(done), **identity)
    foreign = done - {job.job_id for job in plan.jobs}
    if foreign:
        raise ResumeRefused(f"{args.audit} holds {len(foreign)} jobs that are not "
                            f"in {plan.path} (e.g. {sorted(foreign)[:3]})")
    return done


def assemble_job(ask, job: PlannedJob, run: PlanRun, guard: RunGuard,
                 out, audit, written: set[str]) -> None:
    """Draft (cache hits after prefetch), fuse, gate, write. Raises
    RunStopped/quota/fatal: the stop wins over this job, which is then
    simply not done yet."""
    anchor, scenario, fusor = job.anchor, job.scenario, run.args.fusor
    teachers = run.teachers
    run.attempted += 1
    drafted = draft_all(ask, teachers, anchor, scenario, reraise=STOP_ERRORS)
    if guard.stop_status and len(drafted) < len(teachers):
        raise RunStopped(guard.stop_status)  # never write a job the stop degraded
    if len(drafted) < 2:
        run.fail(job, "too_few_drafts")
        return
    try:
        fused = clean_reply(ask(fusor, "fusor", anchor.word,
                                fusion_prompt(anchor, [d for _t, d in drafted], scenario),
                                FUSOR_CFG))
    except STOP_ERRORS:
        raise
    except Exception as exc:  # not audited: retried on resume
        print(f"  fusor failed on {job.job_id}: {exc}")
        run.fail(job, "fusor_error")
        return
    report = check_sentence(fused, anchor)
    progress = f"[{len(run.done) + 1}/{len(run.plan.jobs)}] {job.job_id}"
    if report.ok:
        row = sentence_row(fused, anchor, job.scenario_id, fusor)
        if row["sentence_id"] in written:
            run.outcomes["duplicate_id"] += 1
        else:
            # sentence BEFORE audit: the audit row is what marks the job
            # done, so a kill in between redoes the job (and the cached
            # answers make the redo identical) instead of losing a sentence
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            written.add(row["sentence_id"])
            run.outcomes["kept"] += 1
            run.totals["kept"] += 1
    row = audit_row(anchor, scenario, job.scenario_id, fusor, drafted, fused,
                    report, best_of_n(drafted, anchor))
    row.update(job_id=job.job_id, scenario_id=job.scenario_id)
    audit.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit.flush()
    run.done.add(job.job_id)
    run.totals["audited"] += 1
    if report.ok:
        print(f"  {progress}: OK  {fused[:70]}", flush=True)
    else:
        run.outcomes["gated_out"] += 1
        run.totals["gated_out"] += 1
        print(f"  {progress}: REJECTED {report.reasons}", flush=True)


def _run_batches(run: PlanRun, todo: list[PlannedJob], ask, guard: RunGuard,
                 written: set[str], ckpt: pathlib.Path) -> str:
    size = run.args.batch_size
    n_batches = (len(todo) + size - 1) // size
    run.args.out.parent.mkdir(parents=True, exist_ok=True)
    run.args.audit.parent.mkdir(parents=True, exist_ok=True)
    with run.args.out.open("a", encoding="utf-8") as out, \
            run.args.audit.open("a", encoding="utf-8") as audit:
        for number, start in enumerate(range(0, len(todo), size), start=1):
            batch = todo[start:start + size]
            print(f"batch {number}/{n_batches}: {len(batch)} jobs "
                  f"({len(run.done)}/{len(run.plan.jobs)} done)", flush=True)
            prefetch_items(ask, run.teachers, run.args.fusor,
                           [(job.anchor, job.scenario) for job in batch],
                           run.args.workers)
            for job in batch:
                assemble_job(ask, job, run, guard, out, audit, written)
            run.batches += 1
            if guard.stop_status:
                return guard.stop_status  # stopped in prefetch, batch salvaged
            write_checkpoint(ckpt, run.payload("running", guard))
    return "done" if len(run.done) == len(run.plan.jobs) else "incomplete"


def _run_plan_locked(args, plan: Plan, teachers: list[str], call,
                     cache_path: pathlib.Path) -> int:
    ckpt = checkpoint_path(args.audit)
    previous = read_checkpoint(ckpt)
    try:
        done = _prepare(args, plan, teachers, previous)
    except (StaleOutputError, ResumeRefused) as exc:
        print(exc)
        return EXIT_REFUSED
    totals, written = _read_outputs(args.audit, args.out)
    run = PlanRun(args, plan, teachers, previous, done, totals)
    todo = [job for job in plan.jobs if job.job_id not in done]
    guard = RunGuard(max_streak=MAX_STREAK)
    write_checkpoint(ckpt, run.payload("running", guard))
    print(f"plan {plan.path} (sha256 {plan.sha256[:12]}): {len(plan.jobs)} jobs; "
          f"{len(done)} already done, {len(todo)} to run in batches of "
          f"{args.batch_size}; checkpoint -> {ckpt}", flush=True)
    try:  # the cache load is inside: a kill during it is still checkpointed
        ask = make_asker(JsonlCache(cache_path, ("model", "rendering_id", "prompt_sha")),
                         args.timeout, run.latency, call=call, guard=guard)
        status = _run_batches(run, todo, ask, guard, written, ckpt)
    except STOP_ERRORS as exc:
        # a quota/fatal error raised by a LIVE call in assembly arrives here
        # as itself, not as RunStopped; the asker already told the guard
        status = guard.stop_status or f"failed:{type(exc).__name__}"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # recorded in the checkpoint, then reported
        traceback.print_exc()
        status = f"failed:{type(exc).__name__}"
        run.crash = {"model": None, "kind": "crash", "type": type(exc).__name__,
                     "message": str(exc)[:500], "at": utc_now()}
    write_checkpoint(ckpt, run.payload(status, guard, final=True))
    print(f"\nstatus {status}: {len(run.done)}/{len(plan.jobs)} jobs done; "
          f"this invocation {dict(run.outcomes)}")
    if guard.last_error and status != "done":
        print(f"last error: {guard.last_error['model']}: {guard.last_error['message']}")
    report_throughput(run.latency, time.monotonic() - run.started, run.attempted,
                      remaining=len(plan.jobs) - len(run.done))
    print(f"sentences -> {args.out}\naudit    -> {args.audit}\ncheckpoint -> {ckpt}")
    return status_exit_code(status)


def run_plan(args: argparse.Namespace, *, call=None,
             cache_path: pathlib.Path | None = None) -> int:
    """--plan mode. `call` and `cache_path` exist for the test suite's fake
    models; a real run uses call_model and the shared fusion cache."""
    conflicts = [flag for flag, used in (("--split", args.split is not None),
                                         ("--smoke", args.smoke != 0),
                                         ("--scenarios-per-anchor",
                                          args.scenarios_per_anchor != 1))
                 if used]
    if conflicts:
        print(f"--plan fixes the job list; drop {', '.join(conflicts)}")
        return EXIT_REFUSED
    if args.out is None or args.audit is None:
        print("--plan needs explicit --out and --audit (the pilot defaults are "
              "someone else's files)")
        return EXIT_REFUSED
    if args.batch_size < 1:
        print("--batch-size must be at least 1")
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan)
    except (OSError, PlanError) as exc:
        print(f"cannot use plan {args.plan}: {exc}")
        return EXIT_REFUSED
    if not plan.sidecar_checked:
        print(f"note: {args.plan.name}.sha256 missing; the checkpoint pins the sha instead")
    try:
        with run_lock(lock_path(args.audit)):
            return _run_plan_locked(args, plan, parse_teachers(args.teachers), call,
                                    cache_path or CACHE)
    except LockHeld as exc:
        print(f"{exc}: is another run_fusion_v2.sh loop already running?")
        return EXIT_REFUSED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fusor", required=True)
    parser.add_argument("--teachers", required=True,
                        help="comma-separated Ollama tags (paper setting: 5)")
    parser.add_argument("--smoke", type=int, default=0,
                        help="anchors to run (0 = all confirmed-trap anchors)")
    parser.add_argument("--split", type=pathlib.Path, default=None,
                        help="v2: use only the train side of this split")
    parser.add_argument("--scenarios-per-anchor", type=int, default=1)
    parser.add_argument("--plan", type=pathlib.Path, default=None,
                        help="frozen plan from plan_fusion_v2.py (batched, checkpointed)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="--plan: jobs per teacher-major prefetch + checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="continue a run into its existing --out/--audit")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent calls for API teachers/fusor")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help=f"default (pilot only): {OUT.relative_to(ROOT)}")
    parser.add_argument("--audit", type=pathlib.Path, default=None,
                        help=f"default (pilot only): {AUDIT.relative_to(ROOT)}")
    return parser


def _terminate(signum, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")  # checkpointed as "interrupted"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan is not None:
        signal.signal(signal.SIGTERM, _terminate)
        return run_plan(args)
    return run_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())
