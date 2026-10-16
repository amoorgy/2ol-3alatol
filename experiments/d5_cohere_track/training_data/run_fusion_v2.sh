#!/bin/bash
# run_fusion_v2.sh -- unattended Dataset B fusion over a FROZEN plan.
#
# Survives an API quota running out mid-run (the Z.AI coding plan's 5-hour
# and weekly limits are shared by the glm teacher AND the glm fusor):
# build_dataset_b.py --plan exits 75 when a quota is spent; this loop
# sleeps --wait-minutes and resumes. Exit 0 (plan done) ends the loop; any
# other exit code ends it too, logged (the reason is in the checkpoint JSON
# beside the audit).
#
# RUN IT IN THE FOREGROUND. It detaches the loop itself, exactly once
# (nohup + caffeinate -is, in a session of its own), confirms the loop is
# alive, prints the pid, log and stop command, and returns in ~2 seconds.
# Do NOT append `&` and do NOT start it from an already-backgrounded tool
# call: the 2026-08 overnight attempt reported "done" after doing nothing
# because it was backgrounded twice. "LAUNCHED" means started, not finished.
#
#   experiments/d5_cohere_track/training_data/run_fusion_v2.sh \
#       --plan data/generated/v3/fusion_plan.jsonl \
#       --out data/generated/v3/fused_sentences.jsonl \
#       --audit data/generated/v3/fused_audit.jsonl \
#       --fusor zai:glm-5.3 \
#       --teachers "llama3.1:8b,gemma4:e2b,nemotron-3-nano:30b,deepseek:deepseek-v4-flash,zai:glm-5.3-flash" \
#       --batch-size 200 --wait-minutes 60
#
# Loop options (everything else goes to build_dataset_b.py unchanged; the
# loop always adds --resume, which starts fresh when nothing exists yet and
# refuses outputs that do not belong to this plan):
#   --wait-minutes N  sleep after a quota pause (default 60)
#   --max-pauses N    stop after N consecutive-or-not quota pauses (0 = never)
#   --outage-wait-minutes N   sleep after a model-outage pause, exit 76 (default 10)
#   --max-outage-pauses N     give up after N outage pauses in a row WITHOUT job
#                             progress (default 12; 0 = never). Progress, read
#                             from the checkpoint's jobs_done, resets the count.
#   --log FILE        default: <audit without .jsonl>.run.log
#   --foreground      run the loop in this shell (output to stdout) instead
#   --python PATH     default: <repo>/.venv/bin/python, else python3
#   --runner PATH     default: build_dataset_b.py beside this script
#
# Monitor: tail -f <log>;  cat <audit without .jsonl>.checkpoint.json
# Stop:    kill -TERM -- -<pgid printed at launch>  (the runner checkpoints
#          "interrupted"; relaunch the same command to continue)

set -u

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
HERE="$(dirname "$SELF")"
ROOT="$(cd "$HERE/../../.." && pwd)"

WAIT_MINUTES=60
MAX_PAUSES=0
OUTAGE_WAIT_MINUTES=10
MAX_OUTAGE_PAUSES=12
LOG=""
FOREGROUND=0
LOOP=0
PY=""
RUNNER="$HERE/build_dataset_b.py"
PLAN=""
AUDIT=""
PASS=()

die() { echo "run_fusion_v2.sh: $*" >&2; exit 2; }
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_fusion_v2: $*"; }
need() { [ "$1" -ge 2 ] || die "$2 needs a value"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --wait-minutes) need $# "$1"; WAIT_MINUTES="$2"; shift 2 ;;
    --max-pauses)   need $# "$1"; MAX_PAUSES="$2"; shift 2 ;;
    --outage-wait-minutes) need $# "$1"; OUTAGE_WAIT_MINUTES="$2"; shift 2 ;;
    --max-outage-pauses)   need $# "$1"; MAX_OUTAGE_PAUSES="$2"; shift 2 ;;
    --log)          need $# "$1"; LOG="$2"; shift 2 ;;
    --python)       need $# "$1"; PY="$2"; shift 2 ;;
    --runner)       need $# "$1"; RUNNER="$2"; shift 2 ;;
    --foreground)   FOREGROUND=1; shift ;;
    --_loop)        LOOP=1; shift ;;  # internal: the detached/foreground loop
    --resume)       shift ;;          # the loop always resumes
    --plan)         need $# "$1"; PLAN="$2"; PASS+=("$1" "$2"); shift 2 ;;
    --audit)        need $# "$1"; AUDIT="$2"; PASS+=("$1" "$2"); shift 2 ;;
    --*=*)          die "use '--option value', not '$1'" ;;
    *)              PASS+=("$1"); shift ;;
  esac
done

[ -n "$PLAN" ] || die "--plan is required (write one with plan_fusion_v2.py)"
[ -n "$AUDIT" ] || die "--audit is required"
case "$WAIT_MINUTES" in ''|*[!0-9]*) die "--wait-minutes must be a whole number" ;; esac
case "$MAX_PAUSES" in ''|*[!0-9]*) die "--max-pauses must be a whole number" ;; esac
case "$OUTAGE_WAIT_MINUTES" in ''|*[!0-9]*) die "--outage-wait-minutes must be a whole number" ;; esac
case "$MAX_OUTAGE_PAUSES" in ''|*[!0-9]*) die "--max-outage-pauses must be a whole number" ;; esac
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
  else PY="$(command -v python3 || true)"; fi
fi
[ -n "$PY" ] || die "no python found (pass --python)"
[ -f "$RUNNER" ] || die "runner not found: $RUNNER"

STEM="${AUDIT%.jsonl}"
[ -n "$LOG" ] || LOG="$STEM.run.log"
PIDFILE="$STEM.loop.pid"
CHECKPOINT="$STEM.checkpoint.json"
LOOP_ARGS=(--wait-minutes "$WAIT_MINUTES" --max-pauses "$MAX_PAUSES"
           --outage-wait-minutes "$OUTAGE_WAIT_MINUTES" --max-outage-pauses "$MAX_OUTAGE_PAUSES"
           --log "$LOG"
           --python "$PY" --runner "$RUNNER" "${PASS[@]}")

# jobs_done from the checkpoint, or -1 when there is none (yet)
jobs_done() {
  "$PY" -c 'import json, sys
try:
    print(int(json.load(open(sys.argv[1])).get("jobs_done", -1)))
except Exception:
    print(-1)' "$CHECKPOINT"
}

run_loop() {
  local pauses=0 n=0 code outages=0 last_done now
  last_done=$(jobs_done)
  log "loop start (pid $$): plan=$PLAN audit=$AUDIT wait=${WAIT_MINUTES}m max_pauses=$MAX_PAUSES"
  while :; do
    n=$((n + 1))
    log "invocation $n: $PY $RUNNER ${PASS[*]} --resume"
    "$PY" "$RUNNER" "${PASS[@]}" --resume
    code=$?
    case "$code" in
      0)
        log "runner exit 0: plan finished"
        return 0 ;;
      75)
        pauses=$((pauses + 1))
        if [ "$MAX_PAUSES" -gt 0 ] && [ "$pauses" -gt "$MAX_PAUSES" ]; then
          log "runner exit 75 (quota) but --max-pauses $MAX_PAUSES reached: stopping"
          return 75
        fi
        log "runner exit 75: quota pause $pauses, sleeping ${WAIT_MINUTES} min, then resuming"
        sleep $((WAIT_MINUTES * 60)) ;;
      76)
        # a model outage (5 failed calls in a row): wait briefly and resume;
        # give up only after repeated outages that completed no job at all
        now=$(jobs_done)
        if [ "$now" -gt "$last_done" ]; then outages=0; fi
        last_done=$now
        outages=$((outages + 1))
        if [ "$MAX_OUTAGE_PAUSES" -gt 0 ] && [ "$outages" -gt "$MAX_OUTAGE_PAUSES" ]; then
          log "runner exit 76 (model outage) $outages times without progress: stopping"
          return 76
        fi
        log "runner exit 76: model outage pause $outages, sleeping ${OUTAGE_WAIT_MINUTES} min, then resuming"
        sleep $((OUTAGE_WAIT_MINUTES * 60)) ;;
      *)
        log "runner exit $code: stopping (not a quota pause; see $CHECKPOINT and the runner output above)"
        return "$code" ;;
    esac
  done
}

if [ "$LOOP" = 1 ]; then
  trap 'rm -f "$PIDFILE"' EXIT
  trap 'log "loop killed (TERM)"; exit 143' TERM
  trap 'log "loop interrupted (INT)"; exit 130' INT
  run_loop
  code=$?
  log "loop end: exit $code"
  exit "$code"
fi

refuse_if_running() {
  local old
  if [ -f "$PIDFILE" ]; then
    old="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
      die "a loop is already running for $AUDIT (pid $old, $PIDFILE); stop it or wait"
    fi
  fi
}

refuse_if_running
if command -v caffeinate >/dev/null 2>&1; then
  AWAKE=(caffeinate -is)  # -i: no idle sleep, -s: no system sleep on AC power
else
  echo "run_fusion_v2.sh: caffeinate not found; the machine may sleep mid-run" >&2
  AWAKE=()
fi

if [ "$FOREGROUND" = 1 ]; then
  echo "$$" > "$PIDFILE"
  exec ${AWAKE[@]+"${AWAKE[@]}"} /bin/bash "$SELF" --_loop "${LOOP_ARGS[@]}"
fi

# Detach exactly once. setsid gives the loop its own session and process
# group (immune to the launching shell's hangup and group kills); nohup
# ignores SIGHUP; stdin/stdout/stderr leave the terminal entirely.
mkdir -p "$(dirname "$LOG")"
nohup "$PY" -c '
import os, sys
try:
    os.setsid()
except OSError:  # already a group leader (interactive job control)
    pass
os.execvp(sys.argv[1], sys.argv[1:])
' ${AWAKE[@]+"${AWAKE[@]}"} /bin/bash "$SELF" --_loop "${LOOP_ARGS[@]}" \
  >>"$LOG" 2>&1 </dev/null &
child=$!
echo "$child" > "$PIDFILE"
sleep 2
if kill -0 "$child" 2>/dev/null; then
  echo "LAUNCHED -- the loop is running in the background; it has NOT finished."
  echo "  pid/pgid   : $child"
  echo "  log        : $LOG"
  echo "  checkpoint : $CHECKPOINT"
  echo "  monitor    : tail -f '$LOG'"
  echo "  stop       : kill -TERM -- -$child"
  exit 0
fi
wait "$child"
code=$?
rm -f "$PIDFILE"
echo "the loop ended within 2 seconds (exit $code); last log lines:"
tail -n 20 "$LOG"
exit "$code"
