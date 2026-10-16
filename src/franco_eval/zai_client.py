"""Z.AI GLM Coding Plan client -- the one credentialed exception to the
project's no-network/no-env-var rule, scoped narrowly per CLAUDE.md's
Environment notes: used as a build_dataset_b.py fusor arm (originally only
when the Ollama cloud-subscription route glm-5.2:cloud was unavailable),
and, added 2026-08-25, as a run_arena_judge.py external reference
("glm-4.6" -- a genuinely weaker single-shot reference than deepseek-v4-pro,
without needing a new provider since it's the same client, same endpoint).

Uses the coding plan's DEDICATED endpoint (api.z.ai/api/coding/paas/v4),
not the generic pay-as-you-go endpoint (api.z.ai/api/paas/v4) -- the coding
plan's quota lives behind this path specifically. The generic endpoint
returns an "insufficient balance" error even with a valid, funded
coding-plan key, because it checks a different balance entirely.

Reads the API key from .env (the ONLY place in this repo that does).

Error classification (added 2026-09-25) follows Z.AI's official error
table, https://docs.z.ai/api-reference/api-code : errors are JSON
{"error": {"code": "<business code>", "message": ...}} and the business
code, not the HTTP status, is what separates a spent quota from plain
throttling -- both arrive as HTTP 429. The coding plan has a 5-hour and a
weekly limit (https://docs.z.ai/devpack/overview, "Usage Credit
Allowance"), and "once the quota is used up, you'll need to wait until the
next 5-hour cycle" (https://docs.z.ai/devpack/faq).

Stdlib only.
"""

import json
import pathlib
import urllib.error
import urllib.request

from franco_eval.model_clients import (ClassifiedHTTPError, FatalAPIError,
                                       QuotaExhausted, RequestRejected,
                                       TransientHTTPError, call_with_retries,
                                       error_payload, status_class)

ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"

# Business codes from https://docs.z.ai/api-reference/api-code (fetched
# 2026-09-25). Every quota code below is HTTP 429.
QUOTA_CODES = frozenset({
    "1113",  # Insufficient balance or no resource package
    "1308",  # Usage limit reached for {number} {unit}; resets at {time}
    "1310",  # Weekly/Monthly Limit Exhausted; resets at {time}
    "1316", "1318", "1320",  # usage limit for the past 5 hours
    "1317", "1319", "1321",  # usage limit for the past 7 days
})
# Waiting does not fix these: auth, an expired/ineligible plan, a model or
# API that does not exist, or a fair-use restriction that needs an appeal.
FATAL_CODES = frozenset({
    "1000", "1001", "1003", "1005",  # authentication (HTTP 401)
    "1211",  # Unknown Model
    "1220",  # no permission to access the API
    "1221", "1222",  # API taken offline / does not exist
    "1309",  # GLM Coding Plan package has expired
    "1311",  # plan does not include access to this model
    "1313",  # Fair Usage Policy: frequency limited until an appeal
    "1314",  # enterprise package expired
    "1315",  # key limited to enterprise coding package scenarios
})
TRANSIENT_CODES = frozenset({
    "1200", "1230", "1234",  # API call / process / network error (HTTP 500)
    "1302",  # Rate limit reached for requests (HTTP 429)
    "1305",  # service temporarily overloaded (HTTP 429)
})
REJECTED_CODES = frozenset({
    "1210", "1212", "1213", "1214", "1215",  # invalid / missing parameters
    "1261",  # Prompt too long
    "1301",  # unsafe or sensitive content (this prompt only)
})


def _load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("API="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no API= line found in {ENV_PATH}")


def classify(exc: urllib.error.HTTPError, model: str) -> ClassifiedHTTPError:
    body, payload = error_payload(exc)
    error = payload.get("error") if payload else None
    code = message = None
    if isinstance(error, dict):
        code = None if error.get("code") is None else str(error["code"])
        message = error.get("message")
    if code in QUOTA_CODES:
        cls = QuotaExhausted
    elif code in FATAL_CODES:
        cls = FatalAPIError
    elif code in TRANSIENT_CODES:
        cls = TransientHTTPError
    elif code in REJECTED_CODES:
        cls = RequestRejected
    else:  # undocumented code: an unknown 429 is retried, not paused on
        cls = status_class(exc.code)
    text = message or body.decode("utf-8", "replace")[:300] or exc.reason
    return cls(exc, provider="zai", model=model, error_code=code,
               message=str(text), body=body)


def chat(model: str, prompt: str, *, timeout: int, temperature: float = 0.3,
         num_predict: int = 80) -> str:
    # GLM models' thinking is mandatory -- it cannot be turned off, and a
    # too-small max_tokens budget gets consumed entirely by the reasoning
    # trace before any answer is written, even when the actual reasoning
    # need is tiny: glm-4.6 spent all 300/300 tokens on reasoning and
    # returned empty content, then succeeded at max_tokens=1500 using only
    # ~35 reasoning tokens -- a too-tight cap causes degenerate behavior,
    # not a genuine need for hundreds of reasoning tokens. reasoning_effort
    # is Z.AI's documented way to shrink (not eliminate) the trace; the
    # budget floor is padded generously so it costs nothing for models/
    # prompts that need less (billed on tokens actually used, not the cap).
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max(num_predict * 10, 2500),
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_load_key()}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise classify(exc, model) from exc
    return payload["choices"][0]["message"]["content"]


def chat_with_retries(model: str, prompt: str, *, timeout: int, tries: int = 3,
                      backoff: float = 2.0, **kwargs) -> str:
    """Transient failures retried; quota/fatal/rejected raised at once."""
    return call_with_retries(
        lambda: chat(model, prompt, timeout=timeout, **kwargs),
        tries=tries, backoff=backoff)
