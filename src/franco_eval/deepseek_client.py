"""DeepSeek API client -- the external frontier reference for the Dataset B
arena benchmark (FusioN's fused output vs. a single strong model's direct
attempt, paralleling the FusioN paper's own FusioN-vs-Gemini comparison).

Reads the API key from .env under DEEPSEEK= (a separate line from Z.AI's
API= -- different provider, different key). Like glm-5.3, DeepSeek's V4
models default to reasoning that eats a token budget before writing any
answer.

reasoning_effort is "high", not "low": low effort was tried first (to
mirror the fix already needed for zai_client.py) and produced a real,
reproducible regression -- under our multi-constraint prompt (exact
anchor word, exact spelling, exact position, ASCII-only, word-count
range), low reasoning effort made the model drop the "Latin script only"
rule and answer in Arabic script instead, even though it can write fluent
Franco easily in an unconstrained chat. High effort tracks every
constraint correctly (confirmed live across 3/3 test anchors) at the cost
of more reasoning tokens, so the budget is padded further to compensate.

max_tokens was a flat 4500 (raised to 12000 on 2026-09-26, see MAX_TOKENS),
not scaled from num_predict: deepseek-
v4-pro (the larger, 49B-active-param reasoning-tier sibling of v4-flash)
was measured spending 1100-2750 reasoning tokens on this same one-sentence
task before writing any content -- the 800-token budget sized for
v4-flash left v4-pro with finish_reason="length" and empty content on
40/45 calls. One shared, generous budget avoids re-tuning per model;
v4-flash simply stops early and the extra headroom costs nothing (DeepSeek
bills completion tokens actually used, not the max_tokens cap).

Error classification (added 2026-09-25) follows DeepSeek's official
tables, which key on the HTTP status alone:
https://api-docs.deepseek.com/quick_start/error_codes -- 400 Invalid
Format, 401 Authentication Fails, 402 Insufficient Balance ("you have run
out of balance"), 422 Invalid Parameters, 429 Rate Limit Reached, 500
Server Error, 503 Server Overloaded; and
https://api-docs.deepseek.com/quick_start/rate_limit -- 429 is the
per-account CONCURRENCY limit, so it is retried, never paused on. Only 402
is a spent quota.

Stdlib only.
"""

import json
import pathlib
import urllib.error
import urllib.request

from franco_eval.model_clients import (ClassifiedHTTPError, FatalAPIError, TruncatedReply,
                                       call_with_retries, error_payload,
                                       status_class)

# 2026-09-26: 4,500 was too tight. Measured on real teacher prompts,
# deepseek-v4-flash spends 1,109-4,463 reasoning tokens on ONE sentence, so
# calls came back empty (the pilot's "model down" stops) or cut mid-word.
# DeepSeek bills tokens used, not the cap, so headroom costs nothing on the
# calls that finish early.
MAX_TOKENS = 12000
ENDPOINT = "https://api.deepseek.com/chat/completions"
ENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"


def _load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no DEEPSEEK= line found in {ENV_PATH}")


def classify(exc: urllib.error.HTTPError, model: str) -> ClassifiedHTTPError:
    body, payload = error_payload(exc)
    error = payload.get("error") if payload else None
    message = error.get("message") if isinstance(error, dict) else None
    # 402 = the prepaid balance is spent. Fatal, not a quota pause (user
    # decision 2026-09-26): unlike Z.AI's 5-hour window it never refills by
    # waiting, so a pause would retry hourly forever. The run stops, keeps
    # what it finished, and the checkpoint names the reason.
    cls = FatalAPIError if exc.code == 402 else status_class(exc.code)
    text = message or body.decode("utf-8", "replace")[:300] or exc.reason
    return cls(exc, provider="deepseek", model=model, error_code=None,
               message=str(text), body=body)


def chat(model: str, prompt: str, *, timeout: int, temperature: float = 0.3,
         num_predict: int = 80) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max(num_predict * 10, MAX_TOKENS),
        "reasoning_effort": "high",
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
    choice = payload["choices"][0]
    if choice.get("finish_reason") == "length":
        # the budget ran out mid-reasoning or mid-answer: a cut-off sentence
        # must never become a draft, and an empty one is not an answer
        details = (payload.get("usage") or {}).get("completion_tokens_details") or {}
        raise TruncatedReply(f"{model}: hit max_tokens ({details.get('reasoning_tokens')} "
                             f"reasoning tokens of {MAX_TOKENS})")
    return choice["message"]["content"]


def chat_with_retries(model: str, prompt: str, *, timeout: int, tries: int = 3,
                      backoff: float = 2.0, **kwargs) -> str:
    """Transient failures retried; quota/fatal/rejected raised at once."""
    return call_with_retries(
        lambda: chat(model, prompt, timeout=timeout, **kwargs),
        tries=tries, backoff=backoff)
