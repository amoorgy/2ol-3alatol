"""OpenRouter API client -- one credential giving access to several teacher
models that would otherwise each need their own provider integration: the
free nvidia/nemotron-3.5-lightning:free tier, z-ai/glm-5.3-flash, and
qwen/qwen3-235b-a22b (one of the FusioN paper's own five named teachers,
arXiv:2510.00931 -- the closest available match when trying to bring one
teacher genuinely comparable to what the paper used into our own pool).

Reads the API key from .env under OPENROUTER=. model is the OpenRouter
catalog string verbatim (e.g. "nvidia/nemotron-3.5-lightning:free"),
passed straight through -- OpenRouter's own routing picks the provider.

Error classification (added 2026-09-25) follows OpenRouter's official
docs: https://openrouter.ai/docs/api_reference/limits ("Handling 402
errors") -- 402 is insufficient account credits or an exhausted per-key
credit limit (quota), EXCEPT error.metadata.limit_source ==
"openrouter_in_flight_budget", which the docs call transient (wait for
Retry-After, then retry); and
https://openrouter.ai/docs/api_reference/errors-and-debugging -- 401 bad
key, 403 permission/guardrail/moderation, 408 timeout, 429 rate limited
(retry with backoff, honor Retry-After), 502/503 provider down/overloaded.

One UNDOCUMENTED signature is also treated as quota: HTTP 403 with
"Key limit exceeded" in the message, observed live on 2026-08-27
(changes.md) when this repo's key hit its total limit.

Stdlib only.
"""

import json
import pathlib
import urllib.error
import urllib.request

from franco_eval.model_clients import (ClassifiedHTTPError, QuotaExhausted,
                                       TransientHTTPError, call_with_retries,
                                       error_payload, status_class)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATH = pathlib.Path(__file__).resolve().parents[2] / ".env"


def _load_key() -> str:
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"no OPENROUTER= line found in {ENV_PATH}")


def classify(exc: urllib.error.HTTPError, model: str) -> ClassifiedHTTPError:
    body, payload = error_payload(exc)
    error = payload.get("error") if payload else None
    error = error if isinstance(error, dict) else {}
    message = str(error.get("message") or body.decode("utf-8", "replace")[:300]
                  or exc.reason)
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    source = metadata.get("limit_source")
    if exc.code == 402:
        cls = (TransientHTTPError if source == "openrouter_in_flight_budget"
               else QuotaExhausted)
    elif exc.code == 403 and "key limit exceeded" in message.lower():
        cls = QuotaExhausted  # observed live, see module docstring
    else:
        cls = status_class(exc.code)
    return cls(exc, provider="openrouter", model=model,
               error_code=source or metadata.get("error_type"),
               message=message, body=body)


def chat(model: str, prompt: str, *, timeout: int, temperature: float = 0.3,
         num_predict: int = 80) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max(num_predict * 10, 800),
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
