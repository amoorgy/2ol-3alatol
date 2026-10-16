"""Ollama chat access with retries and an append-only JSONL cache.

Extracted from run_level2_inference.py so that level-3 meaning probing does not
copy-paste it. Level 2 itself is frozen with its committed results and keeps
its own copy untouched.

Local Ollama only -- no credentials, no env vars, per the project's environment
rules. Stdlib only.

Also home of the retry CLASSIFICATION shared by every client (Ollama here,
zai/deepseek/openrouter in their own modules). Before 2026-09-25 every
client retried every URLError/OSError alike: a spent quota was hammered
three times, and a missing .env (FileNotFoundError is an OSError) too.
Now each client maps an HTTP error onto one of four classes, taken from
that provider's official error table:

  TransientHTTPError  retried with backoff (5xx, throttling, overload)
  RequestRejected     this request only; never retried (bad param, filter)
  FatalAPIError       account/config; never retried; a long run must stop
  QuotaExhausted      usage limit; never retried; a long run must pause

All four subclass urllib.error.HTTPError on purpose: callers written
before the classification existed catch URLError / OSError / Exception
(run_arena_judge.py catches (OSError, TimeoutError)) and keep working.
"""

import http.client
import io
import json
import pathlib
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, TypeVar

OLLAMA_CHAT = "http://localhost:11434/api/chat"

# A server-sent Retry-After longer than this is capped: an hour-long
# Retry-After is a quota in disguise, and the caller's pause logic -- not a
# sleeping retry loop -- is the place to wait that long.
MAX_RETRY_AFTER = 120.0

T = TypeVar("T")


class ClassifiedHTTPError(urllib.error.HTTPError):
    """An HTTP error mapped onto a provider's documented error table."""

    kind = "http_error"
    retryable = False

    def __init__(self, source: urllib.error.HTTPError, *, provider: str,
                 model: str, error_code: str | None, message: str,
                 body: bytes = b"") -> None:
        where = f"HTTP {source.code}" + (f", code {error_code}" if error_code else "")
        detail = f"{provider}:{model} {self.kind} ({where}): {message}"
        super().__init__(source.filename, source.code, detail, source.headers,
                         io.BytesIO(b""))
        self.close()  # nothing left to stream; the body is kept below
        self.body = body
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.message_text = message
        self.retry_after = retry_after_seconds(source.headers)


class TransientHTTPError(ClassifiedHTTPError):
    kind = "transient"
    retryable = True


class RequestRejected(ClassifiedHTTPError):
    kind = "rejected"


class FatalAPIError(ClassifiedHTTPError):
    kind = "fatal"


class QuotaExhausted(ClassifiedHTTPError):
    kind = "quota"


# Connection-level failures a retry can fix: dropped/refused connections,
# DNS (URLError wraps socket.gaierror), timeouts, truncated responses, and
# a 200 with a blank/non-JSON body (seen live from OpenRouter free tiers).
class TruncatedReply(RuntimeError):
    """A reasoning model spent its whole token budget and the answer was cut
    off (finish_reason "length"). Retryable: reasoning length varies a lot
    call to call (deepseek-v4-flash, one sentence: 1,109-4,463 tokens)."""


TRANSIENT_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, ssl.SSLError,
                    json.JSONDecodeError, TruncatedReply)


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ClassifiedHTTPError):
        return exc.retryable
    if isinstance(exc, urllib.error.HTTPError):  # unclassified: status decides
        return exc.code >= 500 or exc.code in (408, 429)
    return isinstance(exc, TRANSIENT_ERRORS)


def retry_after_seconds(headers) -> float | None:
    """Retry-After in seconds; the HTTP-date form is ignored (None)."""
    value = headers.get("Retry-After") if headers is not None else None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


def error_payload(exc: urllib.error.HTTPError) -> tuple[bytes, dict | None]:
    """(raw body, parsed JSON object or None) -- never raises. Closes exc:
    its body is consumed here, once."""
    try:
        body = exc.read() or b""
    except Exception:  # a body that cannot be read is simply absent
        body = b""
    finally:
        exc.close()
    try:
        payload = json.loads(body) if body else None
    except ValueError:
        payload = None
    return body, payload if isinstance(payload, dict) else None


def status_class(status: int) -> type[ClassifiedHTTPError]:
    """Fallback when a provider's body carries no code we recognise."""
    if status == 401:
        return FatalAPIError
    if status >= 500 or status in (408, 429):
        return TransientHTTPError
    return RequestRejected


def call_with_retries(call: Callable[[], T], *, tries: int = 3,
                      backoff: float = 2.0,
                      sleep: Callable[[float], None] = time.sleep) -> T:
    """call() with retries for transient failures only.

    Linear backoff (backoff * attempt), stretched to a server's Retry-After
    when one is sent (capped at MAX_RETRY_AFTER). Quota, fatal, rejected
    and local errors (a missing .env) raise on the first attempt.
    """
    for attempt in range(1, tries + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == tries or not is_transient(exc):
                raise
            delay = backoff * attempt
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
            sleep(delay)
    raise AssertionError("unreachable: the last attempt returns or raises")


def classify_ollama(exc: urllib.error.HTTPError, model: str) -> ClassifiedHTTPError:
    """docs.ollama.com/api/errors: 400 bad request, 404 model doesn't
    exist, 429 rate limit, 500/502 server. 404 is fatal: a model that is
    not pulled fails every call, and a long run must stop, not degrade."""
    body, payload = error_payload(exc)
    message = str(payload.get("error", "")) if payload else body.decode(
        "utf-8", "replace")[:300]
    cls = FatalAPIError if exc.code == 404 else status_class(exc.code)
    return cls(exc, provider="ollama", model=model, error_code=None,
               message=message, body=body)


@dataclass(frozen=True)
class GenConfig:
    """Pinned generation settings. num_predict=24 because an Arabic word is
    3-8 tokens on a Latin-heavy BPE plus possible leading quote tokens.

    num_ctx is None by default -- Ollama then uses the model's own Modelfile
    default. Some packages default to a huge window (command-r:35b ships
    131072): allocating that KV cache blew every short-prompt call past a
    240s timeout, so callers with short prompts should pin num_ctx down
    explicitly rather than pay for a context size the prompt never uses."""

    temperature: float = 0.0
    num_predict: int = 24
    seed: int = 13
    think: bool = False
    num_ctx: int | None = None


def chat(model: str, prompt: str, *, timeout: int,
         cfg: GenConfig = GenConfig(), url: str = OLLAMA_CHAT) -> str:
    options = {"temperature": cfg.temperature,
               "num_predict": cfg.num_predict, "seed": cfg.seed}
    if cfg.num_ctx is not None:
        options["num_ctx"] = cfg.num_ctx
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": cfg.think,
        "options": options,
    }).encode()
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["message"]["content"]
    except urllib.error.HTTPError as exc:
        raise classify_ollama(exc, model) from exc


def chat_with_retries(model: str, prompt: str, *, timeout: int,
                      cfg: GenConfig = GenConfig(), tries: int = 3,
                      backoff: float = 2.0, url: str = OLLAMA_CHAT) -> str:
    """Model swaps make Ollama 5xx transiently; a lost 27B run taught us that.
    Transient failures only are retried (see call_with_retries)."""
    return call_with_retries(
        lambda: chat(model, prompt, timeout=timeout, cfg=cfg, url=url),
        tries=tries, backoff=backoff)


class JsonlCache:
    """Append-only cache keyed on chosen fields.

    The cache is what makes the 150-item pilot -> 500-item scale-up free: every
    (model, rendering, prompt) already answered is never asked again. Records
    with empty responses are never written, so failures retry next run.
    """

    def __init__(self, path: pathlib.Path, key_fields: tuple[str, ...]) -> None:
        """Unparsable lines are skipped and counted, never fatal: a kill in
        the middle of put() leaves a torn last line, and a multi-day run that
        pauses and resumes must not die on it. A skipped line only costs one
        re-asked call. The file itself is never rewritten -- it holds paid
        answers -- and the next put() starts on a fresh line so the fragment
        stays isolated."""
        self.path = path
        self.key_fields = key_fields
        self._store: dict[tuple, dict] = {}
        self.skipped_lines = 0
        self._torn_tail = False
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        self._store[self._key(record)] = record
                    except (json.JSONDecodeError, KeyError, TypeError):
                        self.skipped_lines += 1
                    self._torn_tail = not line.endswith("\n")
            if self.skipped_lines:
                print(f"JsonlCache: skipped {self.skipped_lines} unparsable "
                      f"line(s) in {path}", file=sys.stderr)

    def _key(self, record: dict) -> tuple:
        return tuple(record[k] for k in self.key_fields)

    def get(self, record: dict) -> dict | None:
        return self._store.get(self._key(record))

    def put(self, record: dict) -> None:
        if not record.get("raw"):
            return  # empty responses are failures, not answers
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            if self._torn_tail:
                handle.write("\n")   # isolate the torn fragment
                self._torn_tail = False
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._store[self._key(record)] = record

    def coverage(self, wanted: list[dict]) -> tuple[int, int]:
        hit = sum(1 for record in wanted if self._key(record) in self._store)
        return hit, len(wanted)
