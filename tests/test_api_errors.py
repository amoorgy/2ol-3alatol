"""Retry classification for the model clients: quota never retried,
transient retried with backoff, everything else raised at once.

Every provider signature here is taken from that provider's official error
table (URLs in each client's docstring). No test touches the network or
.env: urlopen and _load_key are patched.
"""

import http.client
import io
import json
import pathlib
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval import deepseek_client, model_clients, openrouter_client, zai_client
from franco_eval.model_clients import (ClassifiedHTTPError, FatalAPIError,
                                       QuotaExhausted, RequestRejected,
                                       TransientHTTPError, call_with_retries,
                                       is_transient)


_MADE: list[urllib.error.HTTPError] = []


def http_error(status: int, body=None, headers=None) -> urllib.error.HTTPError:
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    error = urllib.error.HTTPError("https://example.invalid/chat", status,
                                   "error", headers or {}, io.BytesIO(body or b""))
    _MADE.append(error)
    return error


def tearDownModule():
    """HTTPError wraps a file object that warns if left unclosed."""
    for error in _MADE:
        error.close()


def zai_body(code: str, message: str = "m") -> dict:
    return {"error": {"code": code, "message": message}}


class ZaiClassification(unittest.TestCase):
    """docs.z.ai/api-reference/api-code: HTTP 429 carries BOTH throttling
    (1302, 1305) and quota exhaustion (1113, 1308, 1310, 1316-1321), so the
    business code in the body decides."""

    CASES = [
        (429, "1113", QuotaExhausted),
        (429, "1308", QuotaExhausted),
        (429, "1310", QuotaExhausted),
        *[(429, str(code), QuotaExhausted) for code in range(1316, 1322)],
        (429, "1302", TransientHTTPError),
        (429, "1305", TransientHTTPError),
        (500, "1200", TransientHTTPError),
        (500, "1230", TransientHTTPError),
        (500, "1234", TransientHTTPError),
        (401, "1000", FatalAPIError),
        (401, "1001", FatalAPIError),
        (401, "1003", FatalAPIError),
        (401, "1005", FatalAPIError),
        (429, "1309", FatalAPIError),  # plan expired: waiting will not help
        (429, "1311", FatalAPIError),  # model not in the plan
        (429, "1313", FatalAPIError),  # fair-use restriction, needs an appeal
        (429, "1314", FatalAPIError),
        (429, "1315", FatalAPIError),
        (400, "1211", FatalAPIError),  # unknown model: every call would fail
        (403, "1220", FatalAPIError),
        (400, "1221", FatalAPIError),
        (400, "1222", FatalAPIError),
        (400, "1210", RequestRejected),
        (400, "1214", RequestRejected),
        (400, "1261", RequestRejected),
        (400, "1301", RequestRejected),  # content filter: this prompt only
    ]

    def test_documented_codes(self):
        for status, code, expected in self.CASES:
            with self.subTest(status=status, code=code):
                got = zai_client.classify(http_error(status, zai_body(code)), "glm-5.3")
                self.assertIs(type(got), expected)
                self.assertEqual(got.error_code, code)
                self.assertEqual(got.code, status)

    def test_quota_message_with_reset_time_is_kept(self):
        message = "Usage limit reached for 5 hour. Your limit will reset at 2026-09-25 18:00:00"
        got = zai_client.classify(http_error(429, zai_body("1308", message)), "glm-5.3")
        self.assertIn("reset at 2026-09-25 18:00:00", str(got))
        self.assertIn("glm-5.3", str(got))

    def test_unknown_codes_fall_back_on_the_status(self):
        self.assertIsInstance(zai_client.classify(http_error(429, zai_body("9999")), "m"),
                              TransientHTTPError)
        self.assertIsInstance(zai_client.classify(http_error(401, "not json"), "m"),
                              FatalAPIError)
        self.assertIsInstance(zai_client.classify(http_error(502, ""), "m"),
                              TransientHTTPError)
        self.assertIsInstance(zai_client.classify(http_error(418, ""), "m"),
                              RequestRejected)

    def test_numeric_code_is_normalised(self):
        got = zai_client.classify(http_error(429, {"error": {"code": 1308}}), "m")
        self.assertIsInstance(got, QuotaExhausted)


class DeepSeekClassification(unittest.TestCase):
    """api-docs.deepseek.com/quick_start/error_codes + rate_limit: 402 is the
    balance, 429 is the concurrency limit (retry after a wait).

    402 is FATAL, not a quota pause (user decision 2026-09-26): a prepaid
    balance never refills by waiting, so a pause would retry hourly forever;
    the run stops and whatever it finished is used."""

    def test_documented_statuses(self):
        cases = {402: FatalAPIError, 401: FatalAPIError, 400: RequestRejected,
                 422: RequestRejected, 429: TransientHTTPError,
                 500: TransientHTTPError, 503: TransientHTTPError}
        for status, expected in cases.items():
            with self.subTest(status=status):
                got = deepseek_client.classify(
                    http_error(status, {"error": {"message": "x"}}), "deepseek-v4-flash")
                self.assertIs(type(got), expected)


class OpenRouterClassification(unittest.TestCase):
    """openrouter.ai/docs/api_reference/limits + errors-and-debugging."""

    def test_key_or_account_credit_exhaustion_is_quota(self):
        body = {"error": {"code": 402, "message": "insufficient credits",
                          "metadata": {"limit_source": "openrouter_key_limit"}}}
        self.assertIsInstance(openrouter_client.classify(http_error(402, body), "m"),
                              QuotaExhausted)

    def test_in_flight_budget_is_transient_and_carries_retry_after(self):
        body = {"error": {"code": 402, "message": "in flight",
                          "metadata": {"reason": "in_flight_budget_exhausted",
                                       "limit_source": "openrouter_in_flight_budget"}}}
        got = openrouter_client.classify(http_error(402, body, {"Retry-After": "7"}), "m")
        self.assertIsInstance(got, TransientHTTPError)
        self.assertEqual(got.retry_after, 7.0)

    def test_observed_403_key_limit_is_quota(self):
        """Not in the docs: seen live on 2026-08-27 (changes.md)."""
        body = {"error": {"code": 403, "message": "Key limit exceeded (total limit)"}}
        self.assertIsInstance(openrouter_client.classify(http_error(403, body), "m"),
                              QuotaExhausted)

    def test_other_statuses(self):
        cases = {401: FatalAPIError, 403: RequestRejected, 404: RequestRejected,
                 408: TransientHTTPError, 429: TransientHTTPError,
                 502: TransientHTTPError, 503: TransientHTTPError}
        for status, expected in cases.items():
            with self.subTest(status=status):
                got = openrouter_client.classify(
                    http_error(status, {"error": {"code": status, "message": "x"}}), "m")
                self.assertIs(type(got), expected)


class OllamaClassification(unittest.TestCase):
    """docs.ollama.com/api/errors: 404 = model doesn't exist."""

    def test_statuses(self):
        cases = {404: FatalAPIError, 400: RequestRejected, 429: TransientHTTPError,
                 500: TransientHTTPError, 502: TransientHTTPError}
        for status, expected in cases.items():
            with self.subTest(status=status):
                got = model_clients.classify_ollama(
                    http_error(status, {"error": "model 'x' not found"}), "x")
                self.assertIs(type(got), expected)


class BackwardCompatibility(unittest.TestCase):
    def test_classified_errors_are_still_http_url_and_os_errors(self):
        """run_arena_judge.py catches (OSError, TimeoutError); older callers
        catch URLError. A classified error must still land in those."""
        for cls in (QuotaExhausted, FatalAPIError, RequestRejected, TransientHTTPError):
            error = cls(http_error(429, b"{}"), provider="p", model="m",
                        error_code=None, message="x")
            for base in (urllib.error.HTTPError, urllib.error.URLError, OSError,
                         ClassifiedHTTPError):
                self.assertIsInstance(error, base, (cls, base))

    def test_error_body_is_kept(self):
        error = zai_client.classify(http_error(429, zai_body("1308", "limit")), "m")
        self.assertIn(b"1308", error.body)


class Transience(unittest.TestCase):
    def test_what_counts_as_transient(self):
        transient = [urllib.error.URLError("refused"), TimeoutError(),
                     ConnectionResetError(), http.client.RemoteDisconnected("x"),
                     http.client.IncompleteRead(b""),
                     json.JSONDecodeError("blank body", "", 0),
                     TransientHTTPError(http_error(503), provider="p", model="m",
                                        error_code=None, message="x"),
                     http_error(503), http_error(429)]
        permanent = [FileNotFoundError(".env"), KeyError("choices"),
                     RuntimeError("no API= line"), http_error(400), http_error(404),
                     QuotaExhausted(http_error(429), provider="p", model="m",
                                    error_code="1308", message="x"),
                     RequestRejected(http_error(400), provider="p", model="m",
                                     error_code=None, message="x")]
        for exc in transient:
            self.assertTrue(is_transient(exc), repr(exc))
        for exc in permanent:
            self.assertFalse(is_transient(exc), repr(exc))


class Retries(unittest.TestCase):
    def run_with(self, errors, *, tries=3, result="ok"):
        calls, sleeps = [], []

        def call():
            calls.append(1)
            if len(calls) <= len(errors):
                raise errors[len(calls) - 1]
            return result

        try:
            value = call_with_retries(call, tries=tries, backoff=2.0,
                                      sleep=sleeps.append)
        except Exception as exc:  # returned for the assertions
            return exc, len(calls), sleeps
        return value, len(calls), sleeps

    def test_quota_is_never_retried(self):
        quota = QuotaExhausted(http_error(429), provider="zai", model="m",
                               error_code="1308", message="x")
        got, calls, sleeps = self.run_with([quota] * 3)
        self.assertIs(got, quota)
        self.assertEqual((calls, sleeps), (1, []))

    def test_rejected_and_local_errors_are_never_retried(self):
        for exc in (RequestRejected(http_error(400), provider="p", model="m",
                                    error_code=None, message="x"),
                    FileNotFoundError(".env")):
            got, calls, _ = self.run_with([exc] * 3)
            self.assertIs(got, exc)
            self.assertEqual(calls, 1)

    def test_transient_is_retried_with_linear_backoff_then_raised(self):
        errors = [urllib.error.URLError("dns")] * 3
        got, calls, sleeps = self.run_with(errors)
        self.assertIs(got, errors[-1])
        self.assertEqual((calls, sleeps), (3, [2.0, 4.0]))

    def test_transient_then_success(self):
        got, calls, _ = self.run_with([TimeoutError()])
        self.assertEqual((got, calls), ("ok", 2))

    def test_retry_after_is_honoured_and_capped(self):
        slow = TransientHTTPError(http_error(429, b"", {"Retry-After": "30"}),
                                  provider="p", model="m", error_code=None, message="x")
        _, _, sleeps = self.run_with([slow])
        self.assertEqual(sleeps, [30.0])
        huge = TransientHTTPError(http_error(429, b"", {"Retry-After": "99999"}),
                                  provider="p", model="m", error_code=None, message="x")
        _, _, sleeps = self.run_with([huge])
        self.assertEqual(sleeps, [model_clients.MAX_RETRY_AFTER])


class ClientWiring(unittest.TestCase):
    """The real chat_with_retries of each client, with urlopen faked."""

    def urlopen_raising(self, error):
        return mock.patch("urllib.request.urlopen", side_effect=error)

    def test_zai_quota_raises_after_one_request(self):
        error = http_error(429, zai_body("1310", "Weekly/Monthly Limit Exhausted"))
        with self.urlopen_raising(error) as urlopen, \
                mock.patch.object(zai_client, "_load_key", return_value="k"):
            with self.assertRaises(QuotaExhausted):
                zai_client.chat_with_retries("glm-5.3", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 1)

    def test_zai_rate_limit_is_retried(self):
        with self.urlopen_raising(http_error(429, zai_body("1302"))) as urlopen, \
                mock.patch.object(zai_client, "_load_key", return_value="k"):
            with self.assertRaises(TransientHTTPError):
                zai_client.chat_with_retries("glm-5.3", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 3)

    def test_deepseek_balance_raises_after_one_request(self):
        # fatal since 2026-09-26: a spent prepaid balance stops the run
        with self.urlopen_raising(http_error(402, {"error": {"message": "Insufficient Balance"}})) as urlopen, \
                mock.patch.object(deepseek_client, "_load_key", return_value="k"):
            with self.assertRaises(FatalAPIError):
                deepseek_client.chat_with_retries("deepseek-v4-flash", "p",
                                                  timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 1)

    def test_openrouter_credits_raise_after_one_request(self):
        with self.urlopen_raising(http_error(402, {"error": {"code": 402, "message": "x"}})) as urlopen, \
                mock.patch.object(openrouter_client, "_load_key", return_value="k"):
            with self.assertRaises(QuotaExhausted):
                openrouter_client.chat_with_retries("m", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 1)

    def test_missing_env_file_is_not_retried(self):
        with mock.patch.object(zai_client, "_load_key",
                               side_effect=FileNotFoundError(".env")) as load:
            with self.assertRaises(FileNotFoundError):
                zai_client.chat_with_retries("glm-5.3", "p", timeout=5, backoff=0)
        self.assertEqual(load.call_count, 1)

    def test_ollama_missing_model_is_not_retried(self):
        with self.urlopen_raising(http_error(404, {"error": "model not found"})) as urlopen:
            with self.assertRaises(FatalAPIError):
                model_clients.chat_with_retries("nope:1b", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 1)

    def test_ollama_model_swap_500_is_still_retried(self):
        with self.urlopen_raising(http_error(500, {"error": "swap"})) as urlopen:
            with self.assertRaises(TransientHTTPError):
                model_clients.chat_with_retries("m", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 3)

    def test_connection_refused_is_still_retried(self):
        with self.urlopen_raising(urllib.error.URLError("refused")) as urlopen:
            with self.assertRaises(urllib.error.URLError):
                model_clients.chat_with_retries("m", "p", timeout=5, backoff=0)
        self.assertEqual(urlopen.call_count, 3)


if __name__ == "__main__":
    unittest.main()
