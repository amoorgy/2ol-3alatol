import io
import json
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from franco_eval import deepseek_client
from franco_eval.model_clients import TruncatedReply, is_transient


def response(content, finish, reasoning=100):
    payload = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
               "usage": {"completion_tokens": reasoning + 20,
                         "completion_tokens_details": {"reasoning_tokens": reasoning}}}
    return io.BytesIO(json.dumps(payload).encode())


class ReasoningBudget(unittest.TestCase):
    """deepseek-v4-flash spent 1,109-4,463 reasoning tokens on one short
    sentence (measured 2026-09-26); at the old 4,500 cap some calls came back
    empty or cut mid-word."""

    def call(self, fake):
        sent = {}

        def urlopen(request, timeout):
            sent.update(json.loads(request.data))
            return fake
        with mock.patch.object(deepseek_client, "_load_key", return_value="k"), \
                mock.patch.object(deepseek_client.urllib.request, "urlopen", urlopen):
            return deepseek_client.chat("deepseek-v4-flash", "p", timeout=5), sent

    def test_a_finished_reply_is_returned(self):
        text, sent = self.call(response("ya khalti ana masdoom", "stop"))
        self.assertEqual(text, "ya khalti ana masdoom")

    def test_the_budget_leaves_room_for_long_reasoning(self):
        _text, sent = self.call(response("ok", "stop"))
        self.assertGreaterEqual(sent["max_tokens"], 12000)
        self.assertEqual(sent["reasoning_effort"], "high")

    def test_a_truncated_reply_is_a_retryable_failure_never_a_draft(self):
        with self.assertRaises(TruncatedReply) as ctx:
            self.call(response("ya 3ammo te3raf lelet el 7ena ma2o", "length", 4463))
        self.assertTrue(is_transient(ctx.exception))
        self.assertIn("4463", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
