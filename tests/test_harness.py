"""Offline unit tests for the vLLM harness. No network, GPU, Docker or inference."""

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark
import common
import prompts


def sse(events):
    """Encode dicts as raw SSE byte lines, including comments and [DONE]."""
    lines = [b": keepalive\n", b"\n"]
    for event in events:
        lines.append(b"data: " + json.dumps(event).encode() + b"\n")
        lines.append(b"\n")
    lines.append(b"data: [DONE]\n")
    return lines


class FakeClock:
    def __init__(self, step=0.5):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def delta(piece, tokens=None, reasoning=None):
    body = {"content": piece} if piece is not None else {}
    if reasoning is not None:
        body["reasoning_content"] = reasoning
    event = {"choices": [{"delta": body}]}
    if tokens is not None:
        event["usage"] = {"prompt_tokens": 10, "completion_tokens": tokens}
    return event


class StreamTests(unittest.TestCase):
    def test_ttft_uses_first_text_not_first_event(self):
        clock = FakeClock(step=0.5)
        events = [{"choices": [{"delta": {"role": "assistant"}}]}, delta("Hello"),
                  delta(" world", tokens=3)]
        result = benchmark.consume_stream(sse(events), 0.0, clock=clock)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertAlmostEqual(result["ttft_seconds"], 1.0)
        self.assertEqual(result["text"], "Hello world")

    def test_reasoning_delta_counts_as_text(self):
        clock = FakeClock()
        events = [delta(None, reasoning="thinking"), delta("answer", tokens=2)]
        result = benchmark.consume_stream(sse(events), 0.0, clock=clock)
        self.assertEqual(result["text"], "thinkinganswer")

    def test_missing_usage_is_fatal(self):
        clock = FakeClock()
        with self.assertRaises(RuntimeError):
            benchmark.consume_stream(sse([delta("text")]), 0.0, clock=clock)

    def test_stream_error_event_is_fatal(self):
        clock = FakeClock()
        events = [{"error": {"message": "boom"}}]
        with self.assertRaises(RuntimeError):
            benchmark.consume_stream(sse(events), 0.0, clock=clock)

    def test_no_text_is_fatal(self):
        clock = FakeClock()
        events = [{"choices": [{"delta": {"role": "assistant"}}],
                   "usage": {"prompt_tokens": 4, "completion_tokens": 1}}]
        with self.assertRaises(RuntimeError):
            benchmark.consume_stream(sse(events), 0.0, clock=clock)


class MetricsTests(unittest.TestCase):
    TEXT = """# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 2.0
vllm:num_requests_waiting{model_name="m"} 0.0
vllm:kv_cache_usage_perc{model_name="m"} 0.25
vllm:prefix_cache_hits_total{model_name="m"} 100.0
vllm:prefix_cache_queries_total{model_name="m"} 400.0
vllm:prompt_tokens_total{model_name="m"} 1000.0
vllm:generation_tokens_total{model_name="m"} 50.0
vllm:request_success_total{finished_reason="stop"} 3.0
vllm:request_success_total{finished_reason="length"} 1.0
"""

    def test_parse_sums_label_dimensions(self):
        series = common.parse_prometheus(self.TEXT)
        self.assertEqual(series["vllm:num_requests_running"], [2.0])
        self.assertEqual(sum(series["vllm:request_success_total"]), 4.0)

    def test_normalize_aliases_and_counters(self):
        metrics = common.normalize_metrics(common.parse_prometheus(self.TEXT))
        self.assertEqual(metrics["running"], 2.0)
        self.assertEqual(metrics["waiting"], 0.0)
        self.assertEqual(metrics["kv_usage"], 0.25)
        self.assertEqual(metrics["success"], 4.0)
        self.assertEqual(metrics["prefix_hits"], 100.0)

    def test_old_gpu_cache_name_is_accepted(self):
        text = 'vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n'
        metrics = common.normalize_metrics(common.parse_prometheus(text))
        self.assertEqual(metrics["kv_usage"], 0.5)

    def test_summarize_metrics_computes_hit_rate(self):
        samples = [{"running": 1, "waiting": 0, "kv_usage": 0.1,
                    "prefix_hits": 0, "prefix_queries": 0},
                   {"running": 0, "waiting": 0, "kv_usage": 0.4,
                    "prefix_hits": 30, "prefix_queries": 60}]
        summary = benchmark.summarize_metrics(samples, samples[0])
        self.assertEqual(summary["peak_running"], 1)
        self.assertEqual(summary["maximum_kv_cache_usage_fraction"], 0.4)
        self.assertEqual(summary["prefix_cache_hit_rate"], 0.5)


class BoundedTokenCountTests(unittest.TestCase):
    class Rejecting:
        def __init__(self, detail):
            self.detail = detail

        def token_count(self, text):
            raise urllib.error.HTTPError("u", 400, f"Bad Request: {self.detail}", {}, None)

    def test_context_overflow_counts_as_above_any_target(self):
        count = benchmark.bounded_token_count(
            self.Rejecting("prompt exceeds model context"), "x")
        self.assertGreater(count, 1 << 20)

    def test_other_http_errors_still_raise(self):
        with self.assertRaises(urllib.error.HTTPError):
            benchmark.bounded_token_count(self.Rejecting("something else"), "x")


class BudgetTests(unittest.TestCase):
    CONFIG = {"status": "ready", "context_length": 262144}

    def test_accepts_a_fitting_request(self):
        common.check_budget(self.CONFIG, 4, 1024, 256, 16)

    def test_rejects_concurrency_above_profile(self):
        with self.assertRaises(ValueError):
            common.check_budget(self.CONFIG, 17, 1024, 256, 16)

    def test_rejects_prompt_plus_output_over_context(self):
        with self.assertRaises(ValueError):
            common.check_budget(self.CONFIG, 1, 262144, 1, 16)

    def test_rejects_unready_server(self):
        with self.assertRaises(ValueError):
            common.check_budget({"status": "unhealthy", "context_length": 100}, 1, 10, 10, 16)


class PromptTests(unittest.TestCase):
    @staticmethod
    def count(text):
        return len(text.split())

    def test_fit_never_exceeds_target_and_is_close(self):
        for target in (50, 500, 3000):
            fitted = prompts.fit_text(self.count, target, "m-1", ())
            self.assertLessEqual(fitted["tokens"], target)
            self.assertGreaterEqual(fitted["tokens"], target - 40)

    def test_fit_refuses_a_target_below_the_preamble(self):
        with self.assertRaises(ValueError):
            prompts.fit_text(self.count, 1, "m-1", ())

    def test_retrieval_prompts_carry_three_codes(self):
        built = prompts.build_prompts(self.count, 400, 1, "set", 7, "retrieval")
        self.assertEqual(len(built[0]["expected_codes"]), 3)
        for code in built[0]["expected_codes"]:
            self.assertIn(code, built[0]["text"])


class EvidenceTests(unittest.TestCase):
    def test_write_json_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            common.write_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                common.write_json(path, {"a": 2})

    def test_scrub_redacts_secrets(self):
        scrubbed = common.scrub({"api_key": "x", "note": "token hf_" + "a" * 20,
                                 "nested": [{"GITHUB_PAT": "y"}]})
        self.assertEqual(scrubbed["api_key"], "[REDACTED]")
        self.assertIn("[REDACTED]", scrubbed["note"])
        self.assertEqual(scrubbed["nested"][0]["GITHUB_PAT"], "[REDACTED]")

    def test_label_validation(self):
        self.assertEqual(common.label("short-c1-cold-20261026"), "short-c1-cold-20261026")
        for bad in ("", "../x", "a b", "x" * 81):
            with self.assertRaises(ValueError):
                common.label(bad)

    def test_percentile(self):
        self.assertEqual(common.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertIsNone(common.percentile([], 0.5))

    def test_arrival_schedule_rejects_negative_stagger(self):
        with self.assertRaises(ValueError):
            benchmark.planned_arrival_offsets(2, -1, 10)


if __name__ == "__main__":
    unittest.main()
