#!/usr/bin/env python3
"""Capacity-first vLLM workload screening. Never restarts or flushes the cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.error
import uuid

from common import (Client, DEFAULT_CONTAINER, DEFAULT_PROFILE, DEFAULT_URL, check_budget,
                    experiment_lock, label, load_profile, percentile, report_path,
                    require_live, scrub, stamp, verify_profile, write_json)
from host_memory import memory_status, sample_memory, summarize_memory
from prompts import build_prompts

# Above any target the prompt fitter can use; returned for probes the server
# rejects because they exceed the model context.
OVER_CONTEXT = 1 << 31


def bounded_token_count(client, text):
    """Token count for a fitting probe that may exceed the model context.

    Prompt fitting doubles candidates until the count passes the target. If the
    tokenizer endpoint enforces the context limit, a candidate can be rejected
    before it is counted. Treat that probe as larger than any target so the
    search backs off; other HTTP errors still propagate.
    """
    try:
        return client.token_count(text)
    except urllib.error.HTTPError as error:
        detail = str(error).lower()
        if "prompt exceeds" in detail or "maximum model length" in detail:
            return OVER_CONTEXT
        raise


def iter_events(lines):
    """Yield decoded SSE events from raw byte lines; ignore comments and `[DONE]`."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        if isinstance(event, dict) and "error" in event:
            raise RuntimeError("Server stream error: " + json.dumps(scrub(event["error"])))
        yield event


def _delta_text(event):
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            or delta.get("reasoning_text") or "")


def consume_stream(lines, started, clock=time.monotonic):
    """OpenAI-style SSE with cumulative usage when continuous stats are requested.

    TTFT is the first non-empty text delta, not the first SSE event: keepalives
    and role chunks precede generation. vLLM can emit several tokens per delta,
    so per-token latency is derived from cumulative usage and labelled an estimate.
    """
    first_text = None
    first_text_tokens = None
    last_text = None
    last_tokens = 0
    usage = None
    text = []
    events = 0
    for event in iter_events(lines):
        events += 1
        arrived = clock() - started
        current = event.get("usage")
        if isinstance(current, dict):
            usage = current
            total = current.get("completion_tokens")
            if isinstance(total, int):
                last_tokens = total
        piece = _delta_text(event)
        if piece:
            if first_text is None:
                first_text = arrived
                first_text_tokens = last_tokens or 1
            last_text = arrived
            text.append(piece)
    if events == 0:
        raise RuntimeError("No generation events in stream")
    if usage is None:
        raise RuntimeError("No usage reported; enable stream_options.include_usage")
    completion = usage.get("completion_tokens")
    if not isinstance(completion, int) or completion < 1:
        raise RuntimeError("Missing completion-token accounting")
    if first_text is None:
        raise RuntimeError("Stream produced no text")
    elapsed = clock() - started
    decoded = last_tokens - (first_text_tokens or 1)
    decode_span = (last_text - first_text) if last_text is not None else 0.0
    return {
        "elapsed_seconds": elapsed,
        "ttft_seconds": first_text,
        "last_text_seconds": last_text,
        "completion_tokens": completion,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "average_tpot_seconds_estimate": (decode_span / decoded) if decoded > 0 and decode_span > 0 else None,
        "decode_tokens_per_second_estimate": (decoded / decode_span) if decoded > 0 and decode_span > 0 else None,
        "end_to_end_tokens_per_second": completion / elapsed,
        "stream_events": events,
        "usage": usage,
        "text": "".join(text),
    }


# Recipe sampling defaults (generation_config.json): greedy is not validated.
FLINT_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def generate(client, prompt, output_tokens, forced, served_name, seed=None,
             timeout=600):
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "ignore_eos": bool(forced),
        **FLINT_SAMPLING,
    }
    if seed is not None:
        payload["seed"] = seed
    started = time.monotonic()
    with client.open("/v1/chat/completions", payload, timeout=timeout) as response:
        result = consume_stream(response, started)
    answer = result.pop("text")
    answer_limit = 16384 if prompt["purpose"] == "retrieval" else 4096
    result["answer_excerpt"] = answer[:answer_limit]
    result["answer_excerpt_truncated"] = len(answer) > answer_limit
    result["retrieved_codes"] = {code: code in answer for code in prompt["expected_codes"]}
    result["retrieval_passed"] = all(result["retrieved_codes"].values())
    result["prompt_sha256"] = prompt["sha256"]
    result["prompt_target_tokens"] = prompt["target_tokens"]
    result["prompt_actual_tokens"] = prompt["actual_tokens"]
    return result


def sample_metrics(client):
    snapshot = client.metrics()
    keys = ("running", "waiting", "swapped", "kv_usage", "prompt_tokens", "generation_tokens",
            "prefix_hits", "prefix_queries", "success", "preemptions")
    result = {"at": stamp()}
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = value
    if "running" not in result or "waiting" not in result:
        raise ValueError("vLLM metrics snapshot lacks running/waiting gauges")
    return result


def wait_for_idle(client, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        snapshot = sample_metrics(client)
        if snapshot["running"] == 0 and snapshot["waiting"] == 0:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError("Service did not return idle after completed requests")
        time.sleep(0.2)


def summarize(rounds):
    samples = [sample for row in rounds for sample in row["samples"]]
    if not samples:
        return {"sample_count": 0}
    return {
        "sample_count": len(samples),
        "median_aggregate_tokens_per_second": statistics.median(
            row["aggregate_tokens_per_second"] for row in rounds if "aggregate_tokens_per_second" in row),
        "median_end_to_end_tokens_per_second_per_request": statistics.median(
            sample["end_to_end_tokens_per_second"] for sample in samples),
        "latency_p50_seconds": percentile([sample["elapsed_seconds"] for sample in samples], 0.5),
        "latency_p95_seconds": percentile([sample["elapsed_seconds"] for sample in samples], 0.95)
        if len(samples) >= 30 else None,
        "ttft_p50_seconds": percentile([sample["ttft_seconds"] for sample in samples], 0.5),
        "ttft_p95_seconds": percentile([sample["ttft_seconds"] for sample in samples], 0.95)
        if len(samples) >= 30 else None,
        "decode_tokens_per_second_p50_estimate": percentile(
            [sample["decode_tokens_per_second_estimate"] for sample in samples], 0.5),
        "percentile_caveat": "p95 omitted below 30 samples; larger sample counts still needed for reliable tails",
    }


def validate_samples(samples, output_tokens, forced, require_retrieval):
    for result in samples:
        if forced and result["completion_tokens"] != output_tokens:
            raise RuntimeError("Forced-output sample did not generate the requested token count")
        if require_retrieval and not result["retrieval_passed"]:
            raise RuntimeError("Retrieval failed; preserve report and investigate before increasing load")


def planned_arrival_offsets(concurrency, stagger_seconds, timeout):
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Concurrency must be a positive integer")
    if isinstance(stagger_seconds, bool) or not isinstance(stagger_seconds, (int, float)):
        raise ValueError("Arrival stagger must be a finite, nonnegative number")
    stagger_seconds = float(stagger_seconds)
    if not math.isfinite(stagger_seconds) or stagger_seconds < 0:
        raise ValueError("Arrival stagger must be a finite, nonnegative number")
    offsets = [index * stagger_seconds for index in range(concurrency)]
    if offsets[-1] > timeout:
        raise ValueError("Arrival schedule span cannot exceed the per-request timeout")
    return offsets


def wait_for_arrival_slot(round_started, scheduled_offset, *, clock=None, sleeper=None):
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    remaining = round_started + scheduled_offset - clock()
    if remaining > 0:
        sleeper(remaining)
    return clock() - round_started


def container_state(name):
    """Optional docker audit; unavailable Docker access is reported, never fatal."""
    try:
        raw = subprocess.run(
            ["docker", "inspect", name], capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        record = json.loads(raw)[0]
    except Exception as error:  # noqa: BLE001 - audit must not mask the workload
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    state = record.get("State") or {}
    config = record.get("Config") or {}
    return {
        "available": True,
        "running": bool(state.get("Running")),
        "oom_killed": bool(state.get("OOMKilled")),
        "restart_count": record.get("RestartCount"),
        "image_id": record.get("Image"),
        "argv": config.get("Entrypoint", []) + config.get("Cmd", []),
    }


def run(args, report, profile):
    offsets = planned_arrival_offsets(args.concurrency, args.arrival_stagger_seconds, args.timeout)
    policy = args.policy
    warning_bytes = args.memory_warning_mib << 20
    report["policy"] = {"mode": policy, "memory_warning_mib": args.memory_warning_mib,
                        "note": "Queueing and memory are reported; only completed work is a capacity gate"}
    report["warnings"] = []
    report["metrics"] = {"samples": [], "errors": []}
    report["memory"] = {"samples": [], "errors": []}

    if args.audit_container:
        report["container_before"] = container_state(args.container)
        if report["container_before"].get("available") and (
                not report["container_before"]["running"] or report["container_before"]["oom_killed"]):
            raise RuntimeError("Container not running or reports OOMKilled; no inference sent")

    client = Client(args.base_url, timeout=args.timeout)
    before = client.info()
    report["server_before"] = before
    if args.profile != "any":
        report["profile"] = verify_profile(before["config"], args.profile)
    else:
        report["profile"] = {"name": "any"}
    check_budget(before["config"], args.concurrency, args.prompt_tokens, args.output_tokens,
                 profile["serving"]["max_num_seqs"])

    idle = wait_for_idle(client)
    report["initial_metrics"] = idle
    if idle["running"] or idle["waiting"]:
        raise RuntimeError("Engine is not idle; no benchmark requests submitted")

    stop = threading.Event()

    def monitor():
        next_memory = time.monotonic()
        while not stop.is_set():
            try:
                report["metrics"]["samples"].append(sample_metrics(client))
            except Exception as error:  # noqa: BLE001 - telemetry gaps are recorded
                report["metrics"]["errors"].append(f"{type(error).__name__}: {error}")
            if time.monotonic() >= next_memory:
                try:
                    report["memory"]["samples"].append(sample_memory())
                except Exception as error:  # noqa: BLE001
                    report["memory"]["errors"].append(f"{type(error).__name__}: {error}")
                next_memory = time.monotonic() + 1.0
            stop.wait(0.2)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    prompt_set_id = args.prompt_set_id or report["run_id"]
    report["prompt_set_id"] = prompt_set_id
    served_name = before["config"]["served_model_name"]
    measured = {}

    def count(text):
        if text not in measured:
            measured[text] = bounded_token_count(client, text)
        return measured[text]

    def prompts_for(namespace):
        prompts = build_prompts(count, args.prompt_tokens, args.concurrency, namespace,
                                args.seed, args.mode)
        report["token_targeting"] = [
            {"target": p["target_tokens"], "actual": p["actual_tokens"],
             "sentences": p["sentences"], "short_clauses": p["short_clauses"]}
            for p in prompts
        ]
        return prompts

    def check_round(row):
        row["metrics_after"] = wait_for_idle(client)
        health = client.info()
        row["service_ready"] = health["config"]["status"] == "ready"
        if health["config"] != before["config"] or not row["service_ready"]:
            raise RuntimeError("Service health/configuration changed; stop before another round")
        if args.audit_container:
            state = container_state(args.container)
            report.setdefault("runtime_audits", []).append(state)
            if state.get("available") and state["oom_killed"]:
                raise RuntimeError("Container reports OOMKilled; stop before another round")

    try:
        warmed = None
        if args.cache == "warm":
            warmed = prompts_for(prompt_set_id)
            for warmup_pass in range(args.warmup_passes):
                for prompt_index, prompt in enumerate(warmed):
                    result = generate(client, prompt, args.output_tokens, False, served_name,
                                      args.seed, args.timeout)
                    result["warmup_pass"] = warmup_pass
                    result["prompt_index"] = prompt_index
                    report.setdefault("warmup", []).append(result)
                    validate_samples([result], args.output_tokens, False,
                                     args.mode == "retrieval")
                    check_round(result)

        for round_index in range(args.rounds):
            prompts = warmed or prompts_for(f"{prompt_set_id}-r{round_index}")
            barrier = threading.Barrier(args.concurrency) if args.arrival_stagger_seconds == 0 else None
            row = {"round": round_index, "samples": [], "errors": [], "started_at": stamp(),
                   "arrival_schedule": {
                       "mode": "synchronized" if barrier is not None else "staggered",
                       "stagger_seconds": float(args.arrival_stagger_seconds),
                       "planned_offsets_seconds": offsets}}
            report["rounds"].append(row)

            def worker(prompt, client_index):
                if barrier is not None:
                    barrier.wait(timeout=30)
                    offset = time.monotonic() - started
                else:
                    offset = wait_for_arrival_slot(started, offsets[client_index])
                result = generate(client, prompt, args.output_tokens, args.mode == "decode",
                                  served_name, args.seed, args.timeout)
                result["client_index"] = client_index
                result["client_start_offset_seconds"] = offset
                return result

            started = time.monotonic()
            metrics_start = len(report["metrics"]["samples"])
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(worker, prompt, index) for index, prompt in enumerate(prompts)]
                for future in futures:
                    try:
                        row["samples"].append(future.result())
                    except Exception as error:  # noqa: BLE001 - keep partial evidence
                        row["errors"].append({"type": type(error).__name__, "message": str(error)})
            row["elapsed_seconds"] = time.monotonic() - started
            if row["errors"]:
                raise RuntimeError("A concurrent request failed; partial results retained")
            observed = report["metrics"]["samples"][metrics_start:]
            row["peak_waiting_observed"] = max((s.get("waiting", 0) for s in observed), default=0)
            if policy == "no-queue" and row["peak_waiting_observed"]:
                raise RuntimeError("Requests waited in the scheduler under the no-queue policy")
            row["aggregate_tokens_per_second"] = (
                sum(sample["completion_tokens"] for sample in row["samples"]) / row["elapsed_seconds"])
            validate_samples(row["samples"], args.output_tokens, args.mode == "decode",
                             args.mode == "retrieval")
            if args.cache == "warm":
                hits = [sample["cached_tokens"] for sample in row["samples"]]
                row["warm_cache_verified"] = all(
                    isinstance(value, int) and value >= sample["prompt_actual_tokens"] * 0.9
                    for value, sample in zip(hits, row["samples"]))
                if not row["warm_cache_verified"]:
                    raise RuntimeError("Warm run did not verify >=90% prefix reuse on every request")
            check_round(row)
            row["completed"] = True
            print(json.dumps({"round": round_index,
                              "aggregate_tokens_per_second": row["aggregate_tokens_per_second"]}),
                  flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=5)
        try:
            after = client.info()
            report["server_after"] = after
            report["configuration_drift"] = {
                key: {"before": value, "after": after["config"].get(key)}
                for key, value in before["config"].items() if after["config"].get(key) != value}
        except Exception as error:  # noqa: BLE001
            report["server_after_error"] = f"{type(error).__name__}: {error}"
        report["memory"].update(summarize_memory(report["memory"]["samples"]))
        report["memory"].update(memory_status(report["memory"]["samples"], warning_bytes))
        report["metrics"]["summary"] = summarize_metrics(
            report["metrics"]["samples"], report.get("initial_metrics"))
        if report["memory"].get("low_memory_warning"):
            report["warnings"].append("Sampled MemAvailable fell below the warning threshold")
        if report["metrics"].get("errors"):
            report["warnings"].append("Telemetry incomplete; missing samples are unknown, not zero")
        if report["metrics"]["summary"].get("peak_waiting"):
            report["warnings"].append("Queueing observed; included in end-to-end timing")

    if report.get("server_after_error") or report.get("configuration_drift"):
        raise RuntimeError("Server health/configuration changed or could not be verified after the test")
    report["summary"] = summarize(report["rounds"])


def summarize_metrics(samples, baseline=None):
    if not samples:
        return {"sample_count": 0}
    peak = {key: max(sample[key] for sample in samples)
            for key in ("running", "waiting") if any(key in sample for sample in samples)}
    first = baseline or {}
    last = samples[-1]

    def delta(key):
        start = first.get(key, 0)
        end = last.get(key)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return end - start
        return None

    hits = delta("prefix_hits")
    queries = delta("prefix_queries")
    usage = [sample["kv_usage"] for sample in samples if sample.get("kv_usage") is not None]
    return {
        "sample_count": len(samples),
        "peak_running": peak.get("running"),
        "peak_waiting": peak.get("waiting"),
        "maximum_kv_cache_usage_fraction": max(usage) if usage else None,
        "generated_tokens": delta("generation_tokens"),
        "prefill_tokens": delta("prompt_tokens"),
        "prefix_cache_hits": hits,
        "prefix_cache_queries": queries,
        "prefix_cache_hit_rate": (hits / queries) if isinstance(hits, (int, float))
        and isinstance(queries, (int, float)) and queries else None,
        "completed_requests": delta("success"),
        "preemptions": delta("preemptions"),
        "counter_scope": "Counters are deltas from the pre-run idle snapshot, not server-lifetime totals",
        "limitation": "Sampled gauges; short spikes can be missed between 0.2 s polls",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, type=label)
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help="Frozen contract to verify, or 'any' to record without matching")
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--mode", choices=("decode", "retrieval"), default="decode")
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--warmup-passes", type=int, choices=(1, 2, 3), default=1,
                        help="Sequential warmup passes per prompt; warm cache only")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--arrival-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--prompt-set-id", type=label,
                        help="Stable prompt namespace for matched A/B runs")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--policy", choices=("capacity", "no-queue"), default="capacity")
    parser.add_argument("--memory-warning-mib", type=int, default=8192)
    parser.add_argument("--audit-container", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-idle", action="store_true")
    args = parser.parse_args()
    require_live(args, parser)
    if args.memory_warning_mib < 0:
        parser.error("--memory-warning-mib must be nonnegative")
    if args.cache != "warm" and args.warmup_passes != 1:
        parser.error("Extra warmup passes require --cache warm")
    if min(args.concurrency, args.rounds, args.prompt_tokens, args.output_tokens,
           args.timeout) < 1:
        parser.error("Numeric arguments must be positive")
    try:
        planned_arrival_offsets(args.concurrency, args.arrival_stagger_seconds, args.timeout)
    except ValueError as error:
        parser.error(str(error))

    profile = load_profile(DEFAULT_PROFILE)
    report = {
        "schema_version": 1,
        "harness_revision": "vllm-capacity-v1-flint",
        "kind": "benchmark",
        "label": args.label,
        "started_at": stamp(),
        "run_id": uuid.uuid4().hex,
        "workload": {k: v for k, v in vars(args).items() if k not in ("execute", "confirm_idle")},
        "rounds": [],
        "passed": False,
        "capacity_passed": False,
        "quality_scope": ("Synthetic forced-output speed only; not coding quality"
                          if args.mode == "decode" else
                          "Synthetic in-context retrieval only; not document reasoning quality"),
    }
    path = report_path(args.label)
    with experiment_lock():
        try:
            run(args, report, profile)
            report["passed"] = True
            report["capacity_passed"] = True
        except Exception as error:  # noqa: BLE001 - failed runs are evidence too
            report["error"] = {"type": type(error).__name__, "message": str(error)}
        except KeyboardInterrupt:
            report["error"] = {"type": "KeyboardInterrupt",
                               "message": "Interrupted; verify pending requests on the server"}
        finally:
            report["finished_at"] = stamp()
            report["completed_requests"] = sum(len(row["samples"]) for row in report["rounds"])
            report["expected_requests"] = args.concurrency * args.rounds
            report["production_qualified"] = False
            if "summary" not in report:
                completed = [row for row in report["rounds"] if "aggregate_tokens_per_second" in row]
                report["summary"] = summarize(completed)
                report["summary"]["incomplete_or_failed_run"] = True
            write_json(path, report)
    print(json.dumps({"report": str(path), "passed": report["passed"],
                      "summary": report.get("summary"), "error": report.get("error")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
