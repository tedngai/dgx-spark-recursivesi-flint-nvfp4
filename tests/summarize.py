#!/usr/bin/env python3
"""Render evidence reports as Markdown. Read-only; never invents missing values."""

import argparse
import json
from pathlib import Path

import common


def load(patterns=("*.json",)):
    rows = []
    for pattern in patterns:
        for path in sorted((common.ROOT / "evidence").glob(pattern)):
            try:
                rows.append((path, json.loads(path.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
    return rows


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def artifact_tables(rows):
    models = [(path, report) for path, report in rows if report.get("kind") == "model"]
    for path, report in models:
        print(f"\nModel: `{report.get('repo')}@{str(report.get('revision'))[:12]}` — "
              f"{report.get('file_count')} files, "
              f"{fmt((report.get('total_bytes') or 0) / (1 << 30))} GiB, "
              f"verified {report.get('verified_at')}")


def benchmark_table(rows):
    print("| Label | Mode | Cache | C | Prompt | Out | Rounds | Median aggregate tok/s | TTFT p50 | Passed |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for path, report in rows:
        if report.get("kind") != "benchmark":
            continue
        workload = report.get("workload") or {}
        summary = report.get("summary") or {}
        print("| {label} | {mode} | {cache} | {c} | {p} | {o} | {rounds} | {agg} | {ttft} | {passed} |".format(
            label=report.get("label"),
            mode=workload.get("mode"),
            cache=workload.get("cache"),
            c=workload.get("concurrency"),
            p=workload.get("prompt_tokens"),
            o=workload.get("output_tokens"),
            rounds=workload.get("rounds"),
            agg=fmt(summary.get("median_aggregate_tokens_per_second")),
            ttft=fmt(summary.get("ttft_p50_seconds"), 3, " s"),
            passed="yes" if report.get("passed") else "no",
        ))


def smoke_table(rows):
    print("\n| Label | Chat | Stream | Reasoning | Tools | Forced | Prefix | Passed |")
    print("|---|---|---|---|---|---|---|---|")
    for path, report in rows:
        if report.get("kind") != "smoke":
            continue
        checks = report.get("checks") or {}

        def mark(name):
            check = checks.get(name)
            if check is None:
                return "—"
            return "pass" if check.get("passed") else "fail"

        print("| {label} | {chat} | {stream} | {reasoning} | {tools} | {forced} | {prefix} | {passed} |".format(
            label=report.get("label"), chat=mark("chat"), stream=mark("stream"),
            reasoning=mark("reasoning"), tools=mark("tool_call"),
            forced=mark("forced_output"), prefix=mark("prefix_reuse"),
            passed="yes" if report.get("passed") else "no",
        ))


def failures(rows):
    failed = [(path, report) for path, report in rows
              if report.get("kind") in ("benchmark", "smoke") and not report.get("passed")]
    if not failed:
        return
    print("\n### Failed or partial runs\n")
    print("| Report | Error |")
    print("|---|---|")
    for path, report in failed:
        error = report.get("error") or {}
        print(f"| `{path.name}` | {error.get('type', '—')}: {error.get('message', '—')} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Only the newest N reports")
    args = parser.parse_args()
    rows = load()
    if args.limit:
        rows = rows[-args.limit:]
    if not rows:
        print("No evidence reports yet. Run tests/smoke_test.py and tests/benchmark.py.")
        return
    benchmark_table(rows)
    smoke_table(rows)
    artifact_tables(rows)
    failures(rows)
    print(f"\n{len(rows)} report(s) read from evidence/.")


if __name__ == "__main__":
    main()
