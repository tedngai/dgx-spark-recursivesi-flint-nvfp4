#!/usr/bin/env python3
"""Interface smoke for the served Flint checkpoint: chat, stream, reasoning, tools, reuse."""

import argparse
import json
import time
import uuid

from common import (Client, DEFAULT_PROFILE, DEFAULT_URL, label, report_path, require_live,
                    stamp, verify_profile, write_json)


def chat(client, messages, max_tokens=256, tools=None, tool_choice=None,
         stream=False, sampling=None, timeout=600):
    payload = {"model": client.card()["id"], "messages": messages, "max_tokens": max_tokens,
               "stream": stream, **(sampling or {})}
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if not stream:
        started = time.monotonic()
        data = client.json("/v1/chat/completions", payload, timeout=timeout)
        return {"seconds": time.monotonic() - started, "data": data}
    started = time.monotonic()
    events = []
    with client.open("/v1/chat/completions", payload, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            events.append(json.loads(body))
    return {"seconds": time.monotonic() - started, "events": events}


def message_text(data):
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def stream_text(events):
    parts, reasoning = [], []
    for event in events:
        choices = event.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            parts.append(delta["content"])
        for key in ("reasoning", "reasoning_content", "reasoning_text"):
            if delta.get(key):
                reasoning.append(delta[key])
    return "".join(parts), "".join(reasoning)


def run(args, report):
    client = Client(args.base_url, timeout=args.timeout)
    info = client.info()
    report["server"] = info
    if args.profile != "any":
        report["profile"] = verify_profile(info["config"], args.profile)
    report["checks"] = {}
    checks = report["checks"]

    marker = "FLINT-READY-" + uuid.uuid4().hex[:8]
    buffered = chat(client, [{"role": "user",
                              "content": f"Reply with exactly {marker} and nothing else."}],
                    max_tokens=256, sampling={"temperature": 0.1, "top_p": 0.95, "top_k": 20})
    answer = message_text(buffered["data"])
    checks["chat"] = {"passed": marker in answer, "seconds": buffered["seconds"],
                      "answer_excerpt": answer[:200]}

    streamed = chat(client, [{"role": "user",
                              "content": "Count from one to five, separated by spaces. "
                                         "Do not think, answer directly."}],
                    max_tokens=128, stream=True,
                    sampling={"temperature": 0.1, "top_p": 0.95, "top_k": 20})
    text, _ = stream_text(streamed["events"])
    checks["stream"] = {"passed": bool(text.strip()), "seconds": streamed["seconds"],
                        "events": len(streamed["events"]), "answer_excerpt": text[:200]}

    thinking = chat(client, [{"role": "user",
                              "content": "A train travels 120 km in 90 minutes. "
                                         "What is its average speed in km/h? "
                                         "Think through this carefully before answering."}],
                    max_tokens=768, stream=True)
    text, reasoning = stream_text(thinking["events"])
    checks["reasoning"] = {"passed": bool(reasoning.strip()) or "think" in text.lower(),
                           "seconds": thinking["seconds"],
                           "reasoning_characters": len(reasoning),
                           "answer_excerpt": text[:200],
                           "note": "Flint decides when to reason; the glm47 parser should split it"}

    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_room",
            "description": "Look up a room by identifier",
            "parameters": {"type": "object",
                           "properties": {"room_id": {"type": "string"}},
                           "required": ["room_id"]},
        },
    }]
    called = chat(client, [{"role": "user",
                            "content": "Call lookup_room with room_id B204. Do not answer in prose."}],
                  max_tokens=256, tools=tools,
                  tool_choice={"type": "function", "function": {"name": "lookup_room"}})
    choice = (called["data"].get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    checks["tool_call"] = {"passed": bool(calls), "seconds": called["seconds"],
                           "tool_calls": calls[:2],
                           "finish_reason": choice.get("finish_reason")}

    forced = client.json("/v1/chat/completions", {
        "model": info["config"]["served_model_name"],
        "messages": [{"role": "user", "content": "Write a sentence."}],
        "max_tokens": 8, "ignore_eos": True,
        "temperature": 0.1, "top_p": 0.95, "top_k": 20,
        "stream": False,
    }, timeout=args.timeout)
    usage = forced.get("usage") or {}
    checks["forced_output"] = {"passed": usage.get("completion_tokens") == 8,
                               "completion_tokens": usage.get("completion_tokens"),
                               "prompt_tokens": usage.get("prompt_tokens")}

    tokenized = client.token_count("Flint tokenization smoke.")
    checks["tokenize"] = {"passed": tokenized > 0, "prompt_tokens": tokenized}

    # Prefix caching is part of the frozen profile: an identical follow-up should
    # report reused prompt tokens. A miss is a profile failure, not a slow request.
    document = " ".join(["The maintenance log records a routine inspection."] * 120)
    first = chat(client, [{"role": "user", "content": document + " Reply READY."}],
                 max_tokens=8, sampling={"temperature": 0.1, "top_p": 0.95, "top_k": 20})
    second = chat(client, [{"role": "user", "content": document + " Reply READY."}],
                  max_tokens=8, sampling={"temperature": 0.1, "top_p": 0.95, "top_k": 20})
    cached = ((second["data"].get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    checks["prefix_reuse"] = {"passed": isinstance(cached, int) and cached > 0,
                              "cached_tokens": cached,
                              "first_cached_tokens": ((first["data"].get("usage") or {})
                                                      .get("prompt_tokens_details") or {}).get("cached_tokens"),
                              "note": "Second identical request should hit the prefix cache"}

    report["passed"] = all(check["passed"] for check in checks.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="smoke", type=label)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-idle", action="store_true")
    args = parser.parse_args()
    require_live(args, parser)
    report = {"schema_version": 1, "kind": "smoke", "label": args.label,
              "started_at": stamp(), "run_id": uuid.uuid4().hex, "passed": False}
    path = report_path(args.label)
    try:
        run(args, report)
    except Exception as error:  # noqa: BLE001 - failed smoke runs are evidence too
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["finished_at"] = stamp()
        write_json(path, report)
    print(json.dumps({"report": str(path), "passed": report["passed"],
                      "checks": report.get("checks"), "error": report.get("error")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
