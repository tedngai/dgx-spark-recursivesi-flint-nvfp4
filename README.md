# DGX Spark — Recursive Flint (NVFP4) on vLLM

An evidence-first serving workspace for **Recursive Flint** (`RecursiveSI/flint`)
on one NVIDIA DGX Spark (GB10, 121 GiB unified memory). vLLM nightly serves the
NVFP4 GGUF checkpoint behind an OpenAI-compatible API, following the official
[vLLM recipe](https://recipes.vllm.ai/RecursiveSI/flint?variant=nvfp4). Every
measurement is written to `evidence/` and rendered into
[`docs/RESULTS.md`](docs/RESULTS.md).

This is a screening workspace, not a production deployment. `production_qualified`
is false everywhere. The recipe was verified on GB200; running it on GB10 is a
measured finding, not an assumption.

## The model

Flint is a 39.8B-parameter **mixture-of-experts** model (256 experts, top-6
sigmoid routing plus a shared expert) with **~2.5B parameters active per
token**, 49 hybrid-attention layers (35 sliding-window with window 513, 14 full
attention), GQA 32/4, attention sinks, and a 262,144-token context.

Because only ~2.5B parameters are read per token, decode is far less
bandwidth-bound than a dense model of similar total size. This is the reason to
expect much better single-stream latency than Nanbeige4.3-33B (33B dense FP8,
6.6 tok/s measured) on the same host.

## Production recipe (frozen)

| Setting | Value |
|---|---|
| Weights | `RecursiveSI/flint-nvfp4-gguf` @ `6ec43e1…` (21.29 GiB NVFP4 GGUF, W4A16) |
| Config / tokenizer | `RecursiveSI/flint` @ `2a24be0…` (BF16 repo) |
| Architecture | `FlintForCausalLM` |
| Engine | vLLM nightly `0.32.1rc1.dev466+ge4a5b5537` (`vllm/vllm-openai:nightly` pinned by digest; Flint support is nightly-only) |
| GGUF loader | `gguf` 0.19.0 (included in the nightly image) |
| Parallelism | TP1, one GB10 (recipe: single-GPU only; TP>1 not validated) |
| Served context | 262,144 tokens |
| Concurrency | 16 sequences, 8,192-token prefill chunks |
| KV cache | BF16 (per recipe; no FP8 KV) |
| Compilation | `--compilation-config {"mode": 0}` — no torch.compile, per recipe |
| Parsers | `glm47` reasoning and tool calls, auto tool choice, `--exclude-tools-when-tool-choice-none` |
| API | `http://127.0.0.1:8013/v1`, `/health`, `/metrics`, `/tokenize` |
| Auth | Bearer key from `secrets/api_key` (gitignored) |
| Sampling | temperature 1.0, top_p 0.95, top_k 20 (from `generation_config.json`; greedy is not validated) |

`configs/production.json` is the frozen contract; `scripts/start.sh` assembles
the flags. `configs/workloads.json` is the screen matrix (identical to the
Nanbeige workspace for comparability).

## Host prerequisites

- DGX Spark / GB10 with a working NVIDIA container runtime (`docker run --gpus all`).
- Docker access for the current user, or `sg docker` membership.
- ~60 GiB free disk for the GGUF plus ~20 GiB for the nightly image.
- Python 3.12 for the harness (standard library only).

## Quick start

```bash
cp .env.example .env                 # optional overrides
scripts/download-model.sh            # 21.3 GiB, resumes, verifies SHA-256
scripts/start.sh                     # serve and wait for /health
scripts/status.sh

python3 tests/validate_setup.py --expect-running
python3 -B -m unittest discover -s tests -v
```

Then, with an idle service:

```bash
export FLINT_API_KEY="$(cat secrets/api_key)"
python3 -B tests/smoke_test.py --label smoke-interface --execute --confirm-idle
scripts/run-screens.sh --execute --confirm-idle     # restart per screen
python3 tests/summarize.py                          # Markdown tables
```

`scripts/stop.sh` stops and preserves the container; `scripts/stop.sh --remove`
deletes it. Port 8013 is used so the Nanbeige (8012) and Qwen (8011) servers
can coexist stopped/running.

## What the harness measures

- **Decode screens** — 1,024-token prompts, 256 forced output tokens, C1/2/4/8.
- **Prefill screens** — 16K and 32K prompts, TTFT and end-to-end rate.
- **Retrieval** — 32K / 100K / 200K documents with three planted `ACCESS CODE`
  values; a run fails if any code is missing from the answer.
- **Warm reuse** — identical prompts, requiring ≥90% `cached_tokens` per
  request (exercises the prefix cache; support for Flint's hybrid attention is
  itself measured here).
- **Telemetry** — Prometheus counters/gauges every 0.2 s, host `MemAvailable`
  every second. GB10 does not report VRAM through NVML.

## Contents

```
configs/     frozen profile and screen matrix
docs/        method, runbook, results
scripts/     download, start, stop, status, screen driver
tests/       client, prompts, benchmark, smoke, validation, unit tests
evidence/    append-only JSON reports
models/      downloaded weights (gitignored)
secrets/     API key (gitignored)
state/       model verification record (gitignored)
```

## Qualification boundary

Three-round screens are screening evidence, not stable tails. No sustained
load, overload, fairness or task-quality benchmark is claimed. The NVFP4 GGUF
is a publisher-shipped artifact, but its accuracy numbers (GSM8K 0.950,
GPQA-D 0.696, MRCR ~0.46) come from the recipe's GB200 runs, not this host.
