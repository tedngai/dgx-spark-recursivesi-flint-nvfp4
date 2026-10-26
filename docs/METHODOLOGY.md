# Methodology

## Principles

1. **Evidence first.** Every inference run writes an append-only JSON report
   under `evidence/` with a UTC timestamp. Failed and partial runs stay.
2. **Capacity first.** Work completion, service health, token accounting and
   telemetry are separate observations. Queueing and low memory are reported,
   not hidden, unless `--policy no-queue` is selected.
3. **No inference without consent.** Both `--execute` and `--confirm-idle` are
   required by `tests/smoke_test.py` and `tests/benchmark.py`.
4. **Do not weaken a profile to pass.** `configs/production.json` is the frozen
   contract. A mismatch is recorded and fixed at the cause.

## Recipe adherence

The vLLM recipe for `RecursiveSI/flint` (updated 2026-10-21) was verified on one
GB200. This workspace keeps its validated envelope: TP1, BF16 KV cache, no
torch.compile, CUDA graphs on, `glm47` parsers, `--exclude-tools-when-tool-choice-none`,
and the model's sampling defaults (temperature 1.0, top_p 0.95, top_k 20 —
greedy decoding is not validated). GB10 is the one hardware change; kernel
compatibility on SM121 is a measured finding of this workspace, not an
assumption.

## Workloads

### Decode

1,024-token synthetic documents, 256 output tokens with `ignore_eos`, so every
sample generates exactly 256 tokens (reasoning tokens included — the model
decides per turn whether to reason, and reasoning counts toward `max_tokens`).
Concurrency 1, 2, 4 and 8. Each round uses a fresh prompt namespace.

### Prefill

16,384- and 32,768-token documents with 256 forced output tokens at C1. TTFT
isolates prefill cost.

### Retrieval

A synthetic document with three planted `ACCESS CODE` values at 20%, 55% and
85% of its length, followed by an instruction to list every code. The run fails
if any code is missing from the generated text (reasoning or content). This is
an in-context recall check, not document reasoning quality.

### Warm reuse

The same prompt set is sent once as warmup, then measured. A round passes only
when every request reports `cached_tokens >= 90%` of its prompt tokens. Flint's
hybrid sliding-window/full attention may or may not support prefix caching in
vLLM; this screen measures that support directly.

## Definitions

- **TTFT** — time from request start to the first non-empty text delta, not the
  first SSE event. Reasoning deltas count as text.
- **Aggregate tok/s** — completion tokens for all requests in a round divided by
  the round's wall-clock elapsed time.
- **Decode tok/s (estimate)** — tokens between the first and last text delta
  divided by that span. It is an estimate because vLLM may emit several tokens
  per SSE delta.
- **Prompt tokens** — the tokenizer's count of the raw document; the served
  chat template adds a small preamble, visible in `usage.prompt_tokens`.

## Telemetry

| Signal | Source | Cadence |
|---|---|---|
| Running / waiting requests | `vllm:num_requests_running`, `vllm:num_requests_waiting` | 0.2 s |
| KV cache usage | `vllm:kv_cache_usage_perc` (or `gpu_cache_usage_perc`) | 0.2 s |
| Prefix cache hits/queries | `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total` | 0.2 s |
| Token counters | `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` | 0.2 s |
| Host memory | `/proc/meminfo` `MemAvailable`, swap | 1 s |

NVML does not expose GB10 VRAM, so no `nvidia-smi` memory number is used.

## Screens and restarts

`scripts/run-screens.sh` restarts the container before each stage. vLLM keeps
its prefix cache and KV pool across requests, so a restart is the only clean
baseline. GGUF load and NVFP4 repack time is excluded from every measurement
because the benchmark starts only after `/health` is ready.

## Statistics

Three rounds per screen. p50 is reported; p95 is omitted below 30 samples.
Treat the numbers as screening evidence. The recipe notes batch-1 decode varies
~3–4% between sessions for this model; three rounds cannot resolve that.
