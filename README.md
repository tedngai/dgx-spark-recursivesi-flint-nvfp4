# Recursive Flint (NVFP4) on ONE DGX Spark (TP=1)

Self-contained recipe for serving the `RecursiveSI/flint-nvfp4-gguf` checkpoint
(21.29 GiB NVFP4 GGUF, 39.8B-parameter MoE with **~2.5B parameters active per
token**) from a single DGX Spark's 121 GiB unified memory, via vLLM nightly with
the official [vLLM recipe](https://recipes.vllm.ai/RecursiveSI/flint)'s NVFP4
envelope: TP=1, BF16 KV cache, no torch.compile, `glm47` parsers.

Everything below was measured on this host on **2026-10-26**. Reports are in
[`evidence/`](evidence/), one JSON file per run, append-only — including the
failed runs, which are kept on purpose.

```
cp .env.example .env        # optional; every knob has a sane default
./scripts/download-model.sh # 21.3 GiB, resumable, sha256-verified
./scripts/start.sh          # ~2.5 min to /health; serves on :8013
./scripts/stop.sh           # stop and preserve the container
```

`start.sh` never downloads anything — it checks `models/flint-nvfp4` for the
GGUF, config and tokenizer and fails fast if they are absent. Budget ~25 GiB of
free disk: 21.3 GiB for the checkpoint plus the ~20 GiB nightly image. The API
binds `127.0.0.1` by default with a generated key in `secrets/api_key`; set
`BIND_ADDR=0.0.0.0` only on a trusted network.

## Why this model is interesting on a Spark

Flint reads only **~2.5B of its 39.8B parameters per token** (256 experts,
top-6 sigmoid routing plus a shared expert). Decode on GB10 is nowhere near the
bandwidth wall that pins a dense model, and the measurement agrees:

| | Flint NVFP4 | Nanbeige4.3-33B (dense FP8) | ratio |
|---|---:|---:|---:|
| single-stream decode (C1) | **77.7 tok/s** | 6.59 tok/s | **11.8×** |
| aggregate at 8 streams | **298.1 tok/s** | 42.11 tok/s | **7.1×** |
| 200K prefill TTFT | **72.9 s** | 625.2 s | **8.6×** |

The other side of that bargain is attention: 35 of Flint's 49 layers use a
513-token sliding window, 14 run full attention. Prefill stays near-linear, but
deep-document recall pays — see [the 200K retrieval failure](#retrieval-and-the-200k-recall-failure).

## Measured profile

`.env` ships **262,144 context, 16 sequences, 8,192-token prefill chunks,
`gpu_memory_utilization=0.85`, BF16 KV, prefix caching and chunked prefill on,
`--compilation-config {"mode": 0}`** — the recipe's validated envelope, nothing
tuned past it. Sampling defaults come from the model's own
`generation_config.json` (temperature 1.0, top_p 0.95, top_k 20); greedy
decoding is not validated for Flint, and vLLM confirms the override at boot.

| | |
|---|---|
| Weights on the GPU | 21.86 GiB, loaded in 92.5 s (NVFP4 GGUF, Marlin W4A16 — see below) |
| Time to `/health` | **156 s** end-to-end (load + profile + KV + graphs) |
| Available KV cache | 79.87 GiB = **2,505,978 tokens** (9.56× a full 262k request) |
| CUDA graphs | decode-only (`FULL_DECODE_ONLY`, sizes 1–32), 0.59 GiB |
| Prefix cache | **works** — 1,008/1,021 prompt tokens cached (98.7%) warm |
| Host MemAvailable after `/health` | ~96 GiB |
| Concurrency at 262,144 tokens | 9.56× cold |

**GB10 has no native FP4 tensor cores.** vLLM says so at load: *"Your GPU does
not have native support for FP4 computation... weight-only FP4 compression will
be used leveraging the Marlin kernel."* The NVFP4 checkpoint therefore runs as
Marlin W4A16 — 4-bit weight bandwidth, dequantized compute. The recipe's
native-FP4 GB200 numbers are a different regime; what follows is what a Spark
buyer actually gets. One quirk of the GGUF: layernorm weights are FP32 on disk
and cast to BF16 (benign warnings at load).

### Decode, by concurrent streams

1,024-token prompts, 256 forced output tokens, three rounds per level, cold
server per screen (restart before every stage, so no prefix cache carries
over). Median of three rounds:

| streams | TTFT p50 | aggregate | per stream | decode est. | Nanbeige aggregate |
|---|---:|---:|---:|---:|---:|
| 1 | 204 ms | **77.7 tok/s** | 77.7 tok/s | 82.1 tok/s | 6.59 |
| 2 | 374 ms | **126.3 tok/s** | 63.2 tok/s | 68.9 tok/s | 12.97 |
| 4 | 541 ms | **215.9 tok/s** | 54.1 tok/s | 60.5 tok/s | 24.82 |
| 8 | 994 ms | **298.1 tok/s** | 37.3 tok/s | 43.3 tok/s | 42.11 |
| 1 (warm) | **69 ms** | **81.8 tok/s** | 81.8 tok/s | 82.9 tok/s | 6.64 |

The recipe targets a low-latency decode path for up to 8 concurrent requests on
GB200; on GB10 the 8-stream column is reachable and aggregate scales 3.8× from
C1 to C8, but per-stream rate falls ~52% (77.7 → 37.3) — batch-8 decode is
compute- and scheduling-bound long before the KV pool notices. For reference,
Qwen3.8-Flare (earlier workspace) measured 37.35 tok/s single-stream and 102.94
aggregate on this host; Flint beats both by ~2× at C1 and ~2.9× at C8.

Warm TTFT of 69 ms is the prefix cache doing its job on identical prompts — the
open question in the recipe, whether vLLM prefix caching supports Flint's
hybrid sliding-window/full attention, is answered **yes** on this build.

### Prefill

Forced output, C1, cold:

| context | TTFT p50 | aggregate | Nanbeige TTFT |
|---|---:|---:|---:|
| 16,384 | **1.99 s** | 44.8 tok/s | 22.82 s |
| 32,768 | **4.64 s** | 28.7 tok/s | 50.63 s |

### Retrieval and the 200K recall failure

A synthetic document with three `ACCESS CODE` values planted at 20%, 55% and
85% of its length; a round passes only if all three appear in the answer.

| context | TTFT p50 | recall | passed | Nanbeige TTFT |
|---|---:|---|---|---:|
| 32,768 | **4.67 s** | 3/3 every round | yes | 50.51 s |
| 100,000 | **23.17 s** | 3/3 every round | yes | 214.47 s |
| 200,000 (512 budget) | **72.92 s** | 3/3, then **2/3** | **no** | 625.21 s |
| 200,000 (2,048 budget) | **72.91 s** | **2/3** | **no** | — |

**The headline failure, stated plainly: at 200K, Flint misses the code planted
at 85% of the document in 2 of 3 attempts.** The miss is consistent — always
the deepest code, never the first two — and it is *recall*, not budget: the
2,048-token re-run shows the model stopping at 559 tokens (EOS) after listing
only two codes, with the 512-run's one passing round having found all three.
Nanbeige recalled 3/3 at every size on the same host.

Why it matters: with 35 of 49 layers on a 513-token window, deep-document
recall rests on the 14 full-attention layers. Whatever the mechanism, on this
host the 200K recall is not reliable, and that is the caveat to weigh against
the 12× decode win. ≤100K measured clean in every round.

TTFT scaling itself is excellent and near-linear: 0.2 s (1K) → 2.0 s (16K) →
4.6 s (32K) → 23.2 s (100K) → 72.9 s (200K). Nanbeige grew superlinearly
(12.4× TTFT for a 6.1× prompt); the sliding window plausibly contains the
quadratic term, and the 200K recall miss above is the other face of that
design.

### Reasoning tokens are unavoidable

Flint decides per turn whether to reason — there is no `enable_thinking:
false` switch like Qwen/Nanbeige. On retrieval tasks it reasons heavily:
471–511 of 512 tokens at ≤100K (the answers fit by a hair), and a tight
`max_tokens` budget can be consumed entirely by reasoning before any answer
appears. Two early smoke runs captured exactly this and are retained in
`evidence/` (`smoke-interface`, `smoke-interface-rerun` — each failed one
check for this reason; the final run passes **7/7**: chat, stream, reasoning
split by the `glm47` parser, tool call, forced output, tokenize, prefix
reuse). Any output budget you set must include the reasoning.

## Configuration

Precedence is **environment > `.env` > built-in default in `scripts/lib.sh`**,
so any knob can be overridden per launch:

```
MAX_NUM_SEQS=8 ./scripts/start.sh     # one launch
```

| knob | default | notes |
|---|---|---|
| `PORT` | `8013` | host port; container is always 8000. 8011/8012 stay free for the Qwen/Nanbeige servers |
| `BIND_ADDR` | `127.0.0.1` | set `0.0.0.0` only on a trusted network |
| `MAX_MODEL_LEN` | `262144` | the checkpoint's native context |
| `MAX_NUM_SEQS` | `16` | recipe low-latency path covers up to 8 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | prefill chunk |
| `GPU_MEMORY_UTILIZATION` | `0.85` | leaves ~96 GiB host MemAvailable after boot |
| `REASONING_PARSER` / `TOOL_CALL_PARSER` | `glm47` | per recipe; the Rust frontend does not support these flags, so keep the default Python frontend |
| `RESTART_POLICY` | `unless-stopped` | `no` disables restarts |
| `START_TIMEOUT` | `2400` | seconds to wait for `/health` |

`configs/production.json` is the frozen contract: model revisions (GGUF
`6ec43e1…`, config/tokenizer `2a247be…`), the nightly image digest
(`6b90105…`, vLLM `0.32.1rc1.dev466+ge4a5b5537`, gguf 0.19.0, torch
2.14.0+cu130) and every serving flag. **Flint support is nightly-only** — vLLM
0.32.0 has no `FlintForCausalLM` — and `nightly` is a moving tag, so the digest
is the baseline: if you re-pull, re-pin and re-baseline.

## What the harness measures

- **Decode screens** — 1,024-token prompts, 256 forced output tokens, C1/2/4/8,
  one freshly restarted server per screen.
- **Prefill screens** — 16K and 32K prompts; TTFT isolates prefill cost.
- **Retrieval** — three planted codes; a run fails if any is missing.
- **Warm reuse** — a round passes only at ≥90% `cached_tokens`.
- **Telemetry** — vLLM counters every 0.2 s, host `MemAvailable` every second
  (GB10 does not report VRAM through NVML).

Both `tests/smoke_test.py` and `tests/benchmark.py` refuse to send inference
without `--execute --confirm-idle`. Screens:

```bash
export FLINT_API_KEY="$(cat secrets/api_key)"
python3 -B tests/smoke_test.py --label smoke --execute --confirm-idle
scripts/run-screens.sh --execute --confirm-idle   # 10 stages, restart per stage
python3 tests/summarize.py                        # markdown tables from evidence/
```

## Qualification boundary

Three-round screens are screening evidence, not stable tails; the recipe notes
batch-1 decode varies ~3–4% between sessions and three rounds cannot resolve
that. No sustained load, overload, fairness or task-quality benchmark is
claimed. `production_qualified` stays **false**. The NVFP4 GGUF is a
publisher-shipped artifact, but its accuracy numbers (GSM8K 0.950, GPQA-D
0.696, MRCR ~0.46) come from the recipe's GB200 runs, not this host.

## Contents

```
configs/     frozen profile (production.json) and screen matrix (workloads.json)
docs/        method, runbook, full results history
scripts/     download, start, stop, status, screen driver
tests/       client, prompts, benchmark, smoke, validation, unit tests
evidence/    append-only JSON reports, failed runs included
models/      downloaded weights (gitignored)
secrets/     API key (gitignored)
state/       model verification record (gitignored)
```

## Credits

Recursive Flint is `RecursiveSI/flint`; the NVFP4 GGUF is the publisher's
`RecursiveSI/flint-nvfp4-gguf` (config and tokenizer come from the BF16 repo).
Serving follows the official [vLLM recipe](https://recipes.vllm.ai/RecursiveSI/flint).
Benchmark methodology and the comparison baselines come from the sibling
workspaces on this host:
[Qwen3.8-Flare](https://github.com/tedngai/dgx-spark-qwen38-next-flare) and
[Nanbeige4.3-33B](https://github.com/tedngai/dgx-spark-nanbeige-4.3-33B).
