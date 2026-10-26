# Results and experiment history

## 2026-10-26 — workspace prepared

- Created this workspace for **Recursive Flint** on one DGX Spark (GB10, SM121,
  121 GiB unified memory), following the official vLLM recipe's NVFP4 variant.
- Froze the contract in `configs/production.json`: NVFP4 GGUF
  (`RecursiveSI/flint-nvfp4-gguf` @ `6ec43e1…`, 21.29 GiB) with config and
  tokenizer from `RecursiveSI/flint` @ `2a247be…`; vLLM nightly
  (`0.32.1rc1.dev466+ge4a5b5537`, image digest `6b90105…`, gguf 0.19.0, torch
  2.14.0+cu130); TP1; 262,144-token context; 16 sequences; BF16 KV cache; no
  torch.compile; `glm47` parsers; sampling defaults temperature 1.0 / top_p
  0.95 / top_k 20.
- Ported the evidence-first harness from the Nanbeige workspace; the screen
  matrix is identical for comparability.
- vLLM 0.32.0 (the previous workspace's engine) does **not** support Flint;
  nightly is required. This is recorded as a constraint, not a workaround.

No benchmark has been run at this point. Tables below are filled from
`evidence/` only after a screen completes.

## 2026-10-26 — first startup (healthy)

`scripts/start.sh` reached `/health` in **156 s** end-to-end (92.5 s weight load
of 21.86 GiB, 39.7 s profile/KV/graph warmup). Findings from the container log:

- **GB10 has no native FP4 tensor cores.** vLLM logs: *"Your GPU does not have
  native support for FP4 computation... weight-only FP4 compression will be used
  leveraging the Marlin kernel."* The NVFP4 checkpoint therefore runs as Marlin
  W4A16 — 4-bit weight bandwidth, but dequantized compute. The recipe's
  native-FP4 GB200 advantage does not transfer; that gap is part of what this
  workspace measures.
- **CUDA graphs still active under `mode: 0`.** vLLM overrode
  `FULL_AND_PIECEWISE` → `FULL_DECODE_ONLY` (decode-only graphs, sizes
  1–32). Prefill runs eager, decode is graph-captured (0.59 GiB).
- **`--enable-prefix-caching` was accepted** by the hybrid-attention model
  (7 padding layers noted, ≤20% KV waste). Real reuse was proven by the smoke:
  960/960 `cached_tokens` on an identical follow-up.
- **KV cache: 79.87 GiB → 2,505,978 tokens**, i.e. 9.56 concurrent full-length
  262K requests. BF16 KV, `LBNH` layout.
- **Engine confirmed the sampling defaults** from `generation_config.json`
  (temperature 1.0, top_p 0.95, top_k 20) as its own defaults.
- GGUF layernorm weights are FP32-on-disk and cast to BF16 (benign warnings).
- The model decides per turn whether to reason; a tight `max_tokens` budget can
  be consumed entirely by reasoning (smoke budgets were raised accordingly).

Smoke interface (`evidence/…smoke-interface-final.json`): **7/7 PASS** — chat,
stream, reasoning split by the `glm47` parser, tool call, forced output,
tokenize, prefix reuse. Two earlier runs are retained as evidence of the budget
issue described above.

## Measured screens (2026-10-26)

All numbers are medians of 3 rounds; reports in `evidence/`. The screen matrix
matches the Nanbeige4.3-33B workspace (decode/prefill/retrieval identical), so
the comparison column is like-for-like on the same host.

### Decode — 1,024-token prompts, 256 forced output tokens

| Screen | Aggregate tok/s | Per-req tok/s | Decode est. | TTFT p50 | Nanbeige4.3-33B |
|---|---:|---:|---:|---:|---:|
| C1 cold | **77.65** | 77.67 | 82.14 | 0.204 s | 6.59 |
| C2 cold | **126.25** | 63.20 | 68.85 | 0.374 s | 12.97 |
| C4 cold | **215.87** | 54.07 | 60.48 | 0.541 s | 24.82 |
| C8 cold | **298.06** | 37.32 | 43.30 | 0.994 s | 42.11 |
| C1 warm | **81.79** | 81.80 | 82.90 | 0.069 s | 6.64 |

Single-stream decode is **~12× Nanbeige** (77.7 vs 6.59 tok/s) — the 2.5B-active
MoE does what the architecture promised on GB10's ~273 GB/s. Scaling to C8
reaches 298 tok/s aggregate (37.3 per stream), below the low-latency ≤8-request
path the recipe targets, but 7× the dense model's best.

### Prefill — forced output, C1 cold

| Screen | TTFT p50 | Aggregate tok/s | Nanbeige TTFT |
|---|---:|---:|---:|
| 16,384-prompt | **1.99 s** | 44.82 | 22.82 s |
| 32,768-prompt | **4.64 s** | 28.71 | 50.63 s |

### Retrieval — three planted ACCESS CODEs, C1 cold

| Screen | TTFT p50 | Recall | Passed | Nanbeige TTFT |
|---|---:|---|---|---:|
| 32,768-prompt | **4.67 s** | 3/3 every round | yes | 50.51 s |
| 100,000-prompt | **23.17 s** | 3/3 every round | yes | 214.47 s |
| 200,000-prompt (512 budget) | **72.92 s** | 3/3, then **2/3** | **no** | 625.21 s |
| 200,000-prompt (2,048 budget) | **72.91 s** | **2/3** | **no** | — |

**Finding — 200K recall failure.** At 200K, Flint misses the code planted at
85% of the document in 2 of 3 attempts (both budget variants; the one passing
round found all 3). The miss is consistent: always the deepest code, never the
first two. Recall is 3/3 at ≤100K. With 35 of 49 layers on a 513-token sliding
window, deep-document recall rests on the 14 full-attention layers; whatever
the mechanism, on this host the 200K recall is not reliable. This is the
headline caveat for any long-context use.

Budget note: Flint cannot disable reasoning (no `enable_thinking` switch, unlike
Qwen/Nanbeige), and it reasons heavily on retrieval tasks (471–511 of 512
tokens). The 512-token budget is comparable to Nanbeige's matrix, but the 2,048
re-run shows the 200K failure is recall, not budget: the model stopped at 559
tokens (EOS) after listing only two codes.

TTFT scaling is excellent: 0.2 s (1K) → 2.0 s (16K) → 4.6 s (32K) → 23.2 s
(100K) → 72.9 s (200K). The 200K prefill is **8.6× faster than Nanbeige**
(72.9 s vs 625 s).

### Warm reuse — prefix caching works for the hybrid attention

`short-c1-warm`: 1,008 of 1,021 prompt tokens cached (98.7%) on every round,
TTFT 0.069 s. The smoke prefix check agrees (960/960). The recipe's open
question — whether vLLM prefix caching supports Flint's hybrid
sliding-window/full attention — is answered **yes** on this build.

## Telemetry and caveats

- **No native FP4 on GB10.** The NVFP4 checkpoint runs through Marlin
  weight-only W4A16 kernels (vLLM logs the capability warning). Numbers below
  are what a Spark buyer gets; the recipe's native-FP4 GB200 numbers are a
  different regime.
- **Decode estimate vs aggregate.** `decode est.` counts tokens between first
  and last text delta; `aggregate` divides the round wall-clock. At C1 they are
  nearly identical (82.1 vs 77.7), meaning streaming overhead is small.
- **TTFT scaling is near-linear here** (1K→0.2 s, 32K→4.6 s, 200K→72.9 s);
  Nanbeige showed superlinear growth (12.4× TTFT for 6.1× prompt). The sliding
  window on 35 of 49 layers plausibly contains the quadratic term; the 200K
  recall miss above is the other face of that design.
- **Reasoning tokens are unavoidable.** Flint decides per turn whether to
  reason; retrieval answers ride after 130–511 reasoning tokens. Any output
  budget must include them (unlike Qwen/Nanbeige, there is no
  `enable_thinking: false`).
- **KV cache ceiling.** 2,505,978 tokens total → 9.56× concurrency at full
  262K context; the 7 padding layers waste up to 20% of planned KV.
- **Batch-1 variance.** The recipe notes 3–4% between sessions; three rounds
  cannot resolve it.

## Verdict

On one DGX Spark, Flint NVFP4 via vLLM nightly is the fastest long-context
serving profile measured in this series of workspaces:

- **12× single-stream decode** vs the dense 33B FP8 model (77.7 vs 6.59 tok/s).
- **7× aggregate at C8** (298 vs 42.11 tok/s), with room to the KV ceiling.
- **Prefill 8.6× faster at 200K** (72.9 s vs 625 s TTFT), near-linear scaling.
- **Prefix caching works** with the hybrid attention (98.7% cached tokens warm).
- **But 200K recall is unreliable** (misses the deepest of three planted codes
  in 2 of 3 attempts) where the dense model recalled 3/3 at all sizes —
  a real accuracy/efficiency trade-off, screened, not benchmarked.

`production_qualified` remains **false**.

## Known limits

- **GB10 is not the recipe's hardware.** The recipe was verified on GB200
  (HBM3e, ~8 TB/s). GB10's ~273 GB/s unified memory and lower compute will
  scale the headline numbers down; the architecture (2.5B active parameters)
  should still make it much faster than a dense 33B model.
- **Nightly-only.** Flint support is not in any released vLLM; the pinned
  nightly digest is the baseline, and `nightly` is a moving tag.
- **Single GPU only.** TP>1 and expert parallelism fall back to stock vLLM
  layers and are not validated by the recipe.
- **Three-round screens.** Screening evidence only. The recipe notes batch-1
  decode varies ~3–4% between sessions; three rounds cannot resolve that.
