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

## Measured screens

_Not yet measured._

## Telemetry and caveats

_Not yet measured._

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
