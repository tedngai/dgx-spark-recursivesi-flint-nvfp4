# Runbook

## Lifecycle

```bash
scripts/download-model.sh          # resume/verify 21.3 GiB of weights
scripts/start.sh                   # create or start the container, wait for /health
scripts/status.sh                  # container state, model card, headline metrics
scripts/stop.sh                    # stop, preserve the container
scripts/stop.sh --remove           # stop and delete the container
```

`scripts/start.sh` is idempotent: an existing running container is reused; a
stopped one is started. If the container exists with stale flags, remove it
(`scripts/stop.sh --remove`) and start again.

## What start does

1. Resolves Docker access (direct, then `sg docker`, then passwordless sudo).
2. Checks the NVIDIA runtime and that `models/flint-nvfp4` has the GGUF,
   config and tokenizer.
3. Generates `secrets/api_key` if absent (mode 600).
4. Refuses to start if port 8013 is held by something else.
5. Runs `vllm serve` with the frozen flags and waits up to 40 minutes for
   `/health`.

The NVFP4 GGUF is repacked losslessly at load; expect a few minutes from start
to `/health`. The engine runs without torch.compile (`--compilation-config
{"mode": 0}`) per the recipe.

## Why the nightly image

Flint support (`FlintForCausalLM`) is not in vLLM 0.32.0; the recipe requires
nightly, and the NVFP4 GGUF loader needs `gguf>=0.19.0`, which the nightly
image includes. The nightly tag is a moving target: `configs/production.json`
pins the digest pulled on 2026-10-26. If you re-pull, re-pin and re-baseline.

## Screens

```bash
export FLINT_API_KEY="$(cat secrets/api_key)"
scripts/run-screens.sh --execute --confirm-idle
ONLY=retrieval scripts/run-screens.sh --execute --confirm-idle
```

The driver stops and restarts the container before every stage so each screen
starts with an empty prefix cache and a fresh KV pool. Set
`RESTART_PER_SCREEN=0` only for debugging.

## Direct benchmark calls

```bash
python3 -B tests/benchmark.py --label short-c1-cold-20261026 \
  --mode decode --cache cold --concurrency 1 \
  --prompt-tokens 1024 --output-tokens 256 --rounds 3 \
  --execute --confirm-idle

python3 -B tests/benchmark.py --label retrieval-200000-c1-cold-20261026 \
  --mode retrieval --cache cold --concurrency 1 \
  --prompt-tokens 200000 --output-tokens 512 --rounds 3 \
  --execute --confirm-idle
```

Both scripts refuse to send inference without `--execute --confirm-idle`.

## Logs and evidence

```bash
docker logs --tail 100 -f vllm-flint-nvfp4
ls -lt evidence/ | head
python3 tests/summarize.py
```

Reports are created exclusively and never edited. `state/model.json` records
the downloaded file hashes.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Failed to import vllm._C` / device errors | Image run without GPU | Run the container with `--gpus all` |
| Literal `<think>` or `<tool_call>` in `content` | Parsers not enabled | Start with `--reasoning-parser glm47 --tool-call-parser glm47` |
| Rust frontend rejects the flags | Recipe unsupported on the Rust frontend | Keep the default Python frontend (do not set `VLLM_USE_RUST_FRONTEND`) |
| GGUF load failure | `gguf` missing or file truncated | Use the nightly image (gguf 0.19.0 included); re-run `download-model.sh --verify` |
| Startup rejects `--enable-prefix-caching` | Hybrid attention without prefix-cache support | Record it, remove the flag, note the limitation in RESULTS |
| `permission denied` on Docker | Login session lacks the group | Log in again, or let the scripts use `sg docker` |
| Port 8013 already bound | Another server is running | `scripts/stop.sh` or change `PORT` |

## Security notes

- The API is bound to `127.0.0.1` by default. Set `BIND_ADDR=0.0.0.0` only on a
  trusted network.
- `secrets/api_key` is generated locally and gitignored. vLLM receives it as
  `--api-key`, so it is visible in `docker inspect` to local users with Docker
  access.
- The BF16 repo (config/tokenizer source) is the publisher's artifact; the GGUF
  repo ships weights only. Verify the recorded SHA-256 values in
  `state/model.json` before trusting a new download.
