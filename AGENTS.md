# Working in this workspace

This repository is an evidence-first serving workspace, not an application. Read
`README.md`, `docs/METHODOLOGY.md` and `configs/production.json` before changing
anything.

## Hard rules

- **No inference without both flags.** `tests/smoke_test.py` and
  `tests/benchmark.py` require `--execute --confirm-idle`. Do not add an implicit
  or default-on path.
- **Evidence is append-only.** Reports are created exclusively under `evidence/`
  with a UTC timestamp. Never edit, rename or delete a report to make a result
  look better. Failed runs stay.
- **Never overwrite a download.** `scripts/download-model.sh` verifies sizes and
  SHA-256 values from the pinned revisions; it does not replace a file whose
  size already matches. Deleting 21 GiB of weights is a deliberate human action.
- **Do not weaken a profile to pass.** `configs/production.json`'s
  `expected_config` is the recipe contract. If the live server differs, record
  the drift and fix the cause.
- **Do not run `sudo`.** Scripts may use `sg docker` for group membership
  without a new login.
- **Pin, do not float.** The nightly image digest, model revisions and serving
  flags are frozen. A new digest or revision is a new baseline. `nightly` is a
  moving tag: the digest in `configs/production.json` is the contract.
- **Respect the recipe's validated envelope.** TP=1, BF16 KV cache, no
  torch.compile, GLM-4.7 parsers. Deviations are experiments, not defaults, and
  must be recorded as such.

## Conventions

- Python is standard-library only; tests run with `python3 -B -m unittest
  discover -s tests -v` and must not touch the network, GPU or Docker.
- Shell scripts are `bash` with `set -euo pipefail` and source `scripts/lib.sh`.
- JSON evidence keys are stable. Add fields, do not repurpose them.
- Report `null` for an unavailable measurement and explain it in a caveat; never
  substitute zero.
- Sampling uses the model's generation defaults (temperature 1.0, top_p 0.95,
  top_k 20); greedy decoding is not validated for Flint.
- Keep observations separate from failures, and state scope explicitly.

## Validation before proposing a change

```bash
python3 -B -m unittest discover -s tests -v
python3 tests/validate_setup.py --allow-missing-model
```

Only claim runtime behavior you actually exercised.
