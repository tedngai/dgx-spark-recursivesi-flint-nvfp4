#!/usr/bin/env bash
# Run every stage in configs/workloads.json, restarting the server before each
# one so every screen starts with an empty prefix cache and fresh KV pool.
#
#   scripts/run-screens.sh --execute --confirm-idle
#   ONLY=retrieval scripts/run-screens.sh --execute --confirm-idle
#   DATE=20261026 scripts/run-screens.sh --execute --confirm-idle
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DATE="${DATE:-$(date -u +%Y%m%d)}"
ONLY="${ONLY:-}"
RESTART_PER_SCREEN="${RESTART_PER_SCREEN:-1}"
PASSTHRU=("$@")

if (( ${#PASSTHRU[@]} == 0 )); then
  die "Pass the benchmark flags explicitly, e.g. --execute --confirm-idle"
fi

mapfile -t stages < <(python3 - "${ROOT}/configs/workloads.json" <<'PY'
import json, sys
for stage in json.load(open(sys.argv[1]))["stages"]:
    print("|".join(str(stage[key]) for key in
                   ("label", "mode", "cache", "concurrency", "prompt_tokens",
                    "output_tokens", "rounds")))
PY
)

failures=0
for stage in "${stages[@]}"; do
  IFS='|' read -r label mode cache concurrency prompt output rounds <<< "$stage"
  if [[ -n "$ONLY" && "$label" != *"$ONLY"* ]]; then
    continue
  fi
  if [[ "$RESTART_PER_SCREEN" == "1" ]]; then
    scripts/stop.sh >/dev/null 2>&1 || true
    if ! scripts/start.sh >"${ROOT}/logs/start-${label}.log" 2>&1; then
      warn "restart failed for ${label}; see logs/start-${label}.log"
      failures=$((failures + 1))
      continue
    fi
  fi
  printf '=== %s-%s ===\n' "$label" "$DATE"
  if python3 -B tests/benchmark.py \
      --label "${label}-${DATE}" \
      --mode "$mode" --cache "$cache" --concurrency "$concurrency" \
      --prompt-tokens "$prompt" --output-tokens "$output" --rounds "$rounds" \
      "${PASSTHRU[@]}"; then
    printf 'PASS %s-%s\n' "$label" "$DATE"
  else
    printf 'FAIL %s-%s\n' "$label" "$DATE"
    failures=$((failures + 1))
  fi
done

printf 'DONE failures=%d\n' "$failures"
exit $(( failures > 0 ))
