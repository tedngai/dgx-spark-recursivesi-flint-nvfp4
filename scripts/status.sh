#!/usr/bin/env bash
# Report container, health, model card and the headline vLLM metrics.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

resolve_docker

if ! container_exists; then
  log "Container ${CONTAINER} does not exist"
  exit 1
fi

"${DOCKER[@]}" inspect "$CONTAINER" \
  --format 'container: {{.State.Status}} oom_killed={{.State.OOMKilled}} restarts={{.RestartCount}} started={{.State.StartedAt}}'

if curl -fsS --max-time 5 "${BASE_URL}/health" >/dev/null 2>&1; then
  printf 'health: ok (%s)\n' "$BASE_URL"
else
  printf 'health: unreachable (%s)\n' "$BASE_URL"
  exit 1
fi

KEY="$(api_key_value)"
curl -fsS -H "Authorization: Bearer ${KEY}" "${BASE_URL}/v1/models" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print("model:", json.dumps({"id": d["id"], "root": d.get("root"), "max_model_len": d.get("max_model_len")}))'

curl -fsS -H "Authorization: Bearer ${KEY}" "${BASE_URL}/metrics" \
  | python3 -c '
import json, re, sys
wanted = ("vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
          "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
          "vllm:request_success_total")
values = {}
for line in sys.stdin:
    if line.startswith("#"):
        continue
    name = line.split("{")[0].split(" ")[0]
    if name in wanted:
        try:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
print("metrics:", json.dumps(values, sort_keys=True))
'
