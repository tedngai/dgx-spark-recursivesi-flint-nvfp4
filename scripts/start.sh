#!/usr/bin/env bash
# Start the vLLM serving container and wait for /health.
#
#   scripts/start.sh                 # create, or start the preserved container
#   BIND_ADDR=0.0.0.0 scripts/start.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

resolve_docker
require_gpu_runtime
require_model

"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "Image ${IMAGE} is missing. Run: docker pull ${IMAGE}"

API_KEY="$(api_key_value)"

if container_exists; then
  if container_running; then
    log "Container ${CONTAINER} is already running"
  else
    log "Starting the preserved container ${CONTAINER}"
    "${DOCKER[@]}" start "$CONTAINER" >/dev/null
  fi
  if wait_for_health "${START_TIMEOUT:-2400}"; then
    log "Ready at ${BASE_URL}"
    exit 0
  fi
  "${DOCKER[@]}" logs --tail 80 "$CONTAINER" || true
  die "Container started but /health never became ready"
fi

if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
  die "Something already answers ${BASE_URL}/health; stop it before starting ${CONTAINER}"
fi
if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
  die "Port ${PORT} is already in use"
fi

args=(
  "/models/$(basename "$MODEL_DIR")/${GGUF_FILE}"
  --hf-config-path "/models/$(basename "$MODEL_DIR")"
  --tokenizer "/models/$(basename "$MODEL_DIR")"
  --chat-template "/models/$(basename "$MODEL_DIR")/chat_template.jinja"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
  --reasoning-parser "$REASONING_PARSER"
  --exclude-tools-when-tool-choice-none
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --uvicorn-log-level warning
  --compilation-config '{"mode": 0}'
  --api-key "$API_KEY"
)

log "Starting ${CONTAINER} from ${IMAGE} on ${BIND_ADDR:-127.0.0.1}:${PORT} -> :8000"
"${DOCKER[@]}" run -d --name "$CONTAINER" \
  --restart "$RESTART_POLICY" \
  --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -p "${BIND_ADDR:-127.0.0.1}:${PORT}:8000" \
  -v "$ROOT/models:/models:ro" \
  "$IMAGE" "${args[@]}" >/dev/null

log "Waiting for /health (loading a 21 GiB GGUF and repacking NVFP4 can take a few minutes)"
if wait_for_health "${START_TIMEOUT:-2400}"; then
  log "Ready at ${BASE_URL}"
  python3 - "$BASE_URL" "$(api_key_path)" <<'PY'
import json, sys, urllib.request
base, key_file = sys.argv[1:3]
request = urllib.request.Request(base + "/v1/models",
                                 headers={"Authorization": "Bearer " + open(key_file).read().strip()})
card = json.load(urllib.request.urlopen(request, timeout=10))["data"][0]
print(json.dumps({"id": card["id"], "root": card.get("root"),
                  "max_model_len": card.get("max_model_len")}, indent=2))
PY
  exit 0
fi

"${DOCKER[@]}" logs --tail 120 "$CONTAINER" || true
die "Container did not become healthy. Inspect: ${DOCKER[*]} logs ${CONTAINER}"
