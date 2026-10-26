#!/usr/bin/env bash
# Shared defaults and guards for the workspace scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
fi

# NVFP4 GGUF weights; config and tokenizer come from the BF16 repository.
GGUF_REPO="${GGUF_REPO:-RecursiveSI/flint-nvfp4-gguf}"
GGUF_REV="${GGUF_REV:-6ec43e19b565fc8d560e2d04985d7724f20673bf}"
CONFIG_REPO="${CONFIG_REPO:-RecursiveSI/flint}"
CONFIG_REV="${CONFIG_REV:-2a247be0c0aec895ccb0ac7ca92229ec6df37288}"
MODEL_DIR="${MODEL_DIR:-models/flint-nvfp4}"
MODEL_PATH="${ROOT}/${MODEL_DIR}"
GGUF_FILE="${GGUF_FILE:-flint-nvfp4.gguf}"

IMAGE="${IMAGE:-vllm/vllm-openai:nightly}"
CONTAINER="${CONTAINER:-vllm-flint-nvfp4}"
PORT="${PORT:-8013}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
SERVED_NAME="${SERVED_NAME:-flint}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-glm47}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm47}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() { printf '[%s] ERROR: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; exit 1; }

# Docker access differs between an interactive admin shell and an agent shell.
# Prefer direct socket access, then the docker group via sg, then passwordless sudo.
DOCKER=()

docker_via_sg() {
  local command
  command="$(printf '%q ' docker "$@")"
  sg docker -c "$command"
}

resolve_docker() {
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
  elif command -v sg >/dev/null 2>&1 && sg docker -c 'docker info' >/dev/null 2>&1; then
    DOCKER=(docker_via_sg)
    warn "Using 'sg docker' for Docker; log in again to pick up the docker group"
  elif command -v sudo >/dev/null && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    warn "Using passwordless sudo for Docker; add the user to the docker group to avoid this"
  else
    die "Docker is not accessible. Run: sudo usermod -aG docker \$USER (then log in again), or provide passwordless sudo"
  fi
}

require_gpu_runtime() {
  local runtimes
  runtimes="$("${DOCKER[@]}" info --format '{{json .Runtimes}}')"
  if [[ "$runtimes" != *'"nvidia"'* && ! -f /etc/cdi/nvidia.yaml && ! -f /var/run/cdi/nvidia.yaml ]]; then
    die "Docker has no NVIDIA runtime or CDI spec. Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  fi
}

require_file() {
  [[ -f "$1" ]] || die "Missing $1${2:+ — $2}"
}

require_model() {
  require_file "${MODEL_PATH}/${GGUF_FILE}" "run scripts/download-model.sh"
  require_file "${MODEL_PATH}/config.json" "run scripts/download-model.sh"
  require_file "${MODEL_PATH}/tokenizer.json" "run scripts/download-model.sh"
}

stamp() { date -u +%Y%m%dT%H%M%S.%6NZ; }

container_running() {
  [[ "$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" == "true" ]]
}

container_exists() {
  "${DOCKER[@]}" inspect "$CONTAINER" >/dev/null 2>&1
}

api_key_path() {
  mkdir -p "${ROOT}/secrets"
  if [[ ! -f "${ROOT}/secrets/api_key" ]]; then
    umask 077
    openssl rand -hex 24 > "${ROOT}/secrets/api_key"
    log "Generated serving API key at secrets/api_key" >&2
  fi
  chmod 600 "${ROOT}/secrets/api_key"
  printf '%s\n' "${ROOT}/secrets/api_key"
}

api_key_value() {
  cat "$(api_key_path)"
}

wait_for_health() {
  local timeout="${1:-2400}" started
  started="$(date +%s)"
  while true; do
    if curl -fsS --max-time 5 "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! container_running; then
      warn "Container exited while waiting for health"
      return 1
    fi
    if (( $(date +%s) - started > timeout )); then
      warn "Health did not become ready within ${timeout}s"
      return 1
    fi
    sleep 5
  done
}
