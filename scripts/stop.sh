#!/usr/bin/env bash
# Stop the serving container. The container is preserved so a restart is quick.
#
#   scripts/stop.sh            # stop, keep the container
#   scripts/stop.sh --remove   # stop and delete the container
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

resolve_docker
REMOVE=0
[[ "${1:-}" == "--remove" ]] && REMOVE=1

if ! container_exists; then
  log "Container ${CONTAINER} does not exist"
  exit 0
fi

if container_running; then
  log "Stopping ${CONTAINER}"
  "${DOCKER[@]}" stop --time 60 "$CONTAINER" >/dev/null
else
  log "Container ${CONTAINER} is already stopped"
fi

if (( REMOVE )); then
  "${DOCKER[@]}" rm "$CONTAINER" >/dev/null
  log "Removed ${CONTAINER}"
else
  log "Preserved ${CONTAINER}; scripts/start.sh starts it again"
fi
