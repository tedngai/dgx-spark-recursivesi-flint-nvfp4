#!/usr/bin/env bash
# Download the Flint NVFP4 GGUF and the BF16 repo's config/tokenizer into models/.
#
#   scripts/download-model.sh            # resume or verify an existing copy
#   scripts/download-model.sh --verify   # hash only, no network
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

# Each line: repo|path|size|lfs-sha256|revision
manifest="$(python3 "${ROOT}/scripts/manifest.py" "$GGUF_REPO" "$GGUF_REV" "$GGUF_FILE")
$(python3 "${ROOT}/scripts/manifest.py" "$CONFIG_REPO" "$CONFIG_REV")"
[[ -n "$manifest" ]] || die "Empty manifest"

mkdir -p "$MODEL_PATH" state
while IFS='|' read -r repo path size oid rev; do
  [[ -n "$path" ]] || continue
  destination="${MODEL_PATH}/${path}"
  if [[ -f "$destination" && "$(stat -c %s "$destination")" == "$size" ]]; then
    log "present ${path}"
    continue
  fi
  if (( VERIFY_ONLY )); then
    die "${path} is missing or has the wrong size"
  fi
  log "downloading ${repo}/${path} ($(numfmt --to=iec --suffix=B "$size"))"
  curl -fL --retry 5 --retry-delay 5 -C - \
    "https://huggingface.co/${repo}/resolve/${rev}/${path}" \
    -o "$destination"
  actual="$(stat -c %s "$destination")"
  [[ "$actual" == "$size" ]] || die "${path}: expected ${size} bytes, got ${actual}"
done <<< "$manifest"

log "Hashing every file and comparing LFS object ids"
manifest="$manifest" GGUF_REPO="$GGUF_REPO" GGUF_REV="$GGUF_REV" CONFIG_REPO="$CONFIG_REPO" CONFIG_REV="$CONFIG_REV" \
python3 - "$MODEL_PATH" "${ROOT}/state/model.json" <<'PY'
import hashlib, json, os, sys, time
directory, state_path = sys.argv[1:3]
env = os.environ
files = []
for line in env["manifest"].splitlines():
    if not line.strip():
        continue
    repo, path, size, oid, rev = line.split("|")
    target = os.path.join(directory, path)
    digest = hashlib.sha256()
    with open(target, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if oid and actual != oid:
        raise SystemExit(f"{path}: SHA-256 {actual} != {oid}")
    files.append({"path": path, "size_bytes": int(size), "sha256": actual,
                  "lfs_verified": bool(oid)})
record = {
    "schema_version": 1,
    "kind": "model",
    "gguf_repo": env["GGUF_REPO"],
    "gguf_revision": env["GGUF_REV"],
    "config_repo": env["CONFIG_REPO"],
    "config_revision": env["CONFIG_REV"],
    "local_dir": os.path.relpath(directory),
    "file_count": len(files),
    "total_bytes": sum(item["size_bytes"] for item in files),
    "lfs_verified_count": sum(1 for item in files if item["lfs_verified"]),
    "files": files,
    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(state_path, "w") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps({"files": len(files), "total_gib": round(record["total_bytes"] / 2**30, 2),
                  "lfs_verified": record["lfs_verified_count"], "state": state_path}, indent=2))
PY
log "Model ready at ${MODEL_DIR}"
