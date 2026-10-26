#!/usr/bin/env python3
"""Read-only preflight for the Nanbeige4.3-33B vLLM workspace. Sends no inference."""

import argparse
import json
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from common import (Client, DEFAULT_CONTAINER, DEFAULT_PROFILE, DEFAULT_URL, ROOT,
                    file_sha256, load_profile)

REQUIRED_FILES = (
    "configs/production.json",
    "configs/workloads.json",
    "scripts/lib.sh",
    "scripts/start.sh",
    "scripts/stop.sh",
    "scripts/download-model.sh",
    "tests/benchmark.py",
    "tests/smoke_test.py",
)


def run(command, timeout=30):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"ok": result.returncode == 0, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(), "returncode": result.returncode}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


_DOCKER = None


def docker_argv(*args):
    """Use direct Docker access, then the docker group via sg, then plain docker."""
    global _DOCKER
    if _DOCKER is None:
        if run(["docker", "info"], timeout=10)["ok"]:
            _DOCKER = "direct"
        elif shutil.which("sg") and run(["sg", "docker", "-c", "docker info"], timeout=10)["ok"]:
            _DOCKER = "sg"
        else:
            _DOCKER = "unavailable"
    if _DOCKER == "sg":
        return ["sg", "docker", "-c", shlex.join(["docker", *args])]
    return ["docker", *args]


def docker_access():
    info = run(docker_argv("info", "--format", "{{json .Runtimes}}"))
    if not info["ok"]:
        detail = info.get("stderr") or info.get("error") or ""
        return {"ok": False, "detail": detail,
                "remediation": ("sudo usermod -aG docker $USER (then log in again), or run the "
                                "scripts with sudo")}
    runtimes = json.loads(info["stdout"] or "{}")
    gpu_runtime = any(name in runtimes for name in ("nvidia", "nvidia-container-runtime"))
    cdi = any(Path(path).exists() for path in ("/etc/cdi/nvidia.yaml", "/var/run/cdi/nvidia.yaml"))
    return {"ok": gpu_runtime or cdi, "runtimes": sorted(runtimes), "cdi_spec": cdi,
            "remediation": ("sudo nvidia-ctk runtime configure --runtime=docker && "
                            "sudo systemctl restart docker") if not (gpu_runtime or cdi) else None}


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def model_state(profile, hash_model):
    directory = ROOT / profile["model"]["local_dir"]
    record_path = ROOT / "state" / "model.json"
    gguf = profile["model"].get("gguf_file", "flint-nvfp4.gguf")
    state = {"present": directory.is_dir(), "path": str(directory)}
    if not state["present"]:
        return state
    # Flint ships a single NVFP4 GGUF; there are no safetensors shards.
    state["gguf_present"] = (directory / gguf).is_file()
    state["gguf_bytes"] = (directory / gguf).stat().st_size if state["gguf_present"] else 0
    state["total_gib"] = round(state["gguf_bytes"] / (1 << 30), 2)
    state["config_present"] = (directory / "config.json").is_file()
    state["tokenizer_present"] = (directory / "tokenizer.json").is_file()
    state["template_present"] = (directory / "chat_template.jinja").is_file()
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        state["record_gguf_revision"] = record.get("gguf_revision")
        state["record_matches_profile"] = (
            record.get("gguf_revision") == profile["model"]["gguf_revision"])
        expected = {item["path"]: item for item in record.get("files", [])}
        missing = [path for path in expected if not (directory / path).is_file()]
        wrong = [path for path, item in expected.items()
                 if (directory / path).is_file()
                 and (directory / path).stat().st_size != item["size_bytes"]]
        state["missing_files"] = missing
        state["size_mismatches"] = wrong
        state["state_ok"] = (not missing and not wrong
                             and state["record_matches_profile"]
                             and state["gguf_present"] and state["config_present"]
                             and state["tokenizer_present"] and state["template_present"])
        if hash_model:
            hashes = {}
            for path, item in expected.items():
                target = directory / path
                if not target.is_file():
                    continue
                hashes[path] = file_sha256(target)
            state["hash_mismatches"] = [path for path, value in hashes.items()
                                        if value != expected[path]["sha256"]]
            state["hashed_files"] = len(hashes)
    else:
        state["state_ok"] = None
        state["state_note"] = "state/model.json missing; run scripts/download-model.sh"
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--allow-missing-model", action="store_true",
                        help="Pre-download check: weights and API key may be absent")
    parser.add_argument("--hash-model", action="store_true",
                        help="Also hash every model file and compare state/model.json (slow)")
    parser.add_argument("--expect-running", action="store_true",
                        help="Post-start check: the service holds the port and host memory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    profile = load_profile(DEFAULT_PROFILE)
    checks = {}

    checks["python"] = {"passed": sys.version_info[:2] >= (3, 12),
                        "version": sys.version.split()[0]}
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    checks["workspace"] = {"passed": not missing, "missing": missing, "root": str(ROOT)}

    docker = docker_access()
    checks["docker"] = {"passed": docker["ok"], **docker}

    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    checks["gpu"] = {"passed": gpu["ok"], "devices": gpu.get("stdout", "").splitlines()}

    image = run(docker_argv("image", "inspect", profile["engine"]["image"],
                            "--format", "{{index .RepoDigests 0}}"))
    digest = (image.get("stdout") or "").split("@")[-1]
    checks["image"] = {"passed": image["ok"] and digest == profile["engine"]["image_digest"],
                       "tag": profile["engine"]["image"],
                       "digest": digest or None,
                       "expected_digest": profile["engine"]["image_digest"]}

    state = model_state(profile, args.hash_model)
    state_ok = state.get("state_ok")
    present = bool(state.get("present") and state.get("gguf_present"))
    checks["model"] = {"passed": (present and state_ok is not False) or args.allow_missing_model,
                       **state}

    checks["api_key"] = {"passed": (ROOT / "secrets/api_key").is_file()
                         or args.allow_missing_model,
                         "path": str(ROOT / "secrets/api_key")}

    usage = shutil.disk_usage(ROOT)
    checks["disk"] = {"passed": usage.free >= (profile["host"]["minimum_free_gib"] << 30)
                      if "host" in profile else usage.free >= (200 << 30),
                      "free_gib": round(usage.free / (1 << 30), 1)}
    meminfo = Path("/proc/meminfo").read_text()
    available_kb = next(int(line.split()[1]) for line in meminfo.splitlines()
                        if line.startswith("MemAvailable:"))
    minimum_gib = (profile.get("host") or {}).get("minimum_available_gib", 60)
    checks["memory"] = {"passed": (available_kb << 10) >= (minimum_gib << 30),
                        "available_gib": round((available_kb << 10) / (1 << 30), 1),
                        "minimum_gib": minimum_gib}

    port_bound = not port_free(args.port)
    checks["port"] = {"passed": port_bound if args.expect_running else not port_bound,
                      "port": args.port, "bound": port_bound}

    client = Client(args.base_url, timeout=10)
    try:
        info = client.info()
        checks["service"] = {"passed": info["config"]["status"] == "ready", "running": True,
                             "config": info["config"]}
        expected = profile.get("expected_config") or {}
        drift = {key: {"expected": value, "actual": info["config"].get(key)}
                 for key, value in expected.items() if info["config"].get(key) != value}
        checks["service_profile"] = {"passed": not drift, "drift": drift}
    except Exception as error:  # noqa: BLE001 - service may legitimately be stopped
        checks["service"] = {"passed": not args.expect_running, "running": False,
                             "detail": f"{type(error).__name__}: {error}"}
        checks["service_profile"] = {"passed": not args.expect_running,
                                     "drift": None, "detail": "service not reachable"}

    if args.expect_running:
        # These two gates protect a cold start: free memory and a free port. While
        # the service runs they invert, so record the serving observation instead
        # of reusing the pre-start threshold.
        ready = checks["service"].get("running") and checks["service"].get("passed")
        checks["memory"].update({"passed": ready, "mode": "serving",
                                 "note": "Host MemAvailable while serving; the "
                                         f"{minimum_gib} GiB gate applies before start."})
        checks["port"].update({"passed": bool(ready and port_bound), "mode": "serving",
                               "note": "The port must be held by the serving container."})

    container = run(docker_argv("inspect", args.container,
                                "--format", "{{.State.Status}} {{.State.OOMKilled}} {{.RestartCount}}"))
    checks["container"] = {"passed": True, "state": container.get("stdout") or "absent"}

    passed = all(check["passed"] for check in checks.values())
    result = {"kind": "validate-setup", "passed": passed, "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, check in checks.items():
            mark = "PASS" if check["passed"] else "FAIL"
            detail = {k: v for k, v in check.items() if k != "passed"}
            print(f"[{mark}] {name}: {json.dumps(detail, default=str)[:400]}")
        print(f"\n{'PASS' if passed else 'FAIL'}: setup validation")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
