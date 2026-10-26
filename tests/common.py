"""Standard-library utilities; no inference or deployment on import."""

import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8013"
DEFAULT_CONTAINER = "vllm-flint-nvfp4"
DEFAULT_PROFILE = "production"
API_KEY_ENV = "FLINT_API_KEY"
SECRET_KEY = re.compile(r"api.?key|password|secret|authorization|^hf_token$|^github_pat$", re.I)
SECRET_VALUE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b|\bgh[pousr]_[A-Za-z0-9]{16,}\b|\bgithub_pat_[A-Za-z0-9_]+\b")

# vLLM metric names have changed across releases; accept both spellings.
ALIASES = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "swapped": ("vllm:num_requests_swapped",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
    "prefix_hits": ("vllm:prefix_cache_hits_total",),
    "prefix_queries": ("vllm:prefix_cache_queries_total",),
    "success": ("vllm:request_success_total",),
    "preemptions": ("vllm:num_preemptions_total",),
}
COUNTERS = {"prompt_tokens", "generation_tokens", "prefix_hits", "prefix_queries",
            "success", "preemptions"}
METRIC_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)\s*$")


def scrub(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SECRET_KEY.search(k) else scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[REDACTED]", value)
    return value


def label(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Label must be 1-80 letters, digits, underscores or hyphens")
    return value


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json(path, data):
    """Exclusive creation: reports and baselines are never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(scrub(data), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def report_path(name):
    return ROOT / "evidence" / f"{label(name)}-{stamp()}.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path, chunk=8 << 20):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def parse_prometheus(text):
    """Return {metric_name: [values]}, summing away label dimensions."""
    series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, _, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            series.setdefault(name, []).append(value)
    return series


def normalize_metrics(series):
    result = {}
    for key, names in ALIASES.items():
        values = next((series[name] for name in names if name in series), None)
        if values is None:
            result[key] = None
        elif key in COUNTERS:
            result[key] = sum(values)
        elif key == "kv_usage":
            result[key] = max(values)
        else:
            result[key] = sum(values)
    result["series_count"] = len(series)
    return result


class Client:
    """Minimal vLLM client: OpenAI surface plus `/health`, `/metrics` and `/tokenize`."""

    def __init__(self, base_url=DEFAULT_URL, timeout=600, api_key=None):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/")):
            raise ValueError("Use an http(s) server origin, without credentials or a path")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV)
        if api_key is None:
            key_file = ROOT / "secrets" / "api_key"
            if key_file.is_file():
                api_key = key_file.read_text().strip()
        self.api_key = api_key

    def open(self, endpoint, payload=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + endpoint, data=data, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:2000]
            raise urllib.error.HTTPError(
                error.url, error.code, f"{error.reason}: {detail}", error.headers, None,
            ) from None

    def json(self, endpoint, payload=None, timeout=None):
        with self.open(endpoint, payload, timeout) as response:
            return json.load(response)

    def text(self, endpoint, timeout=None):
        with self.open(endpoint, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")

    def health(self):
        try:
            with self.open("/health", timeout=10) as response:
                return {"status": "ok", "http_status": response.status}
        except urllib.error.HTTPError as error:
            return {"status": "unhealthy", "http_status": error.code, "detail": error.reason}

    def card(self):
        data = self.json("/v1/models")["data"]
        if len(data) != 1:
            raise ValueError("/v1/models must advertise exactly one loaded model")
        return data[0]

    def info(self):
        """Model card, health and the normalized contract used by the harness."""
        health = self.health()
        card = self.card()
        config = {
            "status": "ready" if health["status"] == "ok" else "unhealthy",
            "served_model_name": card["id"],
            "root": card.get("root"),
            "context_length": card.get("max_model_len"),
            "owned_by": card.get("owned_by"),
        }
        if not isinstance(config["context_length"], int) or config["context_length"] < 1:
            raise ValueError("Model card does not advertise max_model_len")
        return {"health": health, "model": card, "config": config}

    def metrics(self):
        return normalize_metrics(parse_prometheus(self.text("/metrics", timeout=10)))

    def token_count(self, text):
        """Exact tokenizer count from `/tokenize`; no GPU work is submitted."""
        data = self.json("/tokenize", {"model": self.card()["id"], "prompt": text}, timeout=120)
        count = data.get("count")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Tokenization returned no count")
        return count


def check_budget(config, concurrency, prompt_tokens, output_tokens, max_concurrency):
    if min(concurrency, prompt_tokens, output_tokens) < 1:
        raise ValueError("Concurrency and token counts must be positive")
    if config.get("status") != "ready":
        raise ValueError("Server is not ready")
    if not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError("Profile does not declare max_num_seqs")
    if concurrency > max_concurrency:
        raise ValueError("Client concurrency exceeds the profile's max_num_seqs")
    context = config["context_length"]
    if prompt_tokens + output_tokens > context:
        raise ValueError("Prompt plus output exceeds the loaded context window")


def load_profile(name=DEFAULT_PROFILE):
    path = ROOT / "configs" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown profile: {name}")
    return json.loads(path.read_text())


def verify_profile(config, profile_name=DEFAULT_PROFILE):
    """Compare the live contract with the frozen profile; unknown keys are ignored."""
    profile = load_profile(profile_name)
    expected = profile.get("expected_config") or {}
    differences = {k: {"expected": v, "actual": config.get(k)}
                   for k, v in expected.items() if config.get(k) != v}
    if differences:
        raise ValueError("Live server does not match profile: " + json.dumps(differences))
    return profile


def require_live(args, parser):
    if not args.execute or not args.confirm_idle:
        parser.error("No inference sent: both --execute and --confirm-idle are required")


@contextlib.contextmanager
def experiment_lock():
    """Coordinates this workspace's tests, not other clients or other workspaces."""
    with (ROOT / ".benchmark.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another experiment in this workspace is running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    position = (len(values) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)
