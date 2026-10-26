"""Host memory sampling for DGX Spark.

GB10 shares DRAM between CPU and GPU and NVML does not report VRAM capacity, so
`nvidia-smi` memory fields are unavailable. MemAvailable is the meaningful
budget: vLLM plans from it and the OS competes with weights, KV cache and page cache.
"""

import statistics
from pathlib import Path

MEMINFO = Path("/proc/meminfo")


def _read_meminfo(path=MEMINFO):
    values = {}
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0]) * 1024  # kB -> bytes
    return values


def sample_memory(path=MEMINFO):
    values = _read_meminfo(path)
    required = ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError("meminfo lacks " + ", ".join(missing))
    return {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "cached_bytes": values["Cached"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_free_bytes": values["SwapFree"],
    }


def summarize_memory(samples):
    if not samples:
        return {"sample_count": 0}
    available = [sample["mem_available_bytes"] for sample in samples]
    return {
        "sample_count": len(samples),
        "mem_total_bytes": samples[-1]["mem_total_bytes"],
        "minimum_mem_available_bytes": min(available),
        "median_mem_available_bytes": statistics.median(available),
        "maximum_swap_used_bytes": max(
            sample["swap_total_bytes"] - sample["swap_free_bytes"] for sample in samples
        ),
        "limitation": (
            "Sampled MemAvailable is host-wide. It cannot attribute memory to vLLM, "
            "the page cache, or another process, and one-second samples can miss peaks."
        ),
    }


def memory_status(samples, warning_bytes=None):
    if not samples:
        return {"low_memory_warning": None}
    minimum = min(sample["mem_available_bytes"] for sample in samples)
    low = None if warning_bytes is None else minimum < warning_bytes
    return {"minimum_mem_available_bytes": minimum, "low_memory_warning": low}
