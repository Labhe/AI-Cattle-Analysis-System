"""
Performance benchmarking for the AI Cattle Analysis System (Phase 14).

Measures inference speed (latency percentiles, throughput) and GPU memory
usage for any callable/model, on CPU or CUDA. Used to produce the benchmark
section of the evaluation report.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger("evaluation.benchmark")


def benchmark_inference(
    run_fn: Callable[[], Any],
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Benchmark a zero-argument inference callable.

    Args:
        run_fn: Callable performing one inference (closure over model + input).
        n_warmup: Warmup iterations (excluded from timing).
        n_runs: Timed iterations.
        device: 'cpu' or 'cuda' — controls CUDA sync and memory stats.

    Returns:
        Dict with latency stats (ms), throughput (inferences/s), and, on CUDA,
        peak memory (MB).
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(max(0, n_warmup)):
        run_fn()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        start = time.perf_counter()
        run_fn()
        if use_cuda:
            torch.cuda.synchronize()
        latencies[i] = (time.perf_counter() - start) * 1000.0  # ms

    result: Dict[str, Any] = {
        "device": "cuda" if use_cuda else "cpu",
        "n_runs": n_runs,
        "latency_ms": {
            "mean": float(latencies.mean()),
            "std": float(latencies.std()),
            "min": float(latencies.min()),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p99": float(np.percentile(latencies, 99)),
            "max": float(latencies.max()),
        },
        "throughput_per_sec": float(1000.0 / latencies.mean()) if latencies.mean() > 0 else 0.0,
    }
    if use_cuda:
        result["gpu_memory_mb"] = {
            "peak_allocated": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "peak_reserved": torch.cuda.max_memory_reserved() / (1024 ** 2),
        }
    else:
        result["gpu_memory_mb"] = None
    return result


def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    n_warmup: int = 5,
    n_runs: int = 50,
) -> Dict[str, Any]:
    """
    Benchmark a torch model's forward pass on a sample input.

    Moves the model and input to ``device``, runs under ``no_grad``, and
    reports latency, throughput, GPU memory, and parameter count.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev).eval()
    input_tensor = input_tensor.to(dev)

    @torch.no_grad()
    def run():
        return model(input_tensor)

    result = benchmark_inference(run, n_warmup=n_warmup, n_runs=n_runs,
                                 device="cuda" if use_cuda else "cpu")
    result["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    result["input_shape"] = list(input_tensor.shape)
    return result
