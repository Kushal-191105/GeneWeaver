"""
CPU vs GPU alignment benchmarking component for GeneWeaver.
Measures and compares execution performance across CPU and CUDA GPU backends with real timing metrics.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import numpy as np

from src.gpu.device import is_cuda_available, get_device_info
from src.gpu.encoding import encode_sequence, encode_target
from src.gpu.kernels import dna_alignment_kernel, calculate_launch_dimensions, CUDA_AVAILABLE
from src.cpu_alignment import find_matches_with_mismatches as cpu_find_matches
from src.gpu.matcher import GPUAlignmentEngine


@dataclass
class BenchmarkMetrics:
    input_size: int
    target_length: int
    max_mismatches: int
    cpu_matches_count: int
    gpu_matches_count: Optional[int]
    cpu_time_seconds: float
    gpu_total_time_seconds: Optional[float]
    gpu_warmup_time_seconds: Optional[float]
    gpu_kernel_time_seconds: Optional[float]
    gpu_transfer_time_seconds: Optional[float]
    speedup: Optional[float]
    cuda_available: bool
    device_name: Optional[str]
    correctness_verified: bool


def run_alignment_benchmark(
    genome: str,
    target: str,
    max_mismatches: int = 2,
    chunk_size: int = 100_000,
    threads_per_block: int = 128,
) -> BenchmarkMetrics:
    """
    Run an end-to-end benchmark comparing CPU alignment and GPU alignment.

    Args:
        genome: DNA sequence string to search within.
        target: CRISPR target sequence string.
        max_mismatches: Mismatch threshold (default: 2).
        chunk_size: Chunk size in bases for GPU chunking (default: 100,000).
        threads_per_block: CUDA thread block size (default: 128).

    Returns:
        BenchmarkMetrics dataclass with measured execution timings and speedup.
    """
    input_size = len(genome)
    target_length = len(target)
    cuda_active = is_cuda_available()
    dev_info = get_device_info(0) if cuda_active else {}
    device_name = dev_info.get("name") if cuda_active else None

    # 1. CPU Execution Benchmark
    t0_cpu = time.perf_counter()
    cpu_results = cpu_find_matches(
        genome=genome,
        target=target,
        max_mismatches=max_mismatches,
    )
    t1_cpu = time.perf_counter()
    cpu_time = t1_cpu - t0_cpu
    cpu_matches_count = len(cpu_results)

    gpu_warmup_time = None
    gpu_total_time = None
    gpu_kernel_time = None
    gpu_transfer_time = None
    gpu_matches_count = None
    speedup = None
    correctness_verified = False

    if cuda_active and CUDA_AVAILABLE:
        from numba import cuda

        gpu_engine = GPUAlignmentEngine(
            threads_per_block=threads_per_block,
            chunk_size=chunk_size,
            backend="cuda",
        )

        # Warm-up run to isolate Numba JIT compilation overhead
        w_t0 = time.perf_counter()
        _ = gpu_engine.align(genome[:min(len(genome), 1000)], target, max_mismatches=max_mismatches)
        cuda.synchronize()
        w_t1 = time.perf_counter()
        gpu_warmup_time = w_t1 - w_t0

        # Detailed timing run for GPU
        # Measure data transfer + kernel execution
        t0_gpu = time.perf_counter()
        gpu_results = gpu_engine.align(genome, target, max_mismatches=max_mismatches)
        cuda.synchronize()
        t1_gpu = time.perf_counter()

        gpu_total_time = t1_gpu - t0_gpu
        gpu_matches_count = len(gpu_results)

        # Micro-benchmark for isolated kernel and transfer timing on one representative chunk
        try:
            chunk_sample = genome[:chunk_size]
            encoded_sample = encode_sequence(chunk_sample)
            encoded_tgt = encode_target(target)
            num_cands = max(0, len(chunk_sample) - target_length + 1)

            # Host to Device transfer timing
            t_trans0 = time.perf_counter()
            d_c = cuda.to_device(encoded_sample)
            d_t = cuda.to_device(encoded_tgt)
            d_out = cuda.device_array(num_cands, dtype=np.uint8)
            cuda.synchronize()
            t_trans1 = time.perf_counter()

            # Kernel execution timing
            blocks, ths = calculate_launch_dimensions(num_cands, threads_per_block)
            t_k0 = time.perf_counter()
            dna_alignment_kernel[blocks, ths](
                d_c, d_t, len(chunk_sample), target_length, max_mismatches, d_out
            )
            cuda.synchronize()
            t_k1 = time.perf_counter()

            # Device to Host transfer timing
            t_trans2 = time.perf_counter()
            _ = d_out.copy_to_host()
            cuda.synchronize()
            t_trans3 = time.perf_counter()

            gpu_transfer_time = (t_trans1 - t_trans0) + (t_trans3 - t_trans2)
            gpu_kernel_time = (t_k1 - t_k0)
        except Exception:
            gpu_transfer_time = None
            gpu_kernel_time = None

        if gpu_total_time and gpu_total_time > 0:
            speedup = cpu_time / gpu_total_time

        # Verify logical equivalence
        cpu_positions = {m["position"] for m in cpu_results}
        gpu_positions = {m["position"] for m in gpu_results}
        correctness_verified = (cpu_positions == gpu_positions)

    return BenchmarkMetrics(
        input_size=input_size,
        target_length=target_length,
        max_mismatches=max_mismatches,
        cpu_matches_count=cpu_matches_count,
        gpu_matches_count=gpu_matches_count,
        cpu_time_seconds=cpu_time,
        gpu_total_time_seconds=gpu_total_time,
        gpu_warmup_time_seconds=gpu_warmup_time,
        gpu_kernel_time_seconds=gpu_kernel_time,
        gpu_transfer_time_seconds=gpu_transfer_time,
        speedup=speedup,
        cuda_available=cuda_active,
        device_name=device_name,
        correctness_verified=correctness_verified,
    )


def print_benchmark_summary(metrics: BenchmarkMetrics) -> None:
    """
    Print a structured, human-readable summary of the benchmark metrics.
    """
    print("\n" + "=" * 50)
    print("           GENEWEAVER ALIGNMENT BENCHMARK")
    print("=" * 50)
    print(f"Input Genome Length:       {metrics.input_size:,} bp")
    print(f"Target Sequence Length:    {metrics.target_length} bp")
    print(f"Maximum Mismatches:        {metrics.max_mismatches}")
    print(f"CPU Matches Found:         {metrics.cpu_matches_count}")
    print(f"CPU Execution Time:        {metrics.cpu_time_seconds:.6f} s")

    if metrics.cuda_available:
        print(f"\nCUDA Device:               {metrics.device_name}")
        print(f"GPU Matches Found:         {metrics.gpu_matches_count}")
        if metrics.gpu_warmup_time_seconds is not None:
            print(f"GPU JIT Warmup Time:       {metrics.gpu_warmup_time_seconds:.6f} s")
        if metrics.gpu_total_time_seconds is not None:
            print(f"GPU Total Alignment Time:  {metrics.gpu_total_time_seconds:.6f} s")
        if metrics.gpu_kernel_time_seconds is not None:
            print(f"GPU Kernel Execution Time: {metrics.gpu_kernel_time_seconds:.6f} s (sample chunk)")
        if metrics.gpu_transfer_time_seconds is not None:
            print(f"GPU Data Transfer Time:    {metrics.gpu_transfer_time_seconds:.6f} s (sample chunk)")
        if metrics.speedup is not None:
            print(f"\n>>> Speedup (CPU / GPU):   {metrics.speedup:.2f}x <<<")
        print(f"Correctness Verification:  {'PASSED' if metrics.correctness_verified else 'FAILED'}")
    else:
        print("\nCUDA Device:               None detected (Running in CPU mode)")
        print("GPU Benchmark:             Skipped (NVIDIA GPU hardware required)")
    print("=" * 50)
