import time
import math
import numpy as np
from src.parser import read_fasta, read_target
from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import (
    transfer_genome_to_gpu,
    transfer_target_to_gpu,
    cuda_mismatch_count_kernel,
)
from numba import cuda


def benchmark_gpu_alignment(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Measures detailed GPU alignment execution times.
    """
    # 1. Warm-up JIT compilation on a tiny dummy sequence
    dummy_g, _ = transfer_genome_to_gpu("ATGC" * 10)
    dummy_t, _ = transfer_target_to_gpu("ATGC")
    dummy_out = cuda.device_array(37, dtype=np.uint8)
    cuda_mismatch_count_kernel[1, 32](dummy_g, dummy_t, dummy_out, 37, 4, 2)
    cuda.synchronize()

    # 2. Measure Host-to-Device transfer
    t0 = time.perf_counter()
    d_genome, genome_len = transfer_genome_to_gpu(genome)
    d_target, target_len = transfer_target_to_gpu(target)
    total_positions = genome_len - target_len + 1
    d_mismatch_counts = cuda.device_array(total_positions, dtype=np.uint8)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    # 3. Measure Kernel Execution
    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    t1 = time.perf_counter()
    cuda_mismatch_count_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_mismatch_counts, total_positions, target_len, max_mismatches
    )
    cuda.synchronize()
    t_kernel = time.perf_counter() - t1

    # 4. Measure Device-to-Host transfer & parsing
    t2 = time.perf_counter()
    h_mismatch_counts = d_mismatch_counts.copy_to_host()
    valid_positions = np.where(h_mismatch_counts <= max_mismatches)[0]
    matches_count = len(valid_positions)
    t_d2h = time.perf_counter() - t2

    total_gpu_time = t_h2d + t_kernel + t_d2h

    return {
        "genome_length": genome_len,
        "target_length": target_len,
        "total_positions": total_positions,
        "matches_count": matches_count,
        "h2d_transfer_sec": t_h2d,
        "kernel_execution_sec": t_kernel,
        "d2h_transfer_sec": t_d2h,
        "total_gpu_sec": total_gpu_time,
    }


def benchmark_cpu_alignment(genome: str, target: str, max_mismatches: int = 2):
    """
    Measures standard single-threaded CPU alignment execution time.
    """
    t0 = time.perf_counter()
    cpu_matches = cpu_align(genome, target, max_mismatches=max_mismatches)
    t_cpu = time.perf_counter() - t0
    return {
        "matches_count": len(cpu_matches),
        "total_cpu_sec": t_cpu,
    }


def compare_cpu_vs_gpu(sample_length: int = 200000):
    """
    Performs head-to-head performance audit between CPU and GPU alignment engines.
    """
    print(f"\n========== CPU vs GPU Single Benchmark (Sample: {sample_length:,} bp) ==========")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:sample_length]
    target = read_target("data/target.txt")

    print(f"Target Sequence: {target} (Len: {len(target)}) | Max Mismatches: 2")

    print("Running CPU alignment...")
    cpu_res = benchmark_cpu_alignment(genome, target, max_mismatches=2)

    print("Running GPU alignment...")
    gpu_res = benchmark_gpu_alignment(genome, target, max_mismatches=2)

    speedup_total = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
    speedup_kernel = cpu_res["total_cpu_sec"] / gpu_res["kernel_execution_sec"]

    print("\n" + "=" * 65)
    print(f"{'Performance Metric':<30} | {'CPU Baseline':<15} | {'GPU Accelerated':<15}")
    print("-" * 65)
    print(f"{'Matches Found':<30} | {cpu_res['matches_count']:<15} | {gpu_res['matches_count']:<15}")
    print(f"{'Execution Time (Total)':<30} | {cpu_res['total_cpu_sec']*1000:<12.2f} ms | {gpu_res['total_gpu_sec']*1000:<12.2f} ms")
    print(f"{'CUDA Kernel Only':<30} | {'N/A':<15} | {gpu_res['kernel_execution_sec']*1000:<12.3f} ms")
    print("-" * 65)
    print(f"Total End-to-End Speedup : {speedup_total:.2f}x faster")
    print(f"Pure Kernel Speedup       : {speedup_kernel:.2f}x faster")
    print("=" * 65 + "\n")

    return {
        "cpu": cpu_res,
        "gpu": gpu_res,
        "speedup_total": speedup_total,
        "speedup_kernel": speedup_kernel,
    }


def run_repeated_benchmark(iterations: int = 5, sample_length: int = 200000):
    """
    Executes multiple benchmark iterations to calculate statistical distribution
    (mean, median, min, max, std dev) for reliable timing.
    """
    print(f"\n========== Repeated Benchmark Suite ({iterations} Trials, {sample_length:,} bp) ==========")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:sample_length]
    target = read_target("data/target.txt")

    cpu_times = []
    gpu_total_times = []
    gpu_kernel_times = []

    # Warm up GPU
    benchmark_gpu_alignment(genome[:1000], target, max_mismatches=2)

    for i in range(1, iterations + 1):
        print(f"Trial {i}/{iterations}...", end=" ", flush=True)
        # CPU run
        c_res = benchmark_cpu_alignment(genome, target, max_mismatches=2)
        cpu_times.append(c_res["total_cpu_sec"] * 1000)

        # GPU run
        g_res = benchmark_gpu_alignment(genome, target, max_mismatches=2)
        gpu_total_times.append(g_res["total_gpu_sec"] * 1000)
        gpu_kernel_times.append(g_res["kernel_execution_sec"] * 1000)
        print(f"CPU: {cpu_times[-1]:.1f}ms | GPU: {gpu_total_times[-1]:.2f}ms (Kernel: {gpu_kernel_times[-1]:.3f}ms)")

    cpu_arr = np.array(cpu_times)
    gpu_tot_arr = np.array(gpu_total_times)
    gpu_kern_arr = np.array(gpu_kernel_times)

    avg_speedup = np.mean(cpu_arr) / np.mean(gpu_tot_arr)
    kernel_speedup = np.mean(cpu_arr) / np.mean(gpu_kern_arr)

    print("\n" + "=" * 72)
    print(f"{'Statistic':<18} | {'CPU Time (ms)':<16} | {'GPU Total (ms)':<16} | {'GPU Kernel (ms)':<16}")
    print("-" * 72)
    print(f"{'Mean':<18} | {np.mean(cpu_arr):<16.2f} | {np.mean(gpu_tot_arr):<16.3f} | {np.mean(gpu_kern_arr):<16.3f}")
    print(f"{'Median':<18} | {np.median(cpu_arr):<16.2f} | {np.median(gpu_tot_arr):<16.3f} | {np.median(gpu_kern_arr):<16.3f}")
    print(f"{'Min':<18} | {np.min(cpu_arr):<16.2f} | {np.min(gpu_tot_arr):<16.3f} | {np.min(gpu_kern_arr):<16.3f}")
    print(f"{'Max':<18} | {np.max(cpu_arr):<16.2f} | {np.max(gpu_tot_arr):<16.3f} | {np.max(gpu_kern_arr):<16.3f}")
    print(f"{'Std Dev':<18} | {np.std(cpu_arr):<16.2f} | {np.std(gpu_tot_arr):<16.3f} | {np.std(gpu_kern_arr):<16.3f}")
    print("-" * 72)
    print(f"Average Total Speedup  : {avg_speedup:.2f}x faster")
    print(f"Average Kernel Speedup : {kernel_speedup:.2f}x faster")
    print("=" * 72 + "\n")

    return {
        "cpu_mean_ms": float(np.mean(cpu_arr)),
        "gpu_total_mean_ms": float(np.mean(gpu_tot_arr)),
        "gpu_kernel_mean_ms": float(np.mean(gpu_kern_arr)),
        "avg_speedup": float(avg_speedup),
    }


if __name__ == "__main__":
    run_repeated_benchmark(iterations=3, sample_length=200000)
