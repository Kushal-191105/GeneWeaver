import time
import math
import numpy as np
from src.parser import read_fasta, read_target
from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import (
    transfer_genome_to_gpu,
    transfer_target_to_gpu,
    cuda_mismatch_count_kernel,
    cuda_shared_mem_alignment_kernel,
    gpu_count_mismatches_global,
    gpu_count_mismatches_shared_mem
)
from src.distributed_scheduler import (
    dispatch_parallel_alignment,
    gather_and_deduplicate_results,
    run_distributed_pipeline
)
from numba import cuda


def benchmark_gpu_alignment(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Measures detailed GPU alignment execution times (using Shared Memory kernel).
    """
    dummy_g, _ = transfer_genome_to_gpu("ATGC" * 10)
    dummy_t, _ = transfer_target_to_gpu("ATGC")
    dummy_out = cuda.device_array(37, dtype=np.uint8)
    cuda_shared_mem_alignment_kernel[1, 32](dummy_g, dummy_t, dummy_out, 37, 4, 2, 40)
    cuda.synchronize()

    t0 = time.perf_counter()
    d_genome, genome_len = transfer_genome_to_gpu(genome)
    d_target, target_len = transfer_target_to_gpu(target)
    total_positions = genome_len - target_len + 1
    d_mismatch_counts = cuda.device_array(total_positions, dtype=np.uint8)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t0

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    t1 = time.perf_counter()
    cuda_shared_mem_alignment_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_mismatch_counts, total_positions, target_len, max_mismatches, genome_len
    )
    cuda.synchronize()
    t_kernel = time.perf_counter() - t1

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


def benchmark_block_dimensions(genome: str, target: str, block_sizes: list = None, max_mismatches: int = 2):
    """
    Evaluates GPU kernel performance across varying CUDA block dimensions (threads per block)
    to identify optimal occupancy on NVIDIA Ampere (RTX 3050) architecture.
    """
    if block_sizes is None:
        block_sizes = [64, 128, 256, 512]

    print("\n========== CUDA Block Dimension & Occupancy Optimization ==========")
    print(f"Genomic Sequence: {len(genome):,} bp | Target: {target} (len: {len(target)})")

    d_genome, genome_len = transfer_genome_to_gpu(genome)
    d_target, target_len = transfer_target_to_gpu(target)
    total_positions = genome_len - target_len + 1
    d_out = cuda.device_array(total_positions, dtype=np.uint8)

    cuda_shared_mem_alignment_kernel[math.ceil(total_positions / 256), 256](
        d_genome, d_target, d_out, total_positions, target_len, max_mismatches, genome_len
    )
    cuda.synchronize()

    results = []
    for b_dim in block_sizes:
        grid_dim = math.ceil(total_positions / b_dim)
        timings = []
        for _ in range(5):
            t0 = time.perf_counter()
            cuda_shared_mem_alignment_kernel[grid_dim, b_dim](
                d_genome, d_target, d_out, total_positions, target_len, max_mismatches, genome_len
            )
            cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000)

        mean_t = float(np.mean(timings))
        min_t = float(np.min(timings))
        throughput = (len(genome) / 1e6) / (mean_t / 1000)
        results.append({
            "block_dim": b_dim,
            "grid_dim": grid_dim,
            "mean_ms": mean_t,
            "min_ms": min_t,
            "throughput_mbps": throughput
        })

    print("-" * 75)
    print(f"{'Threads / Block':<16} | {'Grid Blocks':<14} | {'Mean Time (ms)':<16} | {'Throughput (Mbp/s)':<18}")
    print("-" * 75)
    for r in results:
        print(f"{r['block_dim']:<16} | {r['grid_dim']:<14} | {r['mean_ms']:<16.3f} | {r['throughput_mbps']:<18.2f}")
    print("-" * 75)

    best = min(results, key=lambda x: x["mean_ms"])
    print(f"Optimal Configuration: {best['block_dim']} threads/block ({best['throughput_mbps']:.2f} Mbp/s)\n")
    return results


def benchmark_shared_vs_global_memory(genome: str, target: str, max_mismatches: int = 2, iterations: int = 5, threads_per_block: int = 256):
    """
    Directly audits CUDA Shared Memory (on-chip SRAM) vs Global Memory (VRAM) latency.
    """
    print(f"\n========== CUDA Shared Memory vs Global Memory Latency Audit ==========")
    print(f"Genomic Sequence: {len(genome):,} bp | Target: {target} (len: {len(target)}) | Trials: {iterations}")

    d_genome, genome_len = transfer_genome_to_gpu(genome)
    d_target, target_len = transfer_target_to_gpu(target)
    total_positions = genome_len - target_len + 1
    d_out_global = cuda.device_array(total_positions, dtype=np.uint8)
    d_out_shared = cuda.device_array(total_positions, dtype=np.uint8)
    blocks_per_grid = math.ceil(total_positions / threads_per_block)

    cuda_mismatch_count_kernel[blocks_per_grid, threads_per_block](d_genome, d_target, d_out_global, total_positions, target_len, max_mismatches)
    cuda_shared_mem_alignment_kernel[blocks_per_grid, threads_per_block](d_genome, d_target, d_out_shared, total_positions, target_len, max_mismatches, genome_len)
    cuda.synchronize()

    global_times = []
    shared_times = []

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        cuda_mismatch_count_kernel[blocks_per_grid, threads_per_block](d_genome, d_target, d_out_global, total_positions, target_len, max_mismatches)
        cuda.synchronize()
        global_times.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        cuda_shared_mem_alignment_kernel[blocks_per_grid, threads_per_block](d_genome, d_target, d_out_shared, total_positions, target_len, max_mismatches, genome_len)
        cuda.synchronize()
        shared_times.append((time.perf_counter() - t1) * 1000)

    g_mean = float(np.mean(global_times))
    s_mean = float(np.mean(shared_times))
    sram_speedup = g_mean / s_mean if s_mean > 0 else 1.0

    print("-" * 75)
    print(f"{'Memory Architecture':<28} | {'Mean Time (ms)':<16} | {'Min Time (ms)':<16} | {'Throughput (Mbp/s)':<18}")
    print("-" * 75)
    print(f"{'Global VRAM (High Latency)':<28} | {g_mean:<16.3f} | {np.min(global_times):<16.3f} | {(len(genome)/1e6)/(g_mean/1000):<18.2f}")
    print(f"{'Shared Memory (On-Chip SRAM)':<28} | {s_mean:<16.3f} | {np.min(shared_times):<16.3f} | {(len(genome)/1e6)/(s_mean/1000):<18.2f}")
    print("-" * 75)
    print(f"SRAM Latency Advantage : {sram_speedup:.2f}x faster execution from on-chip cache\n")

    return {
        "global_mean_ms": g_mean,
        "shared_mean_ms": s_mean,
        "sram_speedup": sram_speedup
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


def benchmark_distributed_scaling(genome: str, target: str, max_mismatches: int = 2, batch_counts: list = None):
    """
    Benchmarks distributed Dask chunk scheduling across varying batch partition counts.
    """
    if batch_counts is None:
        batch_counts = [1, 2, 4, 8]

    print("\n========== Dask Distributed Scaling Benchmark ==========")
    print(f"Dataset Size: {len(genome):,} bp | Target: {target} (len: {len(target)})")

    results = []
    baseline_time = None

    for n_b in batch_counts:
        t0 = time.perf_counter()
        raw_outputs = dispatch_parallel_alignment(genome, target, max_mismatches=max_mismatches, n_batches=n_b)
        unique_matches = gather_and_deduplicate_results(raw_outputs)
        duration = time.perf_counter() - t0

        if baseline_time is None:
            baseline_time = duration

        speedup_rel = baseline_time / duration if duration > 0 else 1.0
        throughput_mbps = (len(genome) / 1e6) / duration if duration > 0 else 0.0

        results.append({
            "batches": n_b,
            "duration_ms": duration * 1000,
            "matches": len(unique_matches),
            "throughput_mbps": throughput_mbps,
            "speedup_rel": speedup_rel
        })

    print("-" * 75)
    print(f"{'Partitions':<12} | {'Time (ms)':<14} | {'Matches':<10} | {'Throughput (Mbp/s)':<20} | {'Scaling':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['batches']:<12} | {r['duration_ms']:<14.2f} | {r['matches']:<10} | {r['throughput_mbps']:<20.2f} | {r['speedup_rel']:<10.2f}x")
    print("-" * 75)

    return results


def run_full_system_benchmark():
    """
    Comprehensive full-genome (5.53M bp) benchmark audit:
    Evaluates Single-GPU Global Memory vs Shared Memory vs Dask Distributed.
    """
    print("\n" + "=" * 75)
    print("      GeneWeaver: Full-Genome Comprehensive System Benchmark      ")
    print("=" * 75)

    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)
    target = read_target("data/target.txt")

    print(f"Full Genome Length: {len(genome):,} bp | Target: {target} (len: {len(target)})")

    # 1. Global Memory Single GPU Run
    t0 = time.perf_counter()
    res_global = gpu_count_mismatches_global(genome, target, max_mismatches=2)
    time_global = (time.perf_counter() - t0) * 1000

    # 2. Shared Memory Single GPU Run
    t1 = time.perf_counter()
    res_shared = gpu_count_mismatches_shared_mem(genome, target, max_mismatches=2)
    time_shared = (time.perf_counter() - t1) * 1000

    # 3. Dask Distributed 4-Batch Run
    t2 = time.perf_counter()
    dist_results = run_distributed_pipeline(genome, target, max_mismatches=2, n_batches=4)
    time_dask = (time.perf_counter() - t2) * 1000

    print("\n" + "-" * 75)
    print(f"{'Architecture / Pipeline':<30} | {'Time (ms)':<12} | {'Throughput (Mbp/s)':<20}")
    print("-" * 75)
    print(f"{'Single-GPU (Global VRAM)':<30} | {time_global:<12.2f} | {(len(genome)/1e6)/(time_global/1000):<20.2f}")
    print(f"{'Single-GPU (Shared Memory SRAM)':<30} | {time_shared:<12.2f} | {(len(genome)/1e6)/(time_shared/1000):<20.2f}")
    print(f"{'Dask Distributed (4 Batches)':<30} | {time_dask:<12.2f} | {(len(genome)/1e6)/(time_dask/1000):<20.2f}")
    print("-" * 75 + "\n")


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

    print("Running GPU alignment (Shared Memory)...")
    gpu_res = benchmark_gpu_alignment(genome, target, max_mismatches=2)

    speedup_total = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
    speedup_kernel = cpu_res["total_cpu_sec"] / gpu_res["kernel_execution_sec"]

    print("\n" + "=" * 65)
    print(f"{'Performance Metric':<30} | {'CPU Baseline':<15} | {'GPU Accelerated (SRAM)':<22}")
    print("-" * 65)
    print(f"{'Matches Found':<30} | {cpu_res['matches_count']:<15} | {gpu_res['matches_count']:<22}")
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


if __name__ == "__main__":
    seqs = read_fasta("data/genome.fasta")
    g = "".join(seqs)[:200000]
    t = read_target("data/target.txt")

    compare_cpu_vs_gpu(sample_length=200000)
    benchmark_shared_vs_global_memory(g, t, max_mismatches=2, iterations=5)
    benchmark_block_dimensions(g, t, block_sizes=[64, 128, 256, 512])
    benchmark_distributed_scaling(g, t, max_mismatches=2, batch_counts=[1, 2, 4])
    run_full_system_benchmark()
