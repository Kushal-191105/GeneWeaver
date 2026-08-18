import time
import math
import numpy as np
from src.parser import read_fasta, read_target
from src.gpu_alignment import (
    transfer_genome_to_gpu,
    transfer_target_to_gpu,
    cuda_mismatch_count_kernel,
    gpu_find_matches_with_mismatches,
)
from numba import cuda


def benchmark_gpu_alignment(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Measures detailed GPU alignment execution times:
    - Data transfer (H2D)
    - Kernel execution
    - Data transfer (D2H)
    - Total end-to-end GPU time
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


def run_gpu_benchmark():
    print("========== GPU Alignment Benchmark ==========")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)
    target = read_target("data/target.txt")

    print(f"Genome Length: {len(genome):,} base pairs")
    print(f"Target Sequence: {target} (Length: {len(target)})")
    print(f"Max Mismatches: 2")
    print("Running GPU alignment benchmark...")

    results = benchmark_gpu_alignment(genome, target, max_mismatches=2)

    print("\n--- GPU Benchmark Results ---")
    print(f"Matches Found:           {results['matches_count']}")
    print(f"Host-to-Device Transfer: {results['h2d_transfer_sec'] * 1000:.3f} ms ({results['h2d_transfer_sec']:.6f} s)")
    print(f"CUDA Kernel Execution:   {results['kernel_execution_sec'] * 1000:.3f} ms ({results['kernel_execution_sec']:.6f} s)")
    print(f"Device-to-Host Transfer: {results['d2h_transfer_sec'] * 1000:.3f} ms ({results['d2h_transfer_sec']:.6f} s)")
    print(f"Total GPU Execution Time:{results['total_gpu_sec'] * 1000:.3f} ms ({results['total_gpu_sec']:.6f} s)")
    print("=============================================")


if __name__ == "__main__":
    run_gpu_benchmark()
