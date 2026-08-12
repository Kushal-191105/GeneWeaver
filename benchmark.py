"""
GeneWeaver CPU vs GPU Alignment Benchmark Script.
Executes sequence alignment across CPU and GPU backends, measuring execution time,
data transfers, kernel time, speedup, and correctness.
"""

import sys
import os
from src.parser import read_fasta, read_target
from src.gpu.benchmark import run_alignment_benchmark, print_benchmark_summary
from src.gpu.device import print_device_info, is_cuda_available


def main():
    print("==================================================")
    print("     GeneWeaver CRISPR Alignment Benchmark")
    print("==================================================")

    # 1. Device Inspection
    print_device_info()

    # 2. Load FASTA Genome and Target Sequence
    fasta_path = "data/genome.fasta"
    target_path = "data/target.txt"

    if not os.path.exists(fasta_path):
        print(f"Error: Genome file '{fasta_path}' not found.")
        sys.exit(1)

    if not os.path.exists(target_path):
        print(f"Error: Target file '{target_path}' not found.")
        sys.exit(1)

    print(f"\nLoading genome from '{fasta_path}'...")
    sequences = read_fasta(fasta_path)
    genome = "".join(sequences)
    print(f"Total genome length loaded: {len(genome):,} bp")

    target = read_target(target_path)
    print(f"Loaded CRISPR target: {target} (length: {len(target)} bp)")

    # 3. Run Benchmark (test on full genome or a subset if desired)
    # By default, runs full benchmark
    print("\nRunning alignment benchmark with max_mismatches=2...")
    metrics = run_alignment_benchmark(
        genome=genome,
        target=target,
        max_mismatches=2,
        chunk_size=100_000,
    )

    # 4. Print Summary
    print_benchmark_summary(metrics)


if __name__ == "__main__":
    main()
