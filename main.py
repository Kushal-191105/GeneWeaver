import sys
import time
import pandas as pd
from src.parser import read_fasta, create_chunks, read_target
from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import gpu_find_matches_with_mismatches as gpu_align
from src.gpu_device import print_gpu_device_info
from src.validator import validate_alignment_results
from benchmark import benchmark_cpu_alignment, benchmark_gpu_alignment


def run_pipeline():
    print("=" * 60)
    print("   GeneWeaver: GPU-Accelerated CRISPR Alignment Engine   ")
    print("=" * 60)

    # 1. Hardware Detection
    print("\n--- 1. Hardware Inspection ---")
    print_gpu_device_info()

    # 2. Sequence Analysis & Ingestion
    print("\n--- 2. Genomic Dataset Ingestion & Validation ---")
    input_file = "data/human_sequences.txt"
    data = pd.read_csv(input_file, sep="\t")

    valid_bases = set("ATGC")
    ambiguous_bases = set("N")

    def classify_sequence(sequence):
        sequence = str(sequence).upper().strip()
        characters = set(sequence)
        if characters.issubset(valid_bases):
            return "valid"
        if characters.issubset(valid_bases | ambiguous_bases):
            return "ambiguous"
        return "invalid"

    data["status"] = data["sequence"].apply(classify_sequence)
    data["length"] = data["sequence"].astype(str).str.len()

    print(f"Total Sequences: {len(data):,}")
    print(f"Valid Sequences: {(data['status'] == 'valid').sum():,}")
    print(f"Total Base Pairs: {data['length'].sum():,}")
    print(f"Average Sequence Length: {round(data['length'].mean(), 2)} bp")

    # 3. FASTA Loading & Chunking
    print("\n--- 3. Genome Chunking ---")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)
    chunk_size = 1000
    chunks = create_chunks(genome, chunk_size)
    print(f"Genome Length: {len(genome):,} bp")
    print(f"Chunk Size: {chunk_size} bp | Number of Chunks: {len(chunks):,}")

    # 4. Target Configuration
    print("\n--- 4. CRISPR Target Sequence ---")
    target = read_target("data/target.txt")
    print(f"Target: {target} (Length: {len(target)} bp)")
    max_mismatches = 2

    # 5. Parity Validation on 50k bp sample
    print("\n--- 5. CPU vs GPU Parity Validation ---")
    validate_alignment_results(genome[:50000], target, max_mismatches=max_mismatches)

    # 6. Benchmark on 200k bp sample
    sample_len = 200000
    print(f"\n--- 6. Comparative Benchmark ({sample_len:,} bp) ---")
    sample_genome = genome[:sample_len]

    print("Running CPU baseline alignment...")
    cpu_res = benchmark_cpu_alignment(sample_genome, target, max_mismatches=max_mismatches)

    print("Running GPU accelerated alignment...")
    gpu_res = benchmark_gpu_alignment(sample_genome, target, max_mismatches=max_mismatches)

    speedup = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
    kernel_speedup = cpu_res["total_cpu_sec"] / gpu_res["kernel_execution_sec"]

    print("\n" + "=" * 65)
    print(f"{'Performance Metric':<30} | {'CPU Baseline':<15} | {'GPU Accelerated':<15}")
    print("-" * 65)
    print(f"{'Matches Found':<30} | {cpu_res['matches_count']:<15} | {gpu_res['matches_count']:<15}")
    print(f"{'Execution Time (Total)':<30} | {cpu_res['total_cpu_sec']*1000:<12.2f} ms | {gpu_res['total_gpu_sec']*1000:<12.2f} ms")
    print(f"{'CUDA Kernel Only':<30} | {'N/A':<15} | {gpu_res['kernel_execution_sec']*1000:<12.3f} ms")
    print("-" * 65)
    print(f"Total Speedup Factor   : {speedup:.2f}x faster")
    print(f"CUDA Kernel Speedup    : {kernel_speedup:.2f}x faster")
    print("=" * 65)

    # 7. Full Genome GPU Alignment
    print("\n--- 7. Full Genome GPU Alignment (5.5M bp) ---")
    t0 = time.perf_counter()
    full_matches = gpu_align(genome, target, max_mismatches=max_mismatches)
    full_gpu_time = time.perf_counter() - t0

    print(f"Full Genome Alignment Completed in: {full_gpu_time*1000:.2f} ms ({full_gpu_time:.4f} s)")
    print(f"Total Off-Target Matches Identified: {len(full_matches)}")
    for m in full_matches[:10]:
        print(f"  Pos {m['position']:,} | Seq: {m['sequence']} | Mismatches: {m['mismatches']}")

    print("\n[SUCCESS] Week 2 Pipeline Integration Verified!")


if __name__ == "__main__":
    if "--tui" in sys.argv:
        from src.tui import GeneWeaverTUI
        app = GeneWeaverTUI()
        app.run()
    else:
        run_pipeline()