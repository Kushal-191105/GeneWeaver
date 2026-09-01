import os
import sys
import time
import pandas as pd
from src.parser import read_fasta, create_chunks, read_target
from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import (
    gpu_find_matches_with_mismatches as gpu_align,
    gpu_count_mismatches_shared_mem,
    gpu_count_mismatches_global
)
from src.gpu_device import print_gpu_device_info
from src.validator import validate_alignment_results
from src.distributed_scheduler import (
    get_available_gpus,
    partition_genome_for_workers,
    run_distributed_pipeline
)
from src.scoring import rank_off_targets
from src.visualizer import (
    format_visual_alignment,
    generate_alignment_track,
    describe_mutations,
    format_off_target_summary_card
)
from src.exporter import export_results_to_json, export_results_to_csv
from benchmark import (
    benchmark_cpu_alignment,
    benchmark_gpu_alignment,
    benchmark_shared_vs_global_memory,
    benchmark_block_dimensions,
    benchmark_distributed_scaling
)


def run_pipeline():
    print("=" * 75)
    print("      GeneWeaver: GPU-Accelerated CRISPR Alignment Engine      ")
    print("      Week 4: Shared Memory Optimization & Visual Mismatch Engine ")
    print("=" * 75)

    # 1. Hardware & Memory Hierarchy Inspection
    print("\n--- 1. Hardware & GPU Memory Hierarchy ---")
    print_gpu_device_info()
    gpus = get_available_gpus()
    print("CUDA Memory Hierarchy: On-Chip SRAM (Shared Memory: 48 KB / SM) Enabled")
    print(f"Dask Multi-GPU Scheduler: {len(gpus)} GPU(s) available for worker affinity.")

    # 2. Sequence Ingestion & Quality Cleansing
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

    # 3. Genome FASTA Ingestion & Block Chunking
    print("\n--- 3. Genome Ingestion & Chunking ---")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)
    chunk_size = 1000
    chunks = create_chunks(genome, chunk_size)
    print(f"Genome Length: {len(genome):,} bp")
    print(f"Chunk Size: {chunk_size} bp | Number of Chunks: {len(chunks):,}")

    # 4. CRISPR Target Specification
    print("\n--- 4. CRISPR Target & Biological Motif Configuration ---")
    target = read_target("data/target.txt")
    print(f"Target Sequence: {target} (Length: {len(target)} bp)")
    print("Endonuclease:    SpCas9")
    print("PAM Motif:       5'-NGG-3' (Canonical), 5'-NAG-3' (Non-canonical)")
    print("Seed Window:     Positions 11 to 20 (Proximal to PAM, High Penalty)")
    max_mismatches = 2

    # 5. Algorithmic Parity Validation (CPU vs GPU vs Dask)
    print("\n--- 5. 3-Way Parity Validation (CPU vs GPU vs Dask) ---")
    sample_val = genome[:50000]
    validate_alignment_results(sample_val, target, max_mismatches=max_mismatches, test_distributed=True)

    # 6. Comparative Benchmark (CPU vs GPU SRAM Acceleration)
    sample_len = 200000
    print(f"\n--- 6. Comparative Benchmark ({sample_len:,} bp) ---")
    sample_genome = genome[:sample_len]

    print("Running CPU baseline alignment...")
    cpu_res = benchmark_cpu_alignment(sample_genome, target, max_mismatches=max_mismatches)

    print("Running GPU SRAM accelerated alignment...")
    gpu_res = benchmark_gpu_alignment(sample_genome, target, max_mismatches=max_mismatches)

    speedup = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
    kernel_speedup = cpu_res["total_cpu_sec"] / gpu_res["kernel_execution_sec"]

    print("\n" + "=" * 65)
    print(f"{'Performance Metric':<30} | {'CPU Baseline':<15} | {'GPU Accelerated (SRAM)':<22}")
    print("-" * 65)
    print(f"{'Matches Found':<30} | {cpu_res['matches_count']:<15} | {gpu_res['matches_count']:<22}")
    print(f"{'Execution Time (Total)':<30} | {cpu_res['total_cpu_sec']*1000:<12.2f} ms | {gpu_res['total_gpu_sec']*1000:<12.2f} ms")
    print(f"{'CUDA Kernel Only':<30} | {'N/A':<15} | {gpu_res['kernel_execution_sec']*1000:<12.3f} ms")
    print("-" * 65)
    print(f"Total Speedup Factor   : {speedup:.2f}x faster")
    print(f"CUDA Kernel Speedup    : {kernel_speedup:.2f}x faster")
    print("=" * 65)

    # 7. Full Genome Distributed Alignment with Shared Memory Acceleration
    print("\n--- 7. Dask Distributed Full Genome Alignment (5.53M bp) ---")
    t0 = time.perf_counter()
    ranked_off_targets = run_distributed_pipeline(genome, target, max_mismatches=max_mismatches, n_batches=4)
    dist_time = time.perf_counter() - t0

    print(f"Distributed Alignment Completed in: {dist_time*1000:.2f} ms ({dist_time:.4f} s)")
    print(f"Total Candidate Off-Target Sites Identified: {len(ranked_off_targets)}")

    high_risk = [r for r in ranked_off_targets if r["risk_tier"] == "HIGH"]
    med_risk = [r for r in ranked_off_targets if r["risk_tier"] == "MEDIUM"]
    low_risk = [r for r in ranked_off_targets if r["risk_tier"] == "LOW"]

    print(f"Risk Stratification: {len(high_risk)} High Risk | {len(med_risk)} Medium Risk | {len(low_risk)} Low Risk")

    print("\n" + "-" * 85)
    print(f"{'Rank':<5} | {'Pos':<9} | {'Sequence':<22} | {'PAM':<5} | {'Type':<12} | {'Score':<8} | {'Risk':<12}")
    print("-" * 85)
    for r in ranked_off_targets[:10]:
        print(f"#{r['rank']:<4} | {r['position']:<9} | {r['sequence']:<22} | {r['pam']:<5} | {r['pam_type']:<12} | {r['severity_score']:<7.1f}% | {r['risk_badge']:<12}")
    print("-" * 85)

    # 8. Visual DNA Alignment Tracks (Mutated Base Pairs Highlighted in Red)
    print("\n--- 8. Visual Representation of Off-Target Mutations ---")
    for r in ranked_off_targets[:3]:
        card = format_off_target_summary_card(r, target, use_rich=False)
        print(card)

    # 9. Automated Report Export
    print("--- 9. Automated Report Export ---")
    json_path = export_results_to_json(ranked_off_targets, target, "data/crispr_off_target_report.json")
    csv_path = export_results_to_csv(ranked_off_targets, "data/crispr_off_target_summary.csv")
    print(f"Structured JSON Report : {json_path}")
    print(f"Tabular CSV Summary    : {csv_path}")

    print("\n[SUCCESS] GeneWeaver Week 4 End-to-End Pipeline Completed Successfully!")


if __name__ == "__main__":
    if "--tui" in sys.argv:
        from src.tui import GeneWeaverTUI
        app = GeneWeaverTUI()
        app.run()
    else:
        run_pipeline()