"""
Deterministic CPU vs GPU alignment correctness tests.
Verifies that CPU matcher and GPU alignment engine produce strictly identical logical results
across all required edge cases.
"""

import pytest
from src.cpu_alignment import find_matches_with_mismatches as cpu_find_matches
from src.gpu.matcher import GPUAlignmentEngine, align_sequence
from src.parser import create_chunks
from src.gpu.device import is_cuda_available


def _normalize_results(results):
    """Sort and extract core fields for deterministic comparison."""
    return sorted(
        [
            (
                r["position"],
                r["sequence"],
                r["mismatches"],
                tuple(r["mismatch_positions"]),
            )
            for r in results
        ],
        key=lambda x: (x[0], x[2]),
    )


def assert_cpu_gpu_results_match(genome_or_chunks, target, max_mismatches=2, chunk_size=100_000):
    """Helper to verify CPU and GPU engine results match on any sequence or chunk input."""
    # 1. CPU reference results
    if isinstance(genome_or_chunks, list):
        cpu_raw = []
        offset = 0
        for chunk in genome_or_chunks:
            matches = cpu_find_matches(chunk, target, max_mismatches=max_mismatches)
            for m in matches:
                cpu_raw.append({
                    "position": offset + m["position"],
                    "sequence": m["sequence"],
                    "mismatches": m["mismatches"],
                    "mismatch_positions": m["mismatch_positions"],
                })
            offset += len(chunk)
    else:
        cpu_raw = cpu_find_matches(genome_or_chunks, target, max_mismatches=max_mismatches)

    cpu_norm = _normalize_results(cpu_raw)

    # 2. GPU engine results (backend='auto')
    engine = GPUAlignmentEngine(chunk_size=chunk_size, backend="auto")
    gpu_res = engine.align(genome_or_chunks, target, max_mismatches=max_mismatches)
    gpu_norm = _normalize_results(gpu_res)

    assert cpu_norm == gpu_norm, f"Mismatch between CPU and GPU results!\nCPU: {cpu_norm}\nGPU: {gpu_norm}"

    # 3. If CUDA is available, test backend='cuda' explicitly
    if is_cuda_available():
        engine_cuda = GPUAlignmentEngine(chunk_size=chunk_size, backend="cuda")
        cuda_res = engine_cuda.align(genome_or_chunks, target, max_mismatches=max_mismatches)
        cuda_norm = _normalize_results(cuda_res)
        assert cpu_norm == cuda_norm, f"Mismatch between CPU and CUDA results!\nCPU: {cpu_norm}\nCUDA: {cuda_norm}"


def test_exact_matches():
    genome = "AAACCCGGGTTTAAACCCGGGTTT"
    target = "AACCC"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0)


def test_one_mismatch():
    genome = "AAACCCGGGTTTAAACCAGGGTTT"
    target = "AACCC"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=1)


def test_multiple_mismatches():
    genome = "AAACCCGGGTTTAAAGGTGGGTTT"
    target = "AACCC"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=2)
    assert_cpu_gpu_results_match(genome, target, max_mismatches=3)


def test_no_matches():
    genome = "AAAAAAAAAAAAAAAAAAAA"
    target = "CCCCCCCCCC"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0)
    assert_cpu_gpu_results_match(genome, target, max_mismatches=2)


def test_target_at_beginning():
    target = "ATGCCCCAACTAAATACTAC"
    genome = target + "GGGGGGGGGGGGGGGGGGGG"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0)
    assert_cpu_gpu_results_match(genome, target, max_mismatches=2)


def test_target_at_end():
    target = "ATGCCCCAACTAAATACTAC"
    genome = "GGGGGGGGGGGGGGGGGGGG" + target
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0)
    assert_cpu_gpu_results_match(genome, target, max_mismatches=2)


def test_n_bases_ambiguity():
    genome = "ACGTNNNNACGTACGTNNNN"
    target = "ACGTN"
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0)
    assert_cpu_gpu_results_match(genome, target, max_mismatches=1)


def test_target_longer_than_chunk():
    target = "ATGCCCCAACTAAATACTAC"  # length 20
    chunk = "ATGCCCC"  # length 7
    assert_cpu_gpu_results_match(chunk, target, max_mismatches=2)


def test_multiple_chunks_and_global_position_calculation():
    # Construct a synthetic multi-chunk sequence with known hits
    chunk_1 = "AAAATGCCCCAACTAAATACTACGGG"  # hit at pos 3
    chunk_2 = "TTTATGCCCCAACTAAATACTACTTT"  # hit at local 3, global 3 + len(chunk_1)
    chunk_3 = "CCCCCCCCCCCCCCCCCCCCCCCCCC"
    chunks = [chunk_1, chunk_2, chunk_3]
    genome = "".join(chunks)
    target = "ATGCCCCAACTAAATACTAC"

    # Test alignment across explicit list of chunks
    assert_cpu_gpu_results_match(chunks, target, max_mismatches=0)
    assert_cpu_gpu_results_match(chunks, target, max_mismatches=2)

    # Test alignment across string genome with custom chunk size matching chunk boundaries
    assert_cpu_gpu_results_match(genome, target, max_mismatches=0, chunk_size=len(chunk_1))
    assert_cpu_gpu_results_match(genome, target, max_mismatches=2, chunk_size=len(chunk_1))


def test_real_dataset_match_parity():
    # Test on a slice of real genome data
    from src.parser import read_fasta, read_target
    import os

    fasta_path = "data/genome.fasta"
    target_path = "data/target.txt"

    if os.path.exists(fasta_path) and os.path.exists(target_path):
        seqs = read_fasta(fasta_path)
        genome_slice = "".join(seqs)[:10_000]
        target = read_target(target_path)
        assert_cpu_gpu_results_match(genome_slice, target, max_mismatches=2, chunk_size=2000)
