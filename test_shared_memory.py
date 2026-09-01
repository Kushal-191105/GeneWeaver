import os
import sys
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import read_fasta, read_target
from src.gpu_alignment import (
    gpu_count_mismatches_global,
    gpu_count_mismatches_shared_mem,
    gpu_find_matches_with_mismatches
)


def test_shared_memory_kernel_parity():
    print("=" * 68)
    print("      Testing CUDA Shared Memory vs Global Memory Parity      ")
    print("=" * 68)

    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:100000]
    target = read_target("data/target.txt")

    print(f"Test Genome Length: {len(genome):,} bp | Target: {target} (len: {len(target)})")

    # 1. Exact matching (k=0)
    print("\n--- Test 1: Exact Alignment Parity (k=0) ---")
    m_global_0 = gpu_count_mismatches_global(genome, target, max_mismatches=0)
    m_shared_0 = gpu_count_mismatches_shared_mem(genome, target, max_mismatches=0)

    np.testing.assert_array_equal(m_global_0, m_shared_0)
    matches_0 = np.where(m_shared_0 == 0)[0]
    print(f"Matches found: {len(matches_0)} (Global == Shared Memory: 100% Match)")
    print("[OK] Exact match parity verified.")

    # 2. Mismatch-tolerant alignment (k=2)
    print("\n--- Test 2: Mismatch-Tolerant Alignment Parity (k=2) ---")
    m_global_2 = gpu_count_mismatches_global(genome, target, max_mismatches=2)
    m_shared_2 = gpu_count_mismatches_shared_mem(genome, target, max_mismatches=2)

    np.testing.assert_array_equal(m_global_2, m_shared_2)
    matches_2 = np.where(m_shared_2 <= 2)[0]
    print(f"Matches found with <= 2 mismatches: {len(matches_2)}")
    print("[OK] Mismatch-tolerant parity verified.")

    # 3. High-level result structures
    print("\n--- Test 3: Structured Result Collection ---")
    struct_res = gpu_find_matches_with_mismatches(genome, target, max_mismatches=2)
    print(f"Structured results collected via Shared Memory Kernel: {len(struct_res)}")
    for r in struct_res[:5]:
        print(f"  Pos: {r['position']:,} | Seq: {r['sequence']} | Mismatches: {r['mismatches']}")

    print("\n[PASSED] CUDA Shared Memory Kernel execution verified with 100% mathematical parity!")
    return True


if __name__ == "__main__":
    test_shared_memory_kernel_parity()
