import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import gpu_find_matches_with_mismatches as gpu_align
from src.distributed_scheduler import dispatch_parallel_alignment, gather_and_deduplicate_results
from src.parser import read_fasta, read_target


def validate_alignment_results(genome: str, target: str, max_mismatches: int = 2, test_distributed: bool = True):
    """
    Validates mathematical and algorithmic parity across:
    1. CPU Baseline
    2. Single-GPU CUDA Kernel
    3. Dask Distributed Parallel Batches
    Returns True if all three execution modes yield 100% identical outputs.
    """
    print(f"Validating alignment parity (Genome len: {len(genome):,}, Target len: {len(target)}, Max mismatches: {max_mismatches})...")

    # 1. CPU
    cpu_results = cpu_align(genome, target, max_mismatches=max_mismatches)
    print(f"  CPU Matches Found:         {len(cpu_results)}")

    # 2. GPU
    gpu_results = gpu_align(genome, target, max_mismatches=max_mismatches)
    print(f"  Single-GPU Matches Found:  {len(gpu_results)}")

    assert len(cpu_results) == len(gpu_results), (
        f"Match count mismatch! CPU found {len(cpu_results)}, GPU found {len(gpu_results)}"
    )

    for i, (cpu_m, gpu_m) in enumerate(zip(cpu_results, gpu_results)):
        assert cpu_m["position"] == gpu_m["position"]
        assert cpu_m["sequence"] == gpu_m["sequence"]
        assert cpu_m["mismatches"] == gpu_m["mismatches"]

    print("  [OK] Single-GPU matches CPU baseline with 100% parity.")

    # 3. Dask Distributed
    if test_distributed:
        batch_outputs = dispatch_parallel_alignment(genome, target, max_mismatches=max_mismatches, n_batches=4)
        dist_results = gather_and_deduplicate_results(batch_outputs)
        print(f"  Dask Distributed Matches:  {len(dist_results)}")

        assert len(gpu_results) == len(dist_results), (
            f"Distributed match count mismatch! Single-GPU={len(gpu_results)}, Dask={len(dist_results)}"
        )

        for i, (gpu_m, dist_m) in enumerate(zip(gpu_results, dist_results)):
            assert gpu_m["position"] == dist_m["position"]
            assert gpu_m["sequence"] == dist_m["sequence"]
            assert gpu_m["mismatches"] == dist_m["mismatches"]

        print("  [OK] Dask Distributed matches Single-GPU with 100% parity.")

    print("[SUCCESS] 100% 3-way parity verified between CPU, Single-GPU, and Dask Distributed!")
    return True


if __name__ == "__main__":
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:50000]
    target = read_target("data/target.txt")

    validate_alignment_results(genome, target, max_mismatches=2, test_distributed=True)
