import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import gpu_find_matches_with_mismatches as gpu_align
from src.parser import read_fasta, read_target


def validate_alignment_results(genome: str, target: str, max_mismatches: int = 2):
    """
    Validates mathematical and algorithmic parity between CPU and GPU alignment engines.
    Returns True if CPU and GPU outputs match identically.
    """
    print(f"Validating alignment parity (Genome len: {len(genome)}, Target len: {len(target)}, Max mismatches: {max_mismatches})...")

    cpu_results = cpu_align(genome, target, max_mismatches=max_mismatches)
    gpu_results = gpu_align(genome, target, max_mismatches=max_mismatches)

    print(f"CPU Matches Found: {len(cpu_results)}")
    print(f"GPU Matches Found: {len(gpu_results)}")

    assert len(cpu_results) == len(gpu_results), (
        f"Match count mismatch! CPU found {len(cpu_results)}, GPU found {len(gpu_results)}"
    )

    # Compare each result entry
    for i, (cpu_m, gpu_m) in enumerate(zip(cpu_results, gpu_results)):
        assert cpu_m["position"] == gpu_m["position"], (
            f"Index {i}: Position mismatch: CPU={cpu_m['position']}, GPU={gpu_m['position']}"
        )
        assert cpu_m["sequence"] == gpu_m["sequence"], (
            f"Index {i}: Sequence mismatch at pos {cpu_m['position']}"
        )
        assert cpu_m["mismatches"] == gpu_m["mismatches"], (
            f"Index {i}: Mismatch count difference: CPU={cpu_m['mismatches']}, GPU={gpu_m['mismatches']}"
        )
        assert cpu_m["mismatch_positions"] == gpu_m["mismatch_positions"], (
            f"Index {i}: Mismatch positions difference: CPU={cpu_m['mismatch_positions']}, GPU={gpu_m['mismatch_positions']}"
        )

    print("SUCCESS: 100% parity verified between CPU and GPU alignment results!")
    return True


if __name__ == "__main__":
    # Test on real dataset
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:50000]  # Test sample for fast validation
    target = read_target("data/target.txt")

    validate_alignment_results(genome, target, max_mismatches=2)
