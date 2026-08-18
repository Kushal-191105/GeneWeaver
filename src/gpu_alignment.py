import math
import numpy as np

# Ensure NumPy 2.x compatibility with Numba CUDA
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from numba import cuda


def dna_to_gpu_array(sequence: str) -> np.ndarray:
    """
    Converts a DNA nucleotide string into a contiguous uint8 NumPy array
    representing ASCII byte values for efficient GPU memory transfer.
    """
    if isinstance(sequence, str):
        cleaned_seq = sequence.strip().upper().encode("ascii")
    elif isinstance(sequence, bytes):
        cleaned_seq = sequence.strip().upper()
    else:
        raise ValueError("Sequence must be a string or bytes.")

    return np.ascontiguousarray(np.frombuffer(cleaned_seq, dtype=np.uint8))


def gpu_array_to_dna(array: np.ndarray) -> str:
    """
    Converts a uint8 NumPy array back to a DNA nucleotide string.
    """
    return bytes(array).decode("ascii")


def transfer_target_to_gpu(target: str):
    """
    Transfers a CRISPR target sequence from Host (CPU RAM) to Device (GPU VRAM).
    Returns (d_target, target_length).
    """
    target_arr = dna_to_gpu_array(target)
    d_target = cuda.to_device(target_arr)
    return d_target, len(target_arr)


def transfer_genome_to_gpu(genome: str):
    """
    Transfers a full genomic sequence from Host (CPU RAM) to Device (GPU VRAM).
    Returns (d_genome, genome_length).
    """
    genome_arr = dna_to_gpu_array(genome)
    d_genome = cuda.to_device(genome_arr)
    return d_genome, len(genome_arr)


@cuda.jit
def cuda_exact_match_kernel(genome, target, match_flags, total_positions, target_len):
    """
    CUDA Kernel: Checks exact string matching in parallel across GPU threads.
    Each thread evaluates a single genomic window starting at `pos`.
    Sets match_flags[pos] = 1 if exact match, otherwise 0.
    """
    pos = cuda.grid(1)
    if pos < total_positions:
        is_match = 1
        for i in range(target_len):
            if genome[pos + i] != target[i]:
                is_match = 0
                break
        match_flags[pos] = is_match


@cuda.jit
def cuda_mismatch_count_kernel(genome, target, mismatch_counts, total_positions, target_len, max_mismatches):
    """
    CUDA Kernel: Counts base pair mismatches for CRISPR off-target identification.
    Each thread processes window starting at `pos`. If mismatches exceed max_mismatches,
    it records 255 (invalid/no match) to optimize evaluation.
    """
    pos = cuda.grid(1)
    if pos < total_positions:
        mismatches = 0
        for i in range(target_len):
            if genome[pos + i] != target[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            mismatch_counts[pos] = mismatches
        else:
            mismatch_counts[pos] = 255


def gpu_exact_match(genome: str, target: str, threads_per_block: int = 256):
    """
    Performs exact DNA sequence alignment on the GPU.
    Returns list of starting indices where exact matches occur.
    """
    genome_len = len(genome)
    target_len = len(target)
    total_positions = genome_len - target_len + 1

    if total_positions <= 0:
        return []

    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_match_flags = cuda.device_array(total_positions, dtype=np.uint8)

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_exact_match_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_match_flags, total_positions, target_len
    )
    cuda.synchronize()

    h_match_flags = d_match_flags.copy_to_host()
    return np.where(h_match_flags == 1)[0].tolist()


def gpu_count_mismatches(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Executes parallel mismatch counting across the genome on GPU.
    Returns host NumPy array of mismatch counts per position (255 for non-matches).
    """
    genome_len = len(genome)
    target_len = len(target)
    total_positions = genome_len - target_len + 1

    if total_positions <= 0:
        return np.array([], dtype=np.uint8)

    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_mismatch_counts = cuda.device_array(total_positions, dtype=np.uint8)

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_mismatch_count_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_mismatch_counts, total_positions, target_len, max_mismatches
    )
    cuda.synchronize()

    return d_mismatch_counts.copy_to_host()


def test_gpu_mismatch_counting():
    """
    Verifies GPU mismatch counting kernel.
    """
    genome = "ATGCGATCGATCGATC"
    target = "GATC"
    # Positions:
    # 0: ATGC -> 4 mismatches (>2 -> 255)
    # 1: TGCG -> 4 mismatches (>2 -> 255)
    # 2: GCGA -> G-C-G-A vs G-A-T-C -> pos 1 (C!=A), pos 2 (G!=T), pos 3 (A!=C) = 3 mismatches (>2 -> 255)
    # 3: CGAT -> C-G-A-T vs G-A-T-C -> 4 mismatches
    # 4: GATC -> 0 mismatches
    # 8: GATC -> 0 mismatches
    # 12: GATC -> 0 mismatches
    counts = gpu_count_mismatches(genome, target, max_mismatches=2)
    valid_positions = np.where(counts <= 2)[0].tolist()
    print("Valid match positions with <= 2 mismatches:", valid_positions)
    print("Mismatch counts at valid positions:", [counts[p] for p in valid_positions])
    assert 4 in valid_positions and 8 in valid_positions and 12 in valid_positions
    print("GPU mismatch counting test PASSED.")
    return True


if __name__ == "__main__":
    test_gpu_mismatch_counting()
