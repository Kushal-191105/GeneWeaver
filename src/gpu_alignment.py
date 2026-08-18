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

    # Transfer data to GPU
    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_match_flags = cuda.device_array(total_positions, dtype=np.uint8)

    # Configure grid launch
    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_exact_match_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_match_flags, total_positions, target_len
    )
    cuda.synchronize()

    # Collect results from GPU to Host
    h_match_flags = d_match_flags.copy_to_host()
    match_indices = np.where(h_match_flags == 1)[0].tolist()

    return match_indices


def test_gpu_exact_matching():
    """
    Verifies GPU exact matching against sample sequence.
    """
    genome = "ATGCGATCGATCGATCGATCGATC"
    target = "GATC"
    matches = gpu_exact_match(genome, target)
    print(f"Target '{target}' exact matches at positions: {matches}")
    expected = [4, 8, 12, 16, 20]
    assert matches == expected, f"Expected {expected}, got {matches}"
    print("GPU exact matching test PASSED.")
    return True


if __name__ == "__main__":
    test_gpu_exact_matching()
