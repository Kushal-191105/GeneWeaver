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
def gpu_kernel_skeleton(input_array, output_array):
    """
    Basic CUDA kernel skeleton demonstrating 1D grid thread indexing.
    Each GPU thread processes one array element at index `pos`.
    """
    pos = cuda.grid(1)
    if pos < input_array.size:
        output_array[pos] = input_array[pos]


def test_transfers():
    """
    Verifies Host-to-Device transfer of target and genome sequences.
    """
    test_target = "GCTCGATCGATCGATCGATC"
    test_genome = "ATGCGATCGATCGCGATCGATCGATCGATCGATCGATC"

    d_target, target_len = transfer_target_to_gpu(test_target)
    d_genome, genome_len = transfer_genome_to_gpu(test_genome)

    h_target_back = d_target.copy_to_host()
    h_genome_back = d_genome.copy_to_host()

    assert gpu_array_to_dna(h_target_back) == test_target, "Target transfer mismatch!"
    assert gpu_array_to_dna(h_genome_back) == test_genome, "Genome transfer mismatch!"

    print(f"Target transferred to GPU: {len(test_target)} base pairs")
    print(f"Genome transferred to GPU: {len(test_genome)} base pairs")
    print("Host-to-Device data transfers verified successfully.")
    return True


if __name__ == "__main__":
    test_transfers()
