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


@cuda.jit
def gpu_kernel_skeleton(input_array, output_array):
    """
    Basic CUDA kernel skeleton demonstrating 1D grid thread indexing.
    Each GPU thread processes one array element at index `pos`.
    """
    pos = cuda.grid(1)
    if pos < input_array.size:
        output_array[pos] = input_array[pos]


def test_target_transfer():
    """
    Verifies Host-to-Device transfer of target sequence.
    """
    test_target = "GCTCGATCGATCGATCGATC"
    d_target, target_len = transfer_target_to_gpu(test_target)
    h_target_back = d_target.copy_to_host()
    reconstructed = gpu_array_to_dna(h_target_back)
    assert reconstructed == test_target, "Target transfer mismatch!"
    print(f"Target transferred to GPU VRAM successfully: {reconstructed} (Length: {target_len})")
    return True


if __name__ == "__main__":
    test_seq = "ATGCGATCGATCG"
    arr = dna_to_gpu_array(test_seq)
    print("Original DNA:", test_seq)
    print("GPU-ready uint8 Array:", arr)
    print("Reconstructed DNA:", gpu_array_to_dna(arr))
    test_target_transfer()
