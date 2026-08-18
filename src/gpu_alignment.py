import numpy as np
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


@cuda.jit
def gpu_kernel_skeleton(input_array, output_array):
    """
    Basic CUDA kernel skeleton demonstrating 1D grid thread indexing.
    Each GPU thread processes one array element at index `pos`.
    """
    pos = cuda.grid(1)
    if pos < input_array.size:
        output_array[pos] = input_array[pos]


if __name__ == "__main__":
    test_seq = "ATGCGATCGATCG"
    arr = dna_to_gpu_array(test_seq)
    print("Original DNA:", test_seq)
    print("GPU-ready uint8 Array:", arr)
    print("Array Shape & Dtype:", arr.shape, arr.dtype)
    print("Reconstructed DNA:", gpu_array_to_dna(arr))
    print("CUDA kernel skeleton defined successfully.")
