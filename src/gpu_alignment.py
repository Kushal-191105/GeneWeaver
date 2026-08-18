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


@cuda.jit
def gpu_kernel_skeleton(input_array, output_array):
    """
    Basic CUDA kernel skeleton demonstrating 1D grid thread indexing.
    Each GPU thread processes one array element at index `pos`.
    """
    pos = cuda.grid(1)
    if pos < input_array.size:
        output_array[pos] = input_array[pos]


def test_simple_gpu_kernel():
    """
    Verifies GPU kernel execution by transferring data to device VRAM,
    launching the skeleton kernel, and copying the result back to host RAM.
    """
    print("Testing simple GPU kernel execution...")
    host_input = np.array([65, 84, 71, 67, 65, 84, 71, 67], dtype=np.uint8)
    n = host_input.size

    # Transfer input to device VRAM and allocate device output
    d_input = cuda.to_device(host_input)
    d_output = cuda.device_array(n, dtype=np.uint8)

    # Configure grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)

    # Launch kernel on GPU
    gpu_kernel_skeleton[blocks_per_grid, threads_per_block](d_input, d_output)
    cuda.synchronize()

    # Copy output back to host RAM
    host_output = d_output.copy_to_host()

    np.testing.assert_array_equal(host_input, host_output)
    print(f"Host Input:  {host_input}")
    print(f"GPU Output:  {host_output}")
    print("Simple GPU kernel verified successfully on GPU hardware!")
    return True


if __name__ == "__main__":
    test_seq = "ATGCGATCGATCG"
    arr = dna_to_gpu_array(test_seq)
    print("Original DNA:", test_seq)
    print("GPU-ready uint8 Array:", arr)
    print("Reconstructed DNA:", gpu_array_to_dna(arr))
    test_simple_gpu_kernel()
