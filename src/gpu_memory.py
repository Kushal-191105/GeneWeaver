import numpy as np
from numba import cuda


BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert DNA characters to integer representation.

    A = 0
    C = 1
    G = 2
    T = 3
    """

    return np.array(
        [BASE_TO_INT[base] for base in sequence.upper()],
        dtype=np.int8,
    )


def encode_target(target: str) -> np.ndarray:
    """
    Encode target DNA sequence.
    """

    return encode_sequence(target)


def copy_to_gpu(array: np.ndarray):
    """
    Transfer a NumPy array from CPU RAM to GPU VRAM.
    """

    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Check your NVIDIA GPU and CUDA installation."
        )

    return cuda.to_device(array)


def copy_to_cpu(device_array):
    """
    Transfer GPU data back to CPU RAM.
    """

    return device_array.copy_to_host()
