import numpy as np


def cuda_available() -> bool:
    """True only when numba + a working NVIDIA CUDA device are present."""
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def gpu_backend_name() -> str:
    return "cuda" if cuda_available() else "numpy (CUDA not available)"


def align_gpu(sequence: str, target: str):
    """
    Sliding-window DNA alignment on the GPU backend.

    Uses the CUDA kernel when an NVIDIA device is available and falls
    back to the vectorized numpy implementation otherwise, so the same
    `--mode gpu` command runs everywhere.
    """
    if cuda_available():
        return align_cuda(sequence, target)

    return align_numpy(sequence, target)


def align_cuda(sequence: str, target: str):
    """
    Sliding-window alignment using the numba CUDA kernel.

    One CUDA thread handles one alignment position.
    """
    from numba import cuda

    from src.cuda_kernels import alignment_kernel
    from src.gpu_memory import encode_sequence, encode_target

    sequence = sequence.upper()
    target = target.upper()

    target_length = len(target)

    if target_length == 0:
        raise ValueError("Target cannot be empty.")

    if len(sequence) < target_length:
        return []

    num_windows = len(sequence) - target_length + 1

    sequence_device = cuda.to_device(encode_sequence(sequence))
    target_device = cuda.to_device(encode_target(target))
    counts_device = cuda.device_array(num_windows, dtype=np.int32)

    threads_per_block = 256
    blocks = (num_windows + threads_per_block - 1) // threads_per_block

    alignment_kernel[blocks, threads_per_block](
        sequence_device,
        target_device,
        counts_device,
    )

    mismatches = counts_device.copy_to_host()

    return _build_results(sequence, target_length, num_windows, mismatches)


def align_numpy(sequence: str, target: str):
    """
    Vectorized sliding-window DNA alignment (numpy-based).

    Calculates Hamming distance between the target
    and every possible window in the sequence.

    Returns:
        list[dict]
    """

    sequence = sequence.upper()
    target = target.upper()

    target_length = len(target)

    if target_length == 0:
        raise ValueError("Target cannot be empty.")

    if len(sequence) < target_length:
        return []

    seq_array = np.frombuffer(sequence.encode(), dtype=np.uint8)
    target_array = np.frombuffer(target.encode(), dtype=np.uint8)

    num_windows = len(seq_array) - target_length + 1

    window_indices = (
        np.arange(target_length)[None, :] + np.arange(num_windows)[:, None]
    )
    windows = seq_array[window_indices]

    mismatches = np.sum(windows != target_array[None, :], axis=1)

    return _build_results(sequence, target_length, num_windows, mismatches)


def _build_results(sequence, target_length, num_windows, mismatches):
    results = []

    for position in range(num_windows):
        results.append({
            "position": position,
            "mismatches": int(mismatches[position]),
            "sequence": sequence[position:position + target_length],
        })

    return results

