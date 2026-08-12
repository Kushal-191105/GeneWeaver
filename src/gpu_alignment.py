import numpy as np


def align_gpu(sequence: str, target: str):
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

    results = []
    for position in range(num_windows):
        results.append({
            "position": position,
            "mismatches": int(mismatches[position]),
            "sequence": sequence[position:position + target_length],
        })

    return results

