"""
CUDA kernels and launch configuration utilities for parallel DNA sequence alignment and mismatch detection using Numba CUDA.
"""

from typing import Tuple
import math

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cuda = None

# Sentinel constant representing no match / mismatch count exceeding threshold
NO_MATCH_SENTINEL = 255


if CUDA_AVAILABLE:
    @cuda.jit
    def dna_alignment_kernel(
        chunk,
        target,
        chunk_len,
        target_len,
        max_mismatches,
        mismatch_counts,
    ):
        """
        Numba CUDA kernel for parallel DNA sequence matching.
        Each thread evaluates one candidate starting position in the genome chunk.

        Parameters:
            chunk: 1D device array of uint8 (encoded genome chunk).
            target: 1D device array of uint8 (encoded target sequence).
            chunk_len: Length of the chunk array.
            target_len: Length of the target array.
            max_mismatches: Maximum allowed mismatches (e.g. 0, 1, 2).
            mismatch_counts: 1D device array of uint8 of size (chunk_len - target_len + 1).
                             Stores mismatch count if <= max_mismatches, else NO_MATCH_SENTINEL (255).
        """
        pos = cuda.grid(1)
        num_candidates = chunk_len - target_len + 1

        if pos < num_candidates:
            mismatches = 0
            for i in range(target_len):
                if chunk[pos + i] != target[i]:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        break

            if mismatches <= max_mismatches:
                mismatch_counts[pos] = mismatches
            else:
                mismatch_counts[pos] = NO_MATCH_SENTINEL

else:
    # Placeholder for environments without numba cuda
    def dna_alignment_kernel(*args, **kwargs):
        raise RuntimeError("Numba CUDA is not available on this system.")


def calculate_launch_dimensions(
    num_candidates: int,
    threads_per_block: int = 128
) -> Tuple[int, int]:
    """
    Compute CUDA grid and block dimensions for a 1D kernel launch.

    Args:
        num_candidates: Total number of parallel candidate positions to process.
        threads_per_block: Number of threads per block (default: 128).

    Returns:
        tuple (blocks_per_grid, threads_per_block)
    """
    if num_candidates <= 0:
        return (0, threads_per_block)

    threads_per_block = max(1, min(threads_per_block, 1024))
    blocks_per_grid = (num_candidates + threads_per_block - 1) // threads_per_block
    return (blocks_per_grid, threads_per_block)
