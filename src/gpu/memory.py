"""
Safe GPU memory management and reusable device buffer utilities for GeneWeaver.
Minimizes host-device allocation overhead and prevents memory fragmentation.
"""

from typing import Optional
import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
    cuda = None


class GPUMemoryBuffer:
    """
    Reusable GPU device buffer to avoid redundant device allocations across multiple genome chunks.
    """

    def __init__(self, initial_chunk_capacity: int = 100_000):
        """
        Initialize GPU memory buffer.

        Args:
            initial_chunk_capacity: Initial capacity in bases for the chunk buffer.
        """
        self.chunk_capacity = max(1024, initial_chunk_capacity)
        self.d_chunk = None
        self.d_mismatch_counts = None
        self.d_target = None
        self._target_bytes: Optional[bytes] = None

    def allocate_target(self, encoded_target: np.ndarray):
        """
        Allocate and transfer the CRISPR target sequence to GPU device memory.
        Re-uses the existing device target buffer if target content has not changed.

        Args:
            encoded_target: 1D uint8 numpy array of encoded target.

        Returns:
            Device array holding the target.
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available on this system.")

        target_bytes = encoded_target.tobytes()
        if self.d_target is not None and self._target_bytes == target_bytes:
            return self.d_target

        self.d_target = cuda.to_device(encoded_target)
        self._target_bytes = target_bytes
        return self.d_target

    def prepare_chunk_buffers(self, chunk_len: int, target_len: int):
        """
        Ensure device buffers are allocated and large enough for the chunk and results.

        Args:
            chunk_len: Length of the genome chunk.
            target_len: Length of the target sequence.

        Returns:
            tuple (d_chunk_view, d_mismatch_counts_view)
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available on this system.")

        num_candidates = max(0, chunk_len - target_len + 1)

        # Allocate or expand chunk buffer if necessary
        if self.d_chunk is None or self.chunk_capacity < chunk_len:
            self.chunk_capacity = max(chunk_len, self.chunk_capacity * 2)
            try:
                self.d_chunk = cuda.device_array((self.chunk_capacity,), dtype=np.uint8)
                self.d_mismatch_counts = cuda.device_array((self.chunk_capacity,), dtype=np.uint8)
            except Exception as e:
                # Memory allocation error handling
                raise MemoryError(
                    f"Failed to allocate {self.chunk_capacity} bytes on GPU device: {e}"
                )

        return self.d_chunk, self.d_mismatch_counts

    def transfer_chunk_to_device(self, encoded_chunk: np.ndarray):
        """
        Copy encoded chunk data from host to device memory.

        Args:
            encoded_chunk: 1D uint8 host array.

        Returns:
            Device array slice for this chunk.
        """
        chunk_len = len(encoded_chunk)
        if self.d_chunk is None or self.chunk_capacity < chunk_len:
            self.d_chunk = cuda.to_device(encoded_chunk)
            return self.d_chunk
        
        # Copy to pre-allocated device buffer
        d_slice = self.d_chunk[:chunk_len]
        d_slice.copy_to_device(encoded_chunk)
        return d_slice

    def free(self):
        """
        Explicitly release device memory references.
        """
        self.d_chunk = None
        self.d_mismatch_counts = None
        self.d_target = None
        self._target_bytes = None
