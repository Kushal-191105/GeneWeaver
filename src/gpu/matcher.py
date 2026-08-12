"""
GPU Alignment Engine for GeneWeaver.
Orchestrates parallel CUDA DNA sequence alignment, mismatch detection, chunk-based processing,
and automatic fallback to CPU alignment when CUDA is unavailable.
"""

from typing import List, Dict, Any, Union, Optional
import math
import numpy as np

from src.gpu.device import is_cuda_available
from src.gpu.encoding import encode_sequence, encode_target
from src.gpu.kernels import (
    dna_alignment_kernel,
    calculate_launch_dimensions,
    NO_MATCH_SENTINEL,
    CUDA_AVAILABLE
)
from src.gpu.memory import GPUMemoryBuffer
from src.parser import create_chunks
from src.cpu_alignment import find_matches_with_mismatches as cpu_find_matches


class GPUAlignmentEngine:
    """
    High-performance GPU-accelerated DNA sequence alignment and mismatch detection engine.
    """

    def __init__(
        self,
        threads_per_block: int = 128,
        chunk_size: int = 100_000,
        backend: str = "auto",
        device_id: int = 0
    ):
        """
        Initialize the GPU alignment engine.

        Args:
            threads_per_block: Number of CUDA threads per block (default: 128).
            chunk_size: Default chunk size in bases for genome chunking (default: 100,000).
            backend: Execution backend choice ('auto', 'cuda', 'cpu').
            device_id: CUDA device ID to use (default: 0).
        """
        self.threads_per_block = max(32, min(threads_per_block, 1024))
        self.chunk_size = max(10, chunk_size)
        self.backend = backend.lower().strip()
        self.device_id = device_id
        self._memory_buffer: Optional[GPUMemoryBuffer] = None

        if self.backend not in ("auto", "cuda", "cpu"):
            raise ValueError(f"Invalid backend '{backend}'. Choose from 'auto', 'cuda', 'cpu'.")

    def _get_memory_buffer(self) -> GPUMemoryBuffer:
        if self._memory_buffer is None:
            self._memory_buffer = GPUMemoryBuffer(initial_chunk_capacity=self.chunk_size)
        return self._memory_buffer

    def is_gpu_active(self) -> bool:
        """
        Determine if CUDA GPU backend is available and active for execution.
        """
        if self.backend == "cpu":
            return False
        if not is_cuda_available():
            return False
        return True

    def align_chunk_cuda(
        self,
        chunk: str,
        target: str,
        max_mismatches: int = 2,
        global_offset: int = 0,
        chunk_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Align a single DNA chunk against the target sequence using Numba CUDA kernel.

        Args:
            chunk: DNA chunk string.
            target: CRISPR target sequence string.
            max_mismatches: Maximum allowed mismatches (default: 2).
            global_offset: Global genomic offset for this chunk (default: 0).
            chunk_id: Identifier index of this chunk (default: 0).

        Returns:
            List of alignment result dictionaries matching the CPU matcher format.
        """
        if not CUDA_AVAILABLE or not is_cuda_available():
            raise RuntimeError("CUDA backend requested, but no CUDA-capable device or driver is available.")

        chunk_str = chunk.upper().strip()
        target_str = target.upper().strip()

        chunk_len = len(chunk_str)
        target_len = len(target_str)

        # If chunk is shorter than target, no match is possible
        if chunk_len < target_len or target_len == 0:
            return []

        num_candidates = chunk_len - target_len + 1

        # Encode strings to uint8 arrays
        encoded_chunk = encode_sequence(chunk_str, validate=True)
        encoded_target = encode_target(target_str)

        # Allocate / transfer to GPU device
        from numba import cuda
        d_chunk = cuda.to_device(encoded_chunk)
        d_target = cuda.to_device(encoded_target)
        d_mismatch_counts = cuda.device_array(num_candidates, dtype=np.uint8)

        # Calculate launch configuration
        blocks_per_grid, threads = calculate_launch_dimensions(
            num_candidates,
            self.threads_per_block
        )

        # Launch CUDA Kernel
        dna_alignment_kernel[blocks_per_grid, threads](
            d_chunk,
            d_target,
            chunk_len,
            target_len,
            max_mismatches,
            d_mismatch_counts,
        )

        # Transfer results back to host
        h_mismatch_counts = d_mismatch_counts.copy_to_host()

        # Find matching candidate positions
        matching_indices = np.where(h_mismatch_counts <= max_mismatches)[0]

        results = []
        for local_pos in matching_indices:
            local_pos_int = int(local_pos)
            mismatches = int(h_mismatch_counts[local_pos_int])
            matched_seq = chunk_str[local_pos_int: local_pos_int + target_len]

            # Compute detailed mismatch positions relative to target
            mismatch_positions = [
                i for i in range(target_len) if matched_seq[i] != target_str[i]
            ]

            results.append({
                "position": global_offset + local_pos_int,
                "sequence": matched_seq,
                "mismatches": mismatches,
                "mismatch_positions": mismatch_positions,
                "chunk_id": chunk_id,
            })

        return results

    def align_chunks(
        self,
        chunks: List[str],
        target: str,
        max_mismatches: int = 2,
        chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple genome chunks sequentially through the alignment engine,
        preserving global genomic positions.

        Args:
            chunks: List of genome chunk strings.
            target: CRISPR target sequence string.
            max_mismatches: Maximum allowed mismatches.
            chunk_size: Explicit chunk size used when computing global offsets.
                        If None, computes cumulative lengths of preceding chunks.

        Returns:
            List of alignment result dictionaries.
        """
        all_results = []
        current_offset = 0

        use_cuda = self.is_gpu_active()
        if self.backend == "cuda" and not use_cuda:
            raise RuntimeError(
                "CUDA backend requested, but no CUDA-capable device or driver is available."
            )

        for chunk_id, chunk in enumerate(chunks):
            if not chunk:
                continue

            if use_cuda:
                chunk_results = self.align_chunk_cuda(
                    chunk=chunk,
                    target=target,
                    max_mismatches=max_mismatches,
                    global_offset=current_offset,
                    chunk_id=chunk_id,
                )
            else:
                # CPU fallback reusing existing CPU matcher
                raw_cpu_matches = cpu_find_matches(
                    genome=chunk,
                    target=target,
                    max_mismatches=max_mismatches,
                )
                chunk_results = []
                for match in raw_cpu_matches:
                    chunk_results.append({
                        "position": current_offset + match["position"],
                        "sequence": match["sequence"],
                        "mismatches": match["mismatches"],
                        "mismatch_positions": match["mismatch_positions"],
                        "chunk_id": chunk_id,
                    })

            all_results.extend(chunk_results)

            if chunk_size is not None:
                current_offset += chunk_size
            else:
                current_offset += len(chunk)

        return all_results

    def align(
        self,
        genome_or_chunks: Union[str, List[str]],
        target: str,
        max_mismatches: int = 2,
        chunk_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified alignment entry point. Accepts either a full genome string or a list of chunks.

        Args:
            genome_or_chunks: Single genome DNA string or pre-split list of DNA chunks.
            target: Target CRISPR sequence.
            max_mismatches: Maximum allowed mismatches.
            chunk_size: Chunk size in bases (defaults to self.chunk_size).

        Returns:
            List of alignment result dictionaries.
        """
        effective_chunk_size = chunk_size or self.chunk_size

        if isinstance(genome_or_chunks, list):
            return self.align_chunks(
                chunks=genome_or_chunks,
                target=target,
                max_mismatches=max_mismatches,
                chunk_size=chunk_size,
            )

        genome = str(genome_or_chunks)

        # If genome is larger than chunk size, use existing create_chunks
        if len(genome) > effective_chunk_size:
            chunks = create_chunks(genome, effective_chunk_size)
            return self.align_chunks(
                chunks=chunks,
                target=target,
                max_mismatches=max_mismatches,
                chunk_size=effective_chunk_size,
            )

        # Single chunk processing
        if self.is_gpu_active():
            return self.align_chunk_cuda(
                chunk=genome,
                target=target,
                max_mismatches=max_mismatches,
                global_offset=0,
                chunk_id=0,
            )

        if self.backend == "cuda":
            raise RuntimeError("CUDA backend requested, but no CUDA-capable device or driver is available.")

        # CPU fallback
        raw_cpu = cpu_find_matches(genome, target, max_mismatches=max_mismatches)
        return [
            {
                "position": m["position"],
                "sequence": m["sequence"],
                "mismatches": m["mismatches"],
                "mismatch_positions": m["mismatch_positions"],
                "chunk_id": 0,
            }
            for m in raw_cpu
        ]


def align_sequence(
    genome: str,
    target: str,
    max_mismatches: int = 2,
    chunk_size: int = 100_000,
    backend: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Convenience function to perform sequence alignment using the GPU alignment engine.

    Args:
        genome: DNA genome string.
        target: CRISPR target sequence string.
        max_mismatches: Maximum allowed mismatches (default: 2).
        chunk_size: Chunk size in bases (default: 100,000).
        backend: Execution backend ('auto', 'cuda', 'cpu').

    Returns:
        List of match dictionaries.
    """
    engine = GPUAlignmentEngine(
        chunk_size=chunk_size,
        backend=backend,
    )
    return engine.align(genome, target, max_mismatches=max_mismatches)
