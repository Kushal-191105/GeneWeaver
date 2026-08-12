"""
Tests for CUDA alignment kernel launch configuration and kernel execution.
"""

import pytest
import numpy as np
from src.gpu.kernels import (
    calculate_launch_dimensions,
    dna_alignment_kernel,
    NO_MATCH_SENTINEL,
    CUDA_AVAILABLE,
)
from src.gpu.encoding import encode_sequence, encode_target
from src.gpu.device import is_cuda_available


def test_calculate_launch_dimensions_zero_items():
    blocks, threads = calculate_launch_dimensions(0, threads_per_block=128)
    assert blocks == 0
    assert threads == 128


def test_calculate_launch_dimensions_negative_items():
    blocks, threads = calculate_launch_dimensions(-5, threads_per_block=128)
    assert blocks == 0
    assert threads == 128


def test_calculate_launch_dimensions_exact_multiple():
    blocks, threads = calculate_launch_dimensions(256, threads_per_block=128)
    assert blocks == 2
    assert threads == 128


def test_calculate_launch_dimensions_non_multiple():
    blocks, threads = calculate_launch_dimensions(257, threads_per_block=128)
    assert blocks == 3
    assert threads == 128


def test_calculate_launch_dimensions_clamped_threads():
    blocks, threads = calculate_launch_dimensions(100, threads_per_block=2048)
    assert threads == 1024
    assert blocks == 1


@pytest.mark.skipif(not CUDA_AVAILABLE or not is_cuda_available(), reason="Requires CUDA GPU or CUDA simulator")
def test_dna_alignment_kernel_execution():
    from numba import cuda

    chunk_str = "ACGTACGTACGT"
    target_str = "ACGT"
    max_mismatches = 0

    chunk_encoded = encode_sequence(chunk_str)
    target_encoded = encode_target(target_str)

    chunk_len = len(chunk_encoded)
    target_len = len(target_encoded)
    num_candidates = chunk_len - target_len + 1

    d_chunk = cuda.to_device(chunk_encoded)
    d_target = cuda.to_device(target_encoded)
    d_out = cuda.device_array(num_candidates, dtype=np.uint8)

    blocks, threads = calculate_launch_dimensions(num_candidates, 128)
    dna_alignment_kernel[blocks, threads](
        d_chunk, d_target, chunk_len, target_len, max_mismatches, d_out
    )

    h_out = d_out.copy_to_host()
    # Matches should be at indices 0, 4, 8 with 0 mismatches
    assert h_out[0] == 0
    assert h_out[4] == 0
    assert h_out[8] == 0
    # Other indices are not exact matches and should be NO_MATCH_SENTINEL (255)
    assert h_out[1] == NO_MATCH_SENTINEL
    assert h_out[2] == NO_MATCH_SENTINEL
    assert h_out[3] == NO_MATCH_SENTINEL
