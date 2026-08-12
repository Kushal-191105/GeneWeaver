"""
Tests for GPUAlignmentEngine configurations, backend fallbacks, and result formats.
"""

import pytest
from src.gpu.matcher import GPUAlignmentEngine, align_sequence
from src.gpu.device import is_cuda_available


def test_engine_initialization():
    engine = GPUAlignmentEngine(
        threads_per_block=256,
        chunk_size=50_000,
        backend="auto",
    )
    assert engine.threads_per_block == 256
    assert engine.chunk_size == 50_000
    assert engine.backend == "auto"


def test_engine_invalid_backend():
    with pytest.raises(ValueError, match="Invalid backend 'opencl'"):
        GPUAlignmentEngine(backend="opencl")


def test_engine_cpu_backend():
    engine = GPUAlignmentEngine(backend="cpu")
    assert engine.is_gpu_active() is False

    genome = "ATGCCCCAACTAAATACTAC"
    target = "ATGCCCCAACTAAATACTAC"
    results = engine.align(genome, target, max_mismatches=0)
    assert len(results) == 1
    res = results[0]
    assert res["position"] == 0
    assert res["sequence"] == target
    assert res["mismatches"] == 0
    assert res["mismatch_positions"] == []
    assert "chunk_id" in res


def test_engine_cuda_backend_unavailable():
    if not is_cuda_available():
        engine = GPUAlignmentEngine(backend="cuda")
        with pytest.raises(RuntimeError, match="CUDA backend requested, but"):
            engine.align("ACGTACGT", "ACGT")


def test_align_sequence_convenience_function():
    genome = "ACGTACGTACGT"
    target = "ACGT"
    results = align_sequence(genome, target, max_mismatches=0, backend="auto")
    assert len(results) == 3
    for r in results:
        assert r["sequence"] == "ACGT"
        assert r["mismatches"] == 0


def test_align_with_list_of_chunks():
    chunks = ["ACGTACGT", "TTTTACGT", "GGGGGGGG"]
    target = "ACGT"
    engine = GPUAlignmentEngine(backend="auto")
    results = engine.align(chunks, target, max_mismatches=0, chunk_size=8)
    assert len(results) == 3
    assert results[0]["position"] == 0
    assert results[1]["position"] == 4
    assert results[2]["position"] == 12  # chunk 1 offset (8) + 4
