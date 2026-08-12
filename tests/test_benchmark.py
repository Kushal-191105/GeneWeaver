"""
Tests for GPU and CPU alignment benchmark execution and metrics reporting.
"""

import pytest
from src.gpu.benchmark import (
    run_alignment_benchmark,
    print_benchmark_summary,
    BenchmarkMetrics,
)


def test_run_alignment_benchmark_metrics():
    genome = "ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCC"
    target = "ATGCCCCAACTAAATACTAC"

    metrics = run_alignment_benchmark(
        genome=genome,
        target=target,
        max_mismatches=2,
        chunk_size=20,
    )

    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.input_size == len(genome)
    assert metrics.target_length == len(target)
    assert metrics.max_mismatches == 2
    assert metrics.cpu_matches_count >= 1
    assert metrics.cpu_time_seconds > 0


def test_print_benchmark_summary_output(capsys):
    metrics = BenchmarkMetrics(
        input_size=1000,
        target_length=20,
        max_mismatches=2,
        cpu_matches_count=3,
        gpu_matches_count=3,
        cpu_time_seconds=0.015,
        gpu_total_time_seconds=0.002,
        gpu_warmup_time_seconds=0.05,
        gpu_kernel_time_seconds=0.0005,
        gpu_transfer_time_seconds=0.0002,
        speedup=7.5,
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 4090",
        correctness_verified=True,
    )

    print_benchmark_summary(metrics)
    out = capsys.readouterr().out
    assert "GENEWEAVER ALIGNMENT BENCHMARK" in out
    assert "7.50x" in out
    assert "NVIDIA GeForce RTX 4090" in out
