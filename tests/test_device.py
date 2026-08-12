"""
Tests for GPU device detection and inspection.
"""

import pytest
from src.gpu.device import (
    is_cuda_available,
    get_device_count,
    get_device_info,
    print_device_info,
)


def test_is_cuda_available_returns_bool():
    result = is_cuda_available()
    assert isinstance(result, bool)


def test_get_device_count_returns_non_negative_integer():
    count = get_device_count()
    assert isinstance(count, int)
    assert count >= 0


def test_get_device_info_structure():
    info = get_device_info(0)
    assert isinstance(info, dict)
    expected_keys = {
        "available",
        "device_id",
        "name",
        "compute_capability",
        "total_memory",
        "total_memory_mb",
        "free_memory",
        "free_memory_mb",
        "multiprocessor_count",
        "max_threads_per_block",
        "warp_size",
    }
    assert expected_keys.issubset(info.keys())


def test_get_device_info_invalid_device_id():
    info = get_device_info(9999)
    assert isinstance(info, dict)
    assert info["available"] is False


def test_print_device_info_does_not_crash(capsys):
    print_device_info(0)
    captured = capsys.readouterr()
    assert "GPU Device Information" in captured.out
