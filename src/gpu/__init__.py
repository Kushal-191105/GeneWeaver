"""
GeneWeaver Module 2: CUDA/Numba GPU Alignment Engine.
Provides high-performance, GPU-accelerated DNA sequence alignment, mismatch detection,
device management, and CPU fallback capabilities.
"""

from src.gpu.device import (
    is_cuda_available,
    get_device_info,
    get_device_count,
    print_device_info,
)
from src.gpu.encoding import (
    encode_sequence,
    decode_sequence,
    encode_target,
    validate_dna_sequence,
    BASE_TO_INT,
    INT_TO_BASE,
)
from src.gpu.kernels import (
    dna_alignment_kernel,
    calculate_launch_dimensions,
    NO_MATCH_SENTINEL,
)
from src.gpu.memory import (
    GPUMemoryBuffer,
)
from src.gpu.matcher import (
    GPUAlignmentEngine,
    align_sequence,
)
from src.gpu.benchmark import (
    run_alignment_benchmark,
    print_benchmark_summary,
    BenchmarkMetrics,
)

__all__ = [
    "is_cuda_available",
    "get_device_info",
    "get_device_count",
    "print_device_info",
    "encode_sequence",
    "decode_sequence",
    "encode_target",
    "validate_dna_sequence",
    "BASE_TO_INT",
    "INT_TO_BASE",
    "dna_alignment_kernel",
    "calculate_launch_dimensions",
    "NO_MATCH_SENTINEL",
    "GPUMemoryBuffer",
    "GPUAlignmentEngine",
    "align_sequence",
    "run_alignment_benchmark",
    "print_benchmark_summary",
    "BenchmarkMetrics",
]
