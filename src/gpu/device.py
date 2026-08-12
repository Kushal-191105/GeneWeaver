"""
GPU device detection and capability inspection component for GeneWeaver.
Provides safe, graceful CUDA detection and device querying without crashing on systems lacking an NVIDIA GPU.
"""

from typing import Dict, Any, Optional
import os


def is_cuda_available() -> bool:
    """
    Check if CUDA is available via Numba without raising unhandled exceptions.

    Returns:
        bool: True if CUDA driver, runtime, and at least one CUDA device are available, False otherwise.
    """
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        int: Number of detected CUDA devices, or 0 if CUDA is unavailable.
    """
    if not is_cuda_available():
        return 0
    try:
        from numba import cuda
        return len(cuda.gpus)
    except Exception:
        return 0


def get_device_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Query information and memory statistics for a specific CUDA device.

    Args:
        device_id: Index of the CUDA device (default 0).

    Returns:
        dict: Device properties including name, compute capability, memory, multiprocessors, etc.
    """
    info: Dict[str, Any] = {
        "available": False,
        "device_id": None,
        "name": None,
        "compute_capability": None,
        "total_memory": None,
        "total_memory_mb": None,
        "free_memory": None,
        "free_memory_mb": None,
        "multiprocessor_count": None,
        "max_threads_per_block": None,
        "warp_size": None,
    }

    if not is_cuda_available():
        return info

    try:
        from numba import cuda
        gpus = cuda.gpus
        if device_id < 0 or device_id >= len(gpus):
            return info

        device = gpus[device_id]
        with device:
            # Device name
            name = getattr(device, "name", None)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            elif name is None:
                name = f"CUDA Device {device_id}"

            compute_capability = getattr(device, "compute_capability", None)
            multiprocessors = getattr(device, "MULTIPROCESSOR_COUNT", None)
            max_threads_per_block = getattr(device, "MAX_THREADS_PER_BLOCK", 1024)
            warp_size = getattr(device, "WARP_SIZE", 32)

            total_mem: Optional[int] = None
            free_mem: Optional[int] = None

            try:
                current_ctx = cuda.current_context()
                mem_info = current_ctx.get_memory_info()
                free_mem = mem_info.free
                total_mem = mem_info.total
            except Exception:
                pass

            info.update({
                "available": True,
                "device_id": device_id,
                "name": str(name),
                "compute_capability": compute_capability,
                "total_memory": total_mem,
                "total_memory_mb": round(total_mem / (1024 * 1024), 2) if total_mem else None,
                "free_memory": free_mem,
                "free_memory_mb": round(free_mem / (1024 * 1024), 2) if free_mem else None,
                "multiprocessor_count": multiprocessors,
                "max_threads_per_block": max_threads_per_block,
                "warp_size": warp_size,
            })

    except Exception:
        pass

    return info


def print_device_info(device_id: int = 0) -> None:
    """
    Print formatted CUDA device properties to standard output.
    """
    info = get_device_info(device_id)
    print("\n========== GPU Device Information ==========")
    if not info["available"]:
        print("CUDA Status: Unavailable (No compatible NVIDIA GPU / CUDA driver detected)")
        print("Backend Fallback: CPU alignment mode active")
        return

    print(f"CUDA Status: Available")
    print(f"Device ID: {info['device_id']}")
    print(f"GPU Name: {info['name']}")
    if info["compute_capability"]:
        print(f"Compute Capability: {info['compute_capability'][0]}.{info['compute_capability'][1]}")
    if info["total_memory_mb"] is not None:
        print(f"Total GPU Memory: {info['total_memory_mb']} MB")
    if info["free_memory_mb"] is not None:
        print(f"Free GPU Memory: {info['free_memory_mb']} MB")
    if info["multiprocessor_count"] is not None:
        print(f"Multiprocessors (SMs): {info['multiprocessor_count']}")
    if info["max_threads_per_block"] is not None:
        print(f"Max Threads Per Block: {info['max_threads_per_block']}")
    if info["warp_size"] is not None:
        print(f"Warp Size: {info['warp_size']}")
