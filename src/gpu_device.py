from numba import cuda


def get_gpu_device_info():
    """Queries and returns hardware specifications of the active CUDA GPU."""
    if not cuda.is_available():
        return None

    device = cuda.get_current_device()
    device_name = device.name.decode("utf-8") if isinstance(device.name, bytes) else str(device.name)
    compute_capability = device.compute_capability

    # Query memory from CUDA context
    free_mem, total_mem = cuda.current_context().get_memory_info()

    # Query multiprocessors if available
    multiprocessors = getattr(device, "MULTIPROCESSOR_COUNT", None)
    warp_size = getattr(device, "WARP_SIZE", 32)

    return {
        "name": device_name,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "total_memory_mb": round(total_mem / (1024 ** 2), 2),
        "free_memory_mb": round(free_mem / (1024 ** 2), 2),
        "multiprocessors": multiprocessors,
        "warp_size": warp_size,
        "device_id": device.id,
    }


def print_gpu_device_info():
    """Prints formatted details of the detected GPU."""
    print("========== GPU Device Detection ==========")
    if not cuda.is_available():
        print("No CUDA-compatible GPU detected.")
        return False

    info = get_gpu_device_info()
    print(f"Device ID: {info['device_id']}")
    print(f"Device Name: {info['name']}")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"Total VRAM: {info['total_memory_mb']} MB ({round(info['total_memory_mb'] / 1024, 2)} GB)")
    print(f"Free VRAM: {info['free_memory_mb']} MB")
    if info['multiprocessors']:
        print(f"Streaming Multiprocessors (SMs): {info['multiprocessors']}")
    print(f"Warp Size: {info['warp_size']}")
    print("==========================================")
    return True
