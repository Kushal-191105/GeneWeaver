import os
import sys
from dask.distributed import Client, LocalCluster
from numba import cuda


def get_available_gpus() -> list:
    """
    Detects and returns all available CUDA GPU devices on the host.
    """
    if not cuda.is_available():
        return []
    return list(range(len(cuda.gpus)))


def get_worker_gpu_device(worker_index: int = 0) -> int:
    """
    Maps a Dask worker index to a target CUDA GPU device ID using round-robin affinity.
    If the system has 2 GPUs (GPU 0 and GPU 1), workers are balanced evenly:
      Worker 0 -> GPU 0
      Worker 1 -> GPU 1
      Worker 2 -> GPU 0
      ...
    """
    gpus = get_available_gpus()
    if not gpus:
        return -1
    return gpus[worker_index % len(gpus)]


def init_worker_cuda_context(worker_index: int = 0) -> dict:
    """
    Initializes the CUDA context on the designated GPU for a worker process.
    Returns the active device info.
    """
    device_id = get_worker_gpu_device(worker_index)
    if device_id >= 0:
        cuda.select_device(device_id)
        dev = cuda.get_current_device()
        name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
        return {
            "device_id": device_id,
            "device_name": name,
            "worker_index": worker_index,
            "status": "gpu_bound"
        }
    return {
        "device_id": -1,
        "device_name": "CPU Fallback",
        "worker_index": worker_index,
        "status": "cpu_bound"
    }


def get_dask_cluster(n_workers: int = 2, threads_per_worker: int = 1, memory_limit: str = "2GB"):
    """
    Initializes and returns a Dask distributed LocalCluster and Client.
    Coordinates distributed alignment tasks across worker processes.
    """
    try:
        client = Client.current()
        return client, None
    except ValueError:
        pass

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        processes=True,
        dashboard_address=None  # Headless mode for terminal stability
    )
    client = Client(cluster)
    return client, cluster


def get_cluster_status(client: Client) -> dict:
    """
    Queries real-time telemetry from the active Dask cluster.
    """
    if client is None:
        return {"status": "inactive", "workers": 0, "threads": 0, "memory": "0 MB"}

    info = client.scheduler_info()
    workers = info.get("workers", {})
    worker_count = len(workers)
    total_threads = sum(w.get("nthreads", 1) for w in workers.values())
    total_memory_bytes = sum(w.get("memory_limit", 0) for w in workers.values())
    total_memory_mb = round(total_memory_bytes / (1024 ** 2), 2)

    return {
        "status": "active",
        "workers": worker_count,
        "threads": total_threads,
        "memory_mb": total_memory_mb,
        "scheduler_address": info.get("address", "N/A"),
        "worker_addresses": list(workers.keys()),
        "available_gpus": len(get_available_gpus())
    }


def close_dask_cluster(client: Client, cluster: LocalCluster = None):
    """
    Safely tears down the Dask distributed client and cluster.
    """
    if client:
        try:
            client.close()
        except Exception:
            pass
    if cluster:
        try:
            cluster.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("Testing GPU worker device affinity...")
    gpus = get_available_gpus()
    print(f"Available GPUs: {len(gpus)} (IDs: {gpus})")

    for w_idx in range(4):
        target_gpu = get_worker_gpu_device(w_idx)
        print(f"Worker {w_idx} assigned to GPU Device: {target_gpu}")

    active_ctx = init_worker_cuda_context(0)
    print(f"Initialized Context: {active_ctx}")
    assert active_ctx["status"] == "gpu_bound" or len(gpus) == 0
    print("GPU worker device affinity verified successfully!")
