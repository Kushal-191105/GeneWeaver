import os
import sys
import math
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
    """
    gpus = get_available_gpus()
    if not gpus:
        return -1
    return gpus[worker_index % len(gpus)]


def init_worker_cuda_context(worker_index: int = 0) -> dict:
    """
    Initializes the CUDA context on the designated GPU for a worker process.
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


def partition_genome_for_workers(genome: str, target_length: int = 20, n_batches: int = 4) -> list:
    """
    Partitions a genomic sequence into balanced batches for distributed Dask workers.
    Includes an overlap of (target_length + 3) bp to prevent boundary truncation of
    target protospacers and adjacent PAM motifs.
    """
    genome_len = len(genome)
    overlap = target_length + 3  # Target + PAM length

    if genome_len == 0:
        return []

    if n_batches <= 1 or genome_len <= overlap * 2:
        return [{
            "batch_id": 0,
            "start_offset": 0,
            "end_offset": genome_len,
            "sequence": genome,
            "length": genome_len
        }]

    base_batch_size = math.ceil(genome_len / n_batches)
    batches = []

    for i in range(n_batches):
        start = i * base_batch_size
        if start >= genome_len:
            break

        # Extend end boundary by overlap unless it is the final batch
        if i == n_batches - 1:
            end = genome_len
        else:
            end = min(start + base_batch_size + overlap, genome_len)

        batches.append({
            "batch_id": i,
            "start_offset": start,
            "end_offset": end,
            "sequence": genome[start:end],
            "length": end - start
        })

    return batches


def get_dask_cluster(n_workers: int = 2, threads_per_worker: int = 1, memory_limit: str = "2GB"):
    """
    Initializes and returns a Dask distributed LocalCluster and Client.
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
        dashboard_address=None
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
    print("Testing distributed chunk partitioner...")
    test_genome = "A" * 10000 + "C" * 10000 + "G" * 10000 + "T" * 10000
    target_l = 20
    batches = partition_genome_for_workers(test_genome, target_length=target_l, n_batches=4)
    print(f"Total genome: {len(test_genome)} bp partitioned into {len(batches)} batches:")

    for b in batches:
        print(f"  Batch {b['batch_id']}: offsets [{b['start_offset']:,} -> {b['end_offset']:,}], len={b['length']:,} bp")

    assert len(batches) == 4
    # Check that overlap covers boundaries
    assert batches[0]["end_offset"] > batches[1]["start_offset"]
    print("Distributed chunk partitioner verified successfully!")
