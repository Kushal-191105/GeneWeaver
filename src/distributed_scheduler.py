import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import dask
from dask.distributed import Client, LocalCluster
from numba import cuda
from src.gpu_alignment import gpu_find_matches_with_mismatches


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


def process_batch_alignment(batch: dict, target: str, max_mismatches: int = 2, worker_index: int = 0) -> list:
    """
    Core worker execution routine:
    1. Sets up GPU context for the worker.
    2. Runs high-throughput CUDA alignment on the batch slice.
    3. Remaps local batch coordinates to absolute global genome positions.
    """
    init_worker_cuda_context(worker_index)

    batch_seq = batch["sequence"]
    start_offset = batch["start_offset"]
    batch_id = batch["batch_id"]

    local_matches = gpu_find_matches_with_mismatches(batch_seq, target, max_mismatches=max_mismatches)
    global_matches = []

    for m in local_matches:
        global_pos = start_offset + m["position"]
        match_record = dict(m)
        match_record["position"] = global_pos
        match_record["batch_id"] = batch_id
        global_matches.append(match_record)

    return global_matches


@dask.delayed
def delayed_align_batch(batch: dict, target: str, max_mismatches: int = 2, worker_index: int = 0):
    """
    Dask delayed wrapper around batch alignment for asynchronous DAG construction.
    """
    return process_batch_alignment(batch, target, max_mismatches, worker_index)


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
    print("Testing process_batch_alignment and delayed_align_batch...")
    dummy_batch = {
        "batch_id": 1,
        "start_offset": 500,
        "end_offset": 540,
        "sequence": "GATC" * 10,
        "length": 40
    }
    target = "GATC"
    results = process_batch_alignment(dummy_batch, target, max_mismatches=0)
    print(f"Batch Alignment found {len(results)} matches:")
    for r in results[:3]:
        print(f"  Batch {r['batch_id']} | Global Pos: {r['position']} | Seq: {r['sequence']}")

    assert len(results) > 0
    assert results[0]["position"] >= 500
    print("Dask delayed alignment task verified successfully!")
