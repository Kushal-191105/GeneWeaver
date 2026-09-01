import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import dask
from dask.distributed import Client, LocalCluster
from numba import cuda
from src.gpu_alignment import gpu_find_matches_with_mismatches
from src.scoring import rank_off_targets


def get_available_gpus() -> list:
    """
    Detects and returns all available CUDA GPU devices on the host.
    """
    try:
        if not cuda.is_available():
            return []
        return list(range(len(cuda.gpus)))
    except Exception:
        return [0]


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
    try:
        device_id = get_worker_gpu_device(worker_index)
        if device_id >= 0 and cuda.is_available():
            cuda.select_device(device_id)
            dev = cuda.get_current_device()
            name = dev.name.decode("utf-8") if isinstance(dev.name, bytes) else str(dev.name)
            return {
                "device_id": device_id,
                "device_name": name,
                "worker_index": worker_index,
                "status": "gpu_bound"
            }
    except Exception:
        pass

    return {
        "device_id": -1,
        "device_name": "Default GPU / CPU",
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
    2. Runs high-throughput CUDA Shared Memory (SRAM) alignment on the batch slice.
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


def dispatch_parallel_alignment(genome: str, target: str, max_mismatches: int = 2, n_batches: int = 4, client: Client = None) -> list:
    """
    Dispatches chunk batches across Dask distributed workers concurrently.
    Builds a computational task graph and executes in parallel.
    """
    batches = partition_genome_for_workers(genome, target_length=len(target), n_batches=n_batches)

    delayed_tasks = [
        delayed_align_batch(batch, target, max_mismatches=max_mismatches, worker_index=i)
        for i, batch in enumerate(batches)
    ]

    if client:
        futures = client.compute(delayed_tasks)
        batch_results = client.gather(futures)
    else:
        batch_results = dask.compute(*delayed_tasks, scheduler="synchronous")

    return list(batch_results)


def gather_and_deduplicate_results(batch_results: list) -> list:
    """
    Aggregates distributed batch outputs, removes boundary duplicates caused by chunk overlap,
    and sorts all matches in ascending order of genomic position.
    """
    seen_positions = set()
    unique_matches = []

    for batch in batch_results:
        for match in batch:
            pos = match["position"]
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_matches.append(match)

    unique_matches.sort(key=lambda x: x["position"])
    return unique_matches


def run_distributed_pipeline(genome: str, target: str, max_mismatches: int = 2, n_batches: int = 4, client: Client = None) -> list:
    """
    High-level distributed execution pipeline:
    1. Dispatches genomic batches across workers.
    2. Gathers and deduplicates hits across boundary overlaps.
    3. Enriches with biological PAM scoring and ranks by severity.
    """
    batch_outputs = dispatch_parallel_alignment(genome, target, max_mismatches=max_mismatches, n_batches=n_batches, client=client)
    merged_matches = gather_and_deduplicate_results(batch_outputs)
    ranked_off_targets = rank_off_targets(merged_matches, genome, target)
    return ranked_off_targets


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
    print("Testing distributed result gathering and deduplication...")
    test_seq = "ATGCCCCAACTAAATACTAC" + "TGG" + "ATGCCCCAACTAAATACTAC" + "CGG" + "GATC" * 50
    target_seq = "ATGCCCCAACTAAATACTAC"

    ranked = run_distributed_pipeline(test_seq, target_seq, max_mismatches=0, n_batches=4)
    print(f"Distributed Pipeline returned {len(ranked)} ranked matches:")
    for r in ranked:
        print(f"  Rank #{r['rank']} | Pos: {r['position']} | PAM: {r['pam']} | Score: {r['severity_score']}% | Risk: {r['risk_badge']}")

    assert len(ranked) == 2
    assert ranked[0]["position"] < ranked[1]["position"] or ranked[0]["severity_score"] >= ranked[1]["severity_score"]
    print("Distributed result gathering and ranking verified successfully!")
