import os
import sys
from dask.distributed import Client, LocalCluster
from numba import cuda


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
        "worker_addresses": list(workers.keys())
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
    print("Testing Dask cluster initialization...")
    client, cluster = get_dask_cluster(n_workers=2, threads_per_worker=1)
    status = get_cluster_status(client)
    print(f"Cluster Status: {status['status'].upper()}")
    print(f"Active Workers: {status['workers']}")
    print(f"Total Threads:  {status['threads']}")
    print(f"Total Memory:   {status['memory_mb']} MB")
    print(f"Workers:        {status['worker_addresses']}")

    assert status["workers"] == 2, f"Expected 2 workers, got {status['workers']}"
    close_dask_cluster(client, cluster)
    print("Dask cluster setup verified successfully!")
