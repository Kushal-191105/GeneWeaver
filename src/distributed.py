"""Distributed scaling across devices.

Week 3 - Distributed Scaling.

Week 2 streamed chunks through one device, one at a time. That leaves a
second GPU idle, and on a chromosome-scale run it leaves it idle for a
long time. This module puts Dask in front of the chunk stream so the
work is scheduled instead of merely iterated.

The scheduling problem is small but real. Chunks are *not* uniform: the
last chunk of every FASTA record is a remainder and can be a few hundred
bases against a neighbour's million. Handing chunks out round-robin
therefore does not balance two GPUs - it balances chunk *counts* while
leaving the base counts lopsided. So assignment is greedy
longest-processing-time-first: each chunk goes to whichever device
currently owns the fewest bases. That is the classic LPT heuristic and
it keeps two identical GPUs inside a couple of percent of each other.

Every layer degrades instead of failing:

* No Dask installed -> a ThreadPoolExecutor runs the same partitions.
* No CUDA -> the numpy backend runs them, still in parallel, still
  balanced, so ``--distributed`` is testable on a laptop.
* One GPU -> one partition, identical results to the Week 2 path.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.chunking import (
    DEFAULT_CHUNK_SIZE,
    count_dataset_chunks,
    encode_bases,
    iter_dataset_chunks,
)
from src.gpu_alignment import (
    active_kernel,
    count_mismatches,
    cuda_available,
    gpu_backend_name,
)
from src.pipeline import _cpu_chunk_hits, _hits_from_counts


# ----------------------------------------------------------------------
# device discovery
# ----------------------------------------------------------------------

def list_cuda_devices():
    """Every visible CUDA device as (id, name, total VRAM bytes)."""
    if not cuda_available():
        return []

    try:
        from numba import cuda

        devices = []

        for device in cuda.list_devices():
            name = device.name

            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")

            devices.append({
                "id": int(device.id),
                "name": name,
                "kind": "cuda",
            })

        return devices
    except Exception:
        return []


def available_devices(mode="gpu", limit=None):
    """Worker devices for a run.

    On a CUDA machine this is one entry per GPU, so a 2-GPU box gets two
    partitions. Everywhere else it is a single numpy worker, which keeps
    the distributed path exercisable off a GPU box.
    """
    devices = list_cuda_devices() if mode == "gpu" else []

    if not devices:
        devices = [{
            "id": 0,
            "name": gpu_backend_name() if mode == "gpu" else "cpu",
            "kind": "numpy" if mode == "gpu" else "cpu",
        }]

    if limit and limit > 0:
        devices = devices[:limit]

    return devices


def select_device(device):
    """Pin the calling thread to a CUDA device, when there is one."""
    if device.get("kind") != "cuda":
        return False

    try:
        from numba import cuda

        cuda.select_device(int(device["id"]))

        return True
    except Exception:
        return False


def device_memory(device):
    """Free/total VRAM for a CUDA device, or None."""
    if device.get("kind") != "cuda":
        return None

    try:
        from numba import cuda

        cuda.select_device(int(device["id"]))
        free, total = cuda.current_context().get_memory_info()

        return {"free": int(free), "total": int(total), "used": int(total - free)}
    except Exception:
        return None


# ----------------------------------------------------------------------
# balancing
# ----------------------------------------------------------------------

def balance_chunks(chunks, device_count):
    """Split chunks across devices so each gets a similar number of bases.

    Greedy LPT: walk the chunks and give each to the least-loaded
    device. Chunks are kept in file order within a partition so results
    stay reproducible.

    Returns a list of `device_count` lists of chunks.
    """
    if device_count <= 1:
        return [list(chunks)]

    partitions = [[] for _ in range(device_count)]
    load = [0] * device_count

    for chunk in chunks:
        target = min(range(device_count), key=lambda index: load[index])

        partitions[target].append(chunk)
        load[target] += chunk.size

    return partitions


def balance_report(partitions):
    """How even the split actually came out, in bases."""
    loads = [sum(chunk.size for chunk in partition) for partition in partitions]
    total = sum(loads)

    if not loads or total == 0:
        return {"loads": loads, "total": 0, "imbalance": 0.0, "shares": []}

    mean = total / len(loads)
    imbalance = (max(loads) - min(loads)) / mean if mean else 0.0

    return {
        "loads": loads,
        "total": total,
        "imbalance": round(imbalance, 6),
        "shares": [round(100.0 * load / total, 2) for load in loads],
    }


# ----------------------------------------------------------------------
# the worker
# ----------------------------------------------------------------------

def _run_partition(chunks, target, target_array, device, mode,
                   max_mismatches, kernel="auto", on_chunk=None):
    """Align one device's share of the chunks.

    Runs inside a Dask task or a worker thread, so it must not touch
    shared state - it returns everything it learned.
    """
    pinned = select_device(device)

    matches = []
    positions = 0
    bases = 0
    processed = 0
    target_length = len(target)

    started = time.perf_counter()

    for chunk in chunks:
        if chunk.size >= target_length:
            if mode == "cpu":
                hits = _cpu_chunk_hits(chunk, target, max_mismatches)
            else:
                counts = count_mismatches(
                    chunk.array, target_array, mode, kernel=kernel)
                hits = _hits_from_counts(chunk, target, counts, max_mismatches)

            matches.extend(hits)
            positions += min(chunk.owned, chunk.size - target_length + 1)

        bases += chunk.size
        processed += 1

        if on_chunk is not None:
            on_chunk(chunk.size, len(matches))

    return {
        "device": device,
        "pinned": pinned,
        "matches": matches,
        "positions": positions,
        "bases": bases,
        "chunks": processed,
        "seconds": time.perf_counter() - started,
    }


# ----------------------------------------------------------------------
# schedulers
# ----------------------------------------------------------------------

def dask_available():
    try:
        import dask  # noqa: F401

        return True
    except Exception:
        return False


def scheduler_name(requested="auto"):
    """Which scheduler a run will actually use."""
    if requested in ("threads", "sync"):
        return requested

    return "dask" if dask_available() else "threads"


def _run_with_dask(partitions, target, target_array, devices, mode,
                   max_mismatches, kernel, on_chunk):
    """Schedule one Dask task per device and compute them together."""
    import dask

    tasks = [
        dask.delayed(_run_partition)(
            partition, target, target_array, device, mode,
            max_mismatches, kernel, on_chunk,
        )
        for partition, device in zip(partitions, devices)
    ]

    # The threaded scheduler is the right one here: the heavy work is
    # numpy and CUDA calls, both of which release the GIL, and threads
    # keep the chunk arrays in shared memory instead of pickling them to
    # worker processes.
    return list(dask.compute(*tasks, scheduler="threads"))


def _run_with_threads(partitions, target, target_array, devices, mode,
                      max_mismatches, kernel, on_chunk):
    """Dask-free fallback with the same partitioning and the same results."""
    if len(partitions) == 1:
        return [_run_partition(
            partitions[0], target, target_array, devices[0], mode,
            max_mismatches, kernel, on_chunk,
        )]

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        futures = [
            pool.submit(
                _run_partition, partition, target, target_array, device,
                mode, max_mismatches, kernel, on_chunk,
            )
            for partition, device in zip(partitions, devices)
        ]

        return [future.result() for future in futures]


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------

def run_distributed_alignment(dataset, target, mode="gpu", max_mismatches=2,
                              chunk_size=DEFAULT_CHUNK_SIZE, progress=None,
                              devices=None, scheduler="auto", kernel="auto",
                              stop_event=None):
    """Align `target` across every available device.

    Same contract as `run_chunked_alignment` - identical matches, same
    return keys - plus per-device accounting and a balance report.
    """
    mode = mode.lower()

    if mode not in ("cpu", "gpu"):
        raise ValueError("mode must be 'cpu' or 'gpu'")

    target = target.strip().upper()
    target_length = len(target)

    if target_length == 0:
        raise ValueError("Target cannot be empty.")

    target_array = encode_bases(target)
    backend = "cpu" if mode == "cpu" else gpu_backend_name()

    if devices is None:
        devices = available_devices(mode)

    total_chunks = count_dataset_chunks(
        dataset, chunk_size=chunk_size, target_length=target_length)

    chunks = list(iter_dataset_chunks(
        dataset, chunk_size=chunk_size, target_length=target_length))

    partitions = balance_chunks(chunks, len(devices))
    balance = balance_report(partitions)

    used = scheduler_name(scheduler)

    # Progress arrives from several workers at once, so it is aggregated
    # behind a lock rather than reported per partition.
    import threading

    state = {"chunks": 0, "bases": 0, "matches": 0}
    lock = threading.Lock()
    started = time.perf_counter()

    def on_chunk(size, partition_matches):
        if stop_event is not None and stop_event.is_set():
            raise _Stopped()

        if progress is None:
            return

        with lock:
            state["chunks"] += 1
            state["bases"] += size
            snapshot = dict(state)

        progress({
            "target": target,
            "backend": backend,
            "chunk": None,
            "chunks_done": snapshot["chunks"],
            "chunks_total": total_chunks,
            "bases_done": snapshot["bases"],
            "matches": snapshot["matches"],
            "elapsed": time.perf_counter() - started,
            "devices": len(devices),
            "scheduler": used,
        })

    runner = _run_with_dask if used == "dask" else _run_with_threads

    try:
        results = runner(
            partitions, target, target_array, devices, mode,
            max_mismatches, kernel, on_chunk,
        )
    except _Stopped:
        results = []

    elapsed = time.perf_counter() - started

    matches = []
    positions = 0
    bases = 0
    processed = 0
    device_rows = []

    for index, result in enumerate(results):
        matches.extend(result["matches"])
        positions += result["positions"]
        bases += result["bases"]
        processed += result["chunks"]

        device_rows.append({
            "id": result["device"]["id"],
            "name": result["device"]["name"],
            "kind": result["device"]["kind"],
            "pinned": result["pinned"],
            "chunks": result["chunks"],
            "bases": result["bases"],
            "matches": len(result["matches"]),
            "seconds": round(result["seconds"], 6),
            "share": balance["shares"][index] if index < len(balance["shares"]) else 0.0,
            "throughput": (
                result["bases"] / result["seconds"] if result["seconds"] else 0.0
            ),
        })

    # File order, so a distributed run and a serial run produce byte-identical CSVs.
    matches.sort(key=lambda match: (str(match["sequence_id"]), int(match["position"])))

    return {
        "mode": mode,
        "backend": backend,
        "scheduler": used,
        "sequences": len(dataset),
        "target": target,
        "target_length": target_length,
        "positions": positions,
        "elapsed": elapsed,
        "matches": matches,
        "chunks": processed,
        "chunk_size": chunk_size,
        "bases": bases,
        "kernel": active_kernel(target_length, mode, kernel),
        "devices": device_rows,
        "balance": balance,
        "stopped": bool(stop_event is not None and stop_event.is_set()),
    }


class _Stopped(Exception):
    """Raised inside a worker when the caller asked the run to stop."""


def cluster_summary(mode="gpu", scheduler="auto"):
    """What the UIs show before a run starts."""
    devices = available_devices(mode)

    return {
        "scheduler": scheduler_name(scheduler),
        "dask": dask_available(),
        "device_count": len(devices),
        "devices": [
            dict(device, memory=device_memory(device)) for device in devices
        ],
    }
