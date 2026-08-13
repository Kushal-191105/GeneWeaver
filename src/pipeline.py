import time

from src.cpu_alignment import find_matches_with_mismatches
from src.gpu_alignment import align_gpu, gpu_backend_name


def count_alignment_positions(sequences, target_length):
    """Total number of sliding windows across the whole dataset."""
    positions = 0

    for sequence in sequences:
        if len(sequence) >= target_length:
            positions += len(sequence) - target_length + 1

    return positions


def _align_one_cpu(sequence, target, max_mismatches):
    return find_matches_with_mismatches(
        sequence,
        target,
        max_mismatches=max_mismatches,
    )


def _align_one_gpu(sequence, target, max_mismatches):
    windows = align_gpu(sequence, target)

    return [
        window for window in windows
        if window["mismatches"] <= max_mismatches
    ]


def run_alignment(dataset, target, mode="cpu", max_mismatches=2):
    """Align `target` against every sequence in `dataset`.

    dataset: DataFrame with sequence_id / sequence columns.
    mode: "cpu" or "gpu".

    Returns a dict with sequences, target_length, positions, elapsed,
    matches and the backend that actually ran.
    """
    mode = mode.lower()

    if mode not in ("cpu", "gpu"):
        raise ValueError("mode must be 'cpu' or 'gpu'")

    target = target.strip().upper()
    target_length = len(target)

    if target_length == 0:
        raise ValueError("Target cannot be empty.")

    align_one = _align_one_cpu if mode == "cpu" else _align_one_gpu
    backend = "cpu" if mode == "cpu" else gpu_backend_name()

    records = dataset[["sequence_id", "sequence"]].itertuples(index=False)
    records = list(records)

    matches = []

    start_time = time.perf_counter()

    for sequence_id, sequence in records:

        if len(sequence) < target_length:
            continue

        for hit in align_one(sequence, target, max_mismatches):
            matches.append({
                "sequence_id": sequence_id,
                "target": target,
                "position": hit["position"],
                "sequence": hit["sequence"],
                "mismatches": hit["mismatches"],
            })

    elapsed = time.perf_counter() - start_time

    positions = count_alignment_positions(
        [sequence for _, sequence in records],
        target_length,
    )

    return {
        "mode": mode,
        "backend": backend,
        "sequences": len(records),
        "target": target,
        "target_length": target_length,
        "positions": positions,
        "elapsed": elapsed,
        "matches": matches,
    }
