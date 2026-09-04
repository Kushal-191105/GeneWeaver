import time

import numpy

from src.chunking import (
    DEFAULT_CHUNK_SIZE,
    count_dataset_chunks,
    encode_bases,
    iter_dataset_chunks,
)
from src.cpu_alignment import find_matches_with_mismatches
from src.gpu_alignment import (
    active_kernel,
    align_gpu,
    count_mismatches,
    gpu_backend_name,
)


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


def _hits_from_counts(chunk, target, counts, max_mismatches):
    """Turn a chunk's mismatch counts into match records.

    Only positions the chunk owns are reported, so the overlap between
    consecutive chunks never yields a duplicate match.
    """
    if counts.size == 0:
        return []

    positions = numpy.nonzero(counts <= max_mismatches)[0]
    positions = positions[positions < chunk.owned]

    if positions.size == 0:
        return []

    text = chunk.text
    target_length = len(target)

    return [
        {
            "sequence_id": chunk.sequence_id,
            "target": target,
            "position": chunk.start + int(position),
            "sequence": text[int(position):int(position) + target_length],
            "mismatches": int(counts[position]),
        }
        for position in positions
    ]


def _cpu_chunk_hits(chunk, target, max_mismatches):
    """Baseline path: pure-Python scan of one chunk."""
    hits = []

    for hit in find_matches_with_mismatches(
        chunk.text,
        target,
        max_mismatches=max_mismatches,
    ):
        if hit["position"] >= chunk.owned:
            continue

        hits.append({
            "sequence_id": chunk.sequence_id,
            "target": target,
            "position": chunk.start + hit["position"],
            "sequence": hit["sequence"],
            "mismatches": hit["mismatches"],
        })

    return hits


def sequence_index(dataset):
    """sequence_id -> full record text, so hits can be given their PAM.

    Scoring needs the bases sitting 3' of a match, which the match dict
    itself does not carry. The loaded dataset already holds every record,
    so the lookup is free.
    """
    records = dataset[["sequence_id", "sequence"]].itertuples(index=False)

    return {sequence_id: sequence for sequence_id, sequence in records}


def attach_scores(matches, dataset=None):
    """Annotate hits with the Week 3 biological score and severity tier."""
    from src.scoring import rank_matches, score_matches

    sequences = sequence_index(dataset) if dataset is not None else {}

    return rank_matches(score_matches(matches, sequences))


def run_chunked_alignment(dataset, target, mode="gpu", max_mismatches=2,
                          chunk_size=DEFAULT_CHUNK_SIZE, progress=None,
                          kernel="auto", stop_event=None):
    """Align `target` against a dataset one chunk at a time.

    Identical results to `run_alignment`, but the genome is streamed
    through fixed-size arrays instead of held whole, and `progress` is
    called after every chunk with a status dict the dashboard renders.

    Returns the same keys as `run_alignment` plus chunk counts.
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

    total_chunks = count_dataset_chunks(
        dataset,
        chunk_size=chunk_size,
        target_length=target_length,
    )

    matches = []
    positions_scanned = 0
    bases_done = 0
    chunks_done = 0

    start_time = time.perf_counter()

    for chunk in iter_dataset_chunks(
        dataset,
        chunk_size=chunk_size,
        target_length=target_length,
    ):
        if stop_event is not None and stop_event.is_set():
            break

        if chunk.size >= target_length:
            if mode == "cpu":
                hits = _cpu_chunk_hits(chunk, target, max_mismatches)
            else:
                counts = count_mismatches(
                    chunk.array, target_array, mode, kernel=kernel)
                hits = _hits_from_counts(chunk, target, counts, max_mismatches)

            matches.extend(hits)
            positions_scanned += min(
                chunk.owned,
                chunk.size - target_length + 1,
            )

        bases_done += chunk.size
        chunks_done += 1

        if progress is not None:
            progress({
                "target": target,
                "backend": backend,
                "chunk": chunk,
                "chunks_done": chunks_done,
                "chunks_total": total_chunks,
                "bases_done": bases_done,
                "matches": len(matches),
                "elapsed": time.perf_counter() - start_time,
            })

    elapsed = time.perf_counter() - start_time

    return {
        "mode": mode,
        "backend": backend,
        "sequences": len(dataset),
        "target": target,
        "target_length": target_length,
        "positions": positions_scanned,
        "elapsed": elapsed,
        "matches": matches,
        "chunks": chunks_done,
        "chunk_size": chunk_size,
        "bases": bases_done,
        "kernel": active_kernel(target_length, mode, kernel),
        "stopped": bool(stop_event is not None and stop_event.is_set()),
    }
