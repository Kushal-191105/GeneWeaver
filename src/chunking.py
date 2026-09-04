"""Genomic chunking.

A chromosome-scale FASTA record does not fit comfortably in GPU VRAM, so
the pipeline slices every record into fixed-size numpy arrays before the
alignment kernels ever see it.

Two rules make the chunking safe:

* Chunks never span two records. Concatenating separate sequences would
  invent matches that do not exist in the biology.
* Consecutive chunks of the same record overlap by ``target_length - 1``
  bases, so a match that straddles a chunk boundary is still fully
  contained in one chunk. Each chunk then only *owns* the first
  ``step`` positions of its window range, which keeps the overlap from
  reporting the same match twice.
"""

import numpy as np

DEFAULT_CHUNK_SIZE = 1_000_000


class Chunk:
    """One slice of one sequence, ready to be copied to the device.

    Attributes:
        sequence_id: id of the record this chunk came from.
        index: 0-based chunk number within that record.
        start: offset of the chunk inside the record.
        array: uint8 numpy array of the chunk's bases (ASCII codes).
        owned: how many alignment positions this chunk is responsible
            for reporting; positions at or beyond this belong to the
            next chunk and would otherwise be double-counted.
    """

    __slots__ = ("sequence_id", "index", "start", "array", "owned")

    def __init__(self, sequence_id, index, start, array, owned):
        self.sequence_id = sequence_id
        self.index = index
        self.start = start
        self.array = array
        self.owned = owned

    @property
    def size(self):
        return int(self.array.size)

    @property
    def text(self):
        """The chunk as a string, for the CPU baseline implementation."""
        return self.array.tobytes().decode("ascii")

    def __repr__(self):
        return (
            f"Chunk({self.sequence_id}, index={self.index}, "
            f"start={self.start}, size={self.size}, owned={self.owned})"
        )


def encode_bases(sequence):
    """Convert a DNA string to a uint8 array of ASCII codes.

    The array is copied so it is writeable: `numpy.frombuffer` returns a
    read-only view of the string's buffer, and read-only arrays are an
    avoidable edge case for CUDA host-to-device transfers.
    """
    return np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).copy()


def chunk_step(chunk_size=DEFAULT_CHUNK_SIZE, target_length=0):
    """How far the chunk window advances between chunks."""
    overlap = max(0, target_length - 1)

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be larger than the target overlap "
            f"({overlap}); got {chunk_size}"
        )

    return chunk_size - overlap


def count_chunks(length, chunk_size=DEFAULT_CHUNK_SIZE, target_length=0):
    """Number of chunks a sequence of `length` bases produces."""
    if length <= 0:
        return 0

    if length <= chunk_size:
        return 1

    step = chunk_step(chunk_size, target_length)

    return 1 + -(-(length - chunk_size) // step)


def count_dataset_chunks(dataset, chunk_size=DEFAULT_CHUNK_SIZE,
                         target_length=0):
    """Total chunks across every record of a loaded dataset."""
    return sum(
        count_chunks(int(length), chunk_size, target_length)
        for length in dataset["length"]
    )


def iter_sequence_chunks(sequence_id, sequence, chunk_size=DEFAULT_CHUNK_SIZE,
                         target_length=0):
    """Slice one sequence into overlapping chunks."""
    length = len(sequence)

    if length == 0:
        return

    array = encode_bases(sequence)

    if length <= chunk_size:
        yield Chunk(sequence_id, 0, 0, array, length)
        return

    step = chunk_step(chunk_size, target_length)

    index = 0
    start = 0

    while start < length:
        end = min(start + chunk_size, length)
        last = end >= length

        yield Chunk(
            sequence_id,
            index,
            start,
            array[start:end],
            (end - start) if last else step,
        )

        if last:
            return

        index += 1
        start += step


def iter_dataset_chunks(dataset, chunk_size=DEFAULT_CHUNK_SIZE,
                        target_length=0):
    """Slice every record of a loaded dataset into chunks."""
    records = dataset[["sequence_id", "sequence"]].itertuples(index=False)

    for sequence_id, sequence in records:
        yield from iter_sequence_chunks(
            sequence_id,
            sequence,
            chunk_size=chunk_size,
            target_length=target_length,
        )


def iter_fasta_chunks(filename, chunk_size=DEFAULT_CHUNK_SIZE,
                      target_length=0, limit=None):
    """Stream chunks straight from a FASTA file.

    Unlike `iter_dataset_chunks` this never holds the whole file in a
    DataFrame, so it is the path to use for chromosome-scale input.
    """
    from src.parser import iter_fasta_records

    for count, (record_id, sequence) in enumerate(
        iter_fasta_records(filename)
    ):
        if limit is not None and count >= limit:
            return

        yield from iter_sequence_chunks(
            record_id or f"sequence_{count}",
            sequence.strip().upper(),
            chunk_size=chunk_size,
            target_length=target_length,
        )


def chunk_summary(dataset, chunk_size=DEFAULT_CHUNK_SIZE, target_length=0):
    """Describe the chunk plan for a dataset, for logs and the dashboard."""
    total_bases = int(dataset["length"].sum())
    chunks = count_dataset_chunks(dataset, chunk_size, target_length)

    return {
        "sequences": len(dataset),
        "total_bases": total_bases,
        "chunk_size": chunk_size,
        "overlap": max(0, target_length - 1),
        "chunks": chunks,
        "bytes_per_chunk": min(chunk_size, total_bases),
    }
