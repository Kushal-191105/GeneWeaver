"""Input parsing for GeneWeaver.

FASTA is the primary dataset format (`data/genome.fasta`); CSV/TSV tables
such as `data/human_sequences.csv` are supported as an alternative input.
Both are loaded into the same DataFrame shape, so the rest of the pipeline
does not care which one was used.
"""

import gzip
import os

import pandas as pd

VALID_BASES = set("ATGCN")

FASTA_EXTENSIONS = (
    ".fasta",
    ".fa",
    ".fas",
    ".fna",
    ".ffn",
    ".faa",
    ".frn",
)

COMPRESSED_EXTENSIONS = (".gz",)

DATASET_COLUMNS = ["sequence_id", "sequence", "length"]

FORMAT_LABELS = {"fasta": "FASTA", "table": "CSV/TSV"}


def format_label(file_format):
    """Human-readable name for a detected format."""
    return FORMAT_LABELS.get(file_format, str(file_format).upper())


def is_compressed(filename):
    """True for files this module should read through gzip."""
    return filename.lower().endswith(COMPRESSED_EXTENSIONS)


def open_text(filename, errors=None):
    """Open a plain or gzipped text file for reading."""
    if is_compressed(filename):
        return gzip.open(filename, "rt", encoding="utf-8", errors=errors)

    return open(filename, "r", encoding="utf-8", errors=errors)


def strip_compression(filename):
    """Drop a trailing .gz so the real extension can be inspected."""
    if is_compressed(filename):
        return os.path.splitext(filename)[0]

    return filename


def detect_format(filename):
    """Return "fasta" or "table" for a dataset file.

    The extension decides first (ignoring a trailing .gz); if it is unknown
    the first non-blank line is inspected, and a leading '>' means FASTA.
    """
    extension = os.path.splitext(strip_compression(filename))[1].lower()

    if extension in FASTA_EXTENSIONS:
        return "fasta"

    if extension in (".csv", ".tsv"):
        return "table"

    try:
        with open_text(filename, errors="ignore") as handle:
            for line in handle:
                line = line.strip()

                if line:
                    return "fasta" if line.startswith(">") else "table"
    except OSError:
        pass

    return "table"


def _iter_fasta_builtin(filename):
    """Minimal FASTA reader used when BioPython is unavailable."""
    with open_text(filename) as handle:
        record_id = None
        chunks = []

        for line in handle:
            line = line.strip()

            if not line or line.startswith((";", "#")):
                continue

            if line.startswith(">"):
                if record_id is not None:
                    yield record_id, "".join(chunks)

                header = line[1:].strip()
                record_id = header.split()[0] if header else ""
                chunks = []
            elif record_id is not None:
                chunks.append(line)

        if record_id is not None:
            yield record_id, "".join(chunks)


def iter_fasta_records(filename):
    """Yield (record_id, sequence) pairs from a FASTA file.

    Uses BioPython's SeqIO when it is installed and falls back to the
    built-in reader otherwise, so the pipeline still runs without it.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("FASTA file not found: " + filename)

    try:
        from Bio import SeqIO
    except ImportError:
        yield from _iter_fasta_builtin(filename)
        return

    with open_text(filename) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            yield record.id, str(record.seq)


def _finalise(dataset, valid_only=False, limit=None):
    """Drop empties, optionally keep valid bases only, add the length column."""
    dataset = dataset[dataset["sequence"].str.len() > 0]

    if valid_only:
        keep = dataset["sequence"].apply(
            lambda sequence: set(sequence).issubset(VALID_BASES)
        )
        dataset = dataset[keep]

    dataset = dataset.copy()
    dataset["length"] = dataset["sequence"].str.len()

    dataset = dataset.reset_index(drop=True)

    if limit is not None:
        dataset = dataset.head(limit)

    return dataset


def load_fasta_dataset(filename, limit=None, valid_only=False):
    """Load sequences from a FASTA file.

    Record ids come from the FASTA headers (the part before the first
    space); ids are generated when a header is empty. FASTA carries no
    class labels, so the returned frame has no 'class' column.

    Returns a DataFrame with columns: sequence_id, sequence, length.
    """
    rows = []

    for index, (record_id, sequence) in enumerate(iter_fasta_records(filename)):
        sequence = sequence.strip().upper()

        if not sequence:
            continue

        if valid_only and not set(sequence).issubset(VALID_BASES):
            continue

        rows.append({
            "sequence_id": record_id or f"sequence_{index}",
            "sequence": sequence,
        })

        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return _finalise(pd.DataFrame(rows), valid_only=False, limit=limit)


def load_table_dataset(filename, limit=None, valid_only=False):
    """Load sequences from a CSV/TSV file.

    The separator is auto-detected, so .csv and tab-separated .txt files
    both work. The sequence column may be named 'sequence' or 'seq';
    otherwise the first column is used.

    Returns a DataFrame with columns: sequence_id, sequence, length
    (plus 'class' when the file provides it).
    """
    with open_text(filename) as handle:
        data = pd.read_csv(handle, sep=None, engine="python")

    columns = {column.lower(): column for column in data.columns}

    if "sequence" in columns:
        sequence_column = columns["sequence"]
    elif "seq" in columns:
        sequence_column = columns["seq"]
    else:
        sequence_column = data.columns[0]

    dataset = pd.DataFrame()

    if "sequence_id" in columns:
        dataset["sequence_id"] = data[columns["sequence_id"]].astype(str)
    elif "id" in columns:
        dataset["sequence_id"] = data[columns["id"]].astype(str)
    else:
        dataset["sequence_id"] = [
            f"sequence_{index}" for index in range(len(data))
        ]

    dataset["sequence"] = (
        data[sequence_column].astype(str).str.strip().str.upper()
    )

    if "class" in columns:
        dataset["class"] = data[columns["class"]]

    return _finalise(dataset, valid_only=valid_only, limit=limit)


def load_sequence_dataset(filename, limit=None, valid_only=False,
                          file_format=None):
    """Load a sequence dataset from FASTA (primary) or CSV/TSV.

    The format is detected from the file unless `file_format` is given
    ("fasta" or "table"/"csv").

    Returns a DataFrame with columns: sequence_id, sequence, length
    (plus 'class' for CSV/TSV files that provide it). The format that was
    used is recorded in `dataset.attrs["format"]`.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("Dataset not found: " + filename)

    file_format = (file_format or detect_format(filename)).lower()

    if file_format in ("csv", "tsv", "table"):
        file_format = "table"
    elif file_format in ("fasta", "fa"):
        file_format = "fasta"
    else:
        raise ValueError("Unknown dataset format: " + file_format)

    if file_format == "fasta":
        dataset = load_fasta_dataset(
            filename,
            limit=limit,
            valid_only=valid_only,
        )
    else:
        dataset = load_table_dataset(
            filename,
            limit=limit,
            valid_only=valid_only,
        )

    dataset.attrs["format"] = file_format
    dataset.attrs["source"] = filename

    return dataset


def create_fasta_from_dataset(input_file, output_file, limit=None):
    """Write a FASTA file from a CSV/TSV sequence dataset.

    Existing sequence ids are reused as FASTA headers so the generated
    file lines up with the table it came from.
    """
    dataset = load_sequence_dataset(input_file, limit=limit)

    print("Creating FASTA file...")
    print("Sequences found:", len(dataset))

    directory = os.path.dirname(output_file)

    if directory:
        os.makedirs(directory, exist_ok=True)

    records = dataset[["sequence_id", "sequence"]].itertuples(index=False)

    with open(output_file, "w", encoding="utf-8") as fasta_file:
        for sequence_id, sequence in records:
            fasta_file.write(f">{sequence_id}\n")
            fasta_file.write(sequence + "\n")

    print("FASTA file created:", output_file)

    return len(dataset)


def read_fasta(filename):
    """Return just the sequences of a FASTA file, in file order."""
    return [sequence for _, sequence in iter_fasta_records(filename)]


def create_chunks(sequence, chunk_size=1000):
    chunks = []

    for start in range(0, len(sequence), chunk_size):
        end = start + chunk_size
        chunk = sequence[start:end]
        chunks.append(chunk)

    return chunks


def read_targets(filename):
    """Read a target dataset from CSV.

    Uses the 'target' column when present, otherwise 'sequence',
    otherwise the first column. Blank values and '#' comments are ignored.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("Target file not found: " + filename)

    data = pd.read_csv(filename)
    columns = {column.lower(): column for column in data.columns}

    if "target" in columns:
        target_column = columns["target"]
    elif "sequence" in columns:
        target_column = columns["sequence"]
    else:
        target_column = data.columns[0]

    targets = []

    for value in data[target_column]:
        target = str(value).strip().upper()

        if not target or target == "NAN" or target.startswith("#"):
            continue

        targets.append(target)

    return targets


def main():
    """CSV/TSV -> FASTA conversion helper.

        python -m src.parser --input data/human_sequences.csv \
            --output data/genome.fasta
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a CSV/TSV sequence dataset to FASTA",
    )
    parser.add_argument("--input", default="data/human_sequences.csv")
    parser.add_argument("--output", default="data/genome.fasta")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Convert only the first N sequences (0 = all)",
    )

    args = parser.parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    create_fasta_from_dataset(args.input, args.output, limit=limit)


if __name__ == "__main__":
    main()
