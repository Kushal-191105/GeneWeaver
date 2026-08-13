import os

import pandas as pd
from Bio import SeqIO


VALID_BASES = set("ATGCN")


def load_sequence_dataset(filename, limit=None, valid_only=False):
    """Load a sequence dataset from a CSV/TSV file.

    The separator is auto-detected, so .csv and tab-separated .txt files
    both work. The sequence column may be named 'sequence' or 'seq';
    otherwise the first column is used.

    Returns a DataFrame with columns: sequence_id, sequence, length
    (plus 'class' when the file provides it).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("Dataset not found: " + filename)

    data = pd.read_csv(filename, sep=None, engine="python")

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

    dataset = dataset[dataset["sequence"].str.len() > 0]

    if valid_only:
        keep = dataset["sequence"].apply(
            lambda sequence: set(sequence).issubset(VALID_BASES)
        )
        dataset = dataset[keep]

    dataset["length"] = dataset["sequence"].str.len()

    dataset = dataset.reset_index(drop=True)

    if limit is not None:
        dataset = dataset.head(limit)

    return dataset




def create_fasta_from_dataset(input_file, output_file):
    data = pd.read_csv(input_file, sep="\t")

    print("Creating FASTA file...")
    print("Sequences found:", len(data))

    with open(output_file, "w", encoding="utf-8") as fasta_file:
        for index, sequence in enumerate(data["sequence"]):

            sequence = str(sequence).strip().upper()

            if sequence:
                fasta_file.write(f">sequence_{index}\n")
                fasta_file.write(sequence + "\n")

    print("FASTA file created successfully.")


def read_fasta(filename):
    sequences = []

    for record in SeqIO.parse(filename, "fasta"):
        sequences.append(str(record.seq))

    return sequences


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