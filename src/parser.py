import pandas as pd
from Bio import SeqIO




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

def read_target(filename):
    with open(filename, "r", encoding="utf-8") as file:
        target = file.read().strip().upper()

    return target