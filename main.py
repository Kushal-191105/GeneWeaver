import pandas as pd
from src.parser import read_fasta, create_chunks
from src.parser import read_target
import time

from src.cpu_alignment import find_matches_with_mismatches

INPUT_FILE = "data/human_sequences.txt"

data = pd.read_csv(INPUT_FILE, sep="\t")

valid_bases = set("ATGC")
ambiguous_bases = set("N")


def classify_sequence(sequence):
    sequence = str(sequence).upper().strip()
    characters = set(sequence)

    if characters.issubset(valid_bases):
        return "valid"

    if characters.issubset(valid_bases | ambiguous_bases):
        return "ambiguous"

    return "invalid"


data["status"] = data["sequence"].apply(classify_sequence)

print("Total sequences:", len(data))
print("Valid sequences:", (data["status"] == "valid").sum())
print("Ambiguous sequences:", (data["status"] == "ambiguous").sum())
print("Invalid sequences:", (data["status"] == "invalid").sum())


# Sequence length analysis
data["length"] = data["sequence"].astype(str).str.len()

print("\nSequence Length Statistics:")
print("Minimum length:", data["length"].min())
print("Maximum length:", data["length"].max())
print("Average length:", round(data["length"].mean(), 2))
print("Total bases:", data["length"].sum())

# Genome statistics
print("\n========== Genome Statistics ==========")

print("Total sequences:", len(data))
print("Total bases:", data["length"].sum())
print("Average sequence length:", round(data["length"].mean(), 2))
print("Shortest sequence:", data["length"].min())
print("Longest sequence:", data["length"].max())

print("\nClass distribution:")
print(data["class"].value_counts().sort_index())

# Genome chunking
sequences = read_fasta("data/genome.fasta")

genome = "".join(sequences)

chunk_size = 1000

chunks = create_chunks(genome, chunk_size)

print("\n========== Genome Chunking ==========")
print("Total genome length:", len(genome))
print("Chunk size:", chunk_size)
print("Number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk length:", len(chunks[0]))

# Chunk validation
print("\n========== Chunk Validation ==========")

valid_chunks = 0
invalid_chunks = 0

valid_bases = set("ATGCN")

for chunk in chunks:
    if set(chunk.upper()).issubset(valid_bases):
        valid_chunks += 1
    else:
        invalid_chunks += 1

print("Valid chunks:", valid_chunks)
print("Invalid chunks:", invalid_chunks)

# Check that no DNA was lost during chunking
reconstructed_genome = "".join(chunks)

print("Original genome length:", len(genome))
print("Reconstructed length:", len(reconstructed_genome))

if genome == reconstructed_genome:
    print("Chunk validation: PASSED")
else:
    print("Chunk validation: FAILED")

# Read CRISPR target
target = read_target("data/target.txt")

print("\n========== Target Sequence ==========")
print("Target:", target)
print("Target length:", len(target))

# CPU exact alignment
'''matches = find_exact_matches(genome, target)

print("\n========== CPU Exact Matching ==========")
print("Target:", target)
print("Matches found:", len(matches))

for match in matches[:10]:
    print(match)'''

# Start CPU timer
start_time = time.perf_counter()

matches = find_matches_with_mismatches(
    genome,
    target,
    max_mismatches=2
)

# Stop CPU timer
end_time = time.perf_counter()

execution_time = end_time - start_time

print("\n========== CPU Alignment ==========")
print("Target:", target)
print("Maximum mismatches allowed:", 2)
print("Matches found:", len(matches))

for match in matches[:10]:
    print(match)

print("\n========== CPU Benchmark ==========")
print("Genome length:", len(genome))
print("Target length:", len(target))
print("Matches found:", len(matches))
print("CPU execution time:", round(execution_time, 6), "seconds")