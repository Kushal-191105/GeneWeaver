from src.parser import read_fasta

FASTA_FILE = "data/genome.fasta"

sequences = read_fasta(FASTA_FILE)

print("Number of sequences:", len(sequences))

print("\nFirst sequence:")
print(sequences[0])

print("\nFirst sequence length:")
print(len(sequences[0]))