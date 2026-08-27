from src.validator import validate_alignment_results
from src.parser import read_fasta, read_target


def main():
    print("Loading genomic dataset for comprehensive 3-way parity validation...")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)[:50000]
    target = read_target("data/target.txt")

    validate_alignment_results(genome, target, max_mismatches=2, test_distributed=True)


if __name__ == "__main__":
    main()
