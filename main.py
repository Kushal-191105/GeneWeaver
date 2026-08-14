import argparse
import os

import pandas as pd

from src.parser import format_label, load_sequence_dataset, read_targets
from src.chunking import DEFAULT_CHUNK_SIZE
from src.pipeline import run_chunked_alignment

DEFAULT_INPUT = "data/genome.fasta"
FALLBACK_INPUT = "data/human_sequences.csv"
DEFAULT_TARGET = "data/targets.csv"
DEFAULT_OUTPUT = "results/matches.csv"
DEFAULT_LIMIT = 200
LINE = "=" * 50


def parse_args():
    parser = argparse.ArgumentParser(description="GeneWeaver alignment")

    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Alignment backend to use (default: cpu)",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Human DNA dataset: FASTA (primary) or CSV/TSV "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["auto", "fasta", "csv"],
        default="auto",
        help="Force the input format instead of detecting it (default: auto)",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target dataset, one target per row (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Use only the first N sequences of the dataset "
            f"(default: {DEFAULT_LIMIT}, use 0 for the whole dataset)"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Bases per chunk sent to the alignment backend "
            f"(default: {DEFAULT_CHUNK_SIZE})"
        ),
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=2,
        help="Maximum mismatches allowed per match (default: 2)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV file for the matches (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write the matches CSV",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics before aligning",
    )

    return parser.parse_args()


def resolve_input(input_file):
    """Return the dataset path to use.

    FASTA is the primary dataset. If the default FASTA is missing but the
    CSV dataset is present, fall back to it instead of failing outright.
    """
    if os.path.exists(input_file):
        return input_file

    if input_file == DEFAULT_INPUT and os.path.exists(FALLBACK_INPUT):
        print(f"{DEFAULT_INPUT} not found, using {FALLBACK_INPUT} instead.")
        print(
            "Regenerate the FASTA with: "
            f"python -m src.parser --input {FALLBACK_INPUT} "
            f"--output {DEFAULT_INPUT}\n"
        )

        return FALLBACK_INPUT

    return input_file


def print_dataset_stats(dataset):
    print("\nDataset Statistics")
    print("-" * 50)
    print("Sequences:", len(dataset))
    print("Total bases:", int(dataset["length"].sum()))
    print("Shortest sequence:", int(dataset["length"].min()))
    print("Longest sequence:", int(dataset["length"].max()))
    print("Average length:", round(float(dataset["length"].mean()), 2))

    if "class" in dataset.columns:
        print("\nClass distribution:")
        print(dataset["class"].value_counts().sort_index().to_string())


def export_matches(matches, output_file):
    directory = os.path.dirname(output_file)

    if directory:
        os.makedirs(directory, exist_ok=True)

    results = pd.DataFrame(matches, columns=[
        "sequence_id",
        "target",
        "position",
        "sequence",
        "mismatches",
    ])

    results.to_csv(output_file, index=False)

    return len(results)


def main():
    args = parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None

    input_file = resolve_input(args.input)
    file_format = None if args.format == "auto" else args.format

    dataset = load_sequence_dataset(
        input_file,
        limit=limit,
        file_format=file_format,
    )
    targets = read_targets(args.target)

    if dataset.empty:
        raise SystemExit("No sequences found in " + input_file)

    if not targets:
        raise SystemExit("No targets found in " + args.target)

    label = "CPU" if args.mode == "cpu" else "GPU"

    print(f"GeneWeaver - {label} Alignment")
    print(LINE)
    print("Dataset:", input_file)
    print("Format:", format_label(dataset.attrs.get("format")))
    print("Sequences:", len(dataset))
    print("Total bases:", int(dataset["length"].sum()))
    print("Targets:", len(targets))

    if limit is not None:
        print(f"(limited to the first {limit} sequences, use --limit 0 for all)")

    if args.stats:
        print_dataset_stats(dataset)

    all_matches = []
    summary_rows = []
    total_time = 0.0

    for target in targets:
        result = run_chunked_alignment(
            dataset,
            target,
            mode=args.mode,
            max_mismatches=args.max_mismatches,
            chunk_size=args.chunk_size,
        )

        all_matches.extend(result["matches"])
        total_time += result["elapsed"]

        summary_rows.append({
            "target": result["target"],
            "target_length": result["target_length"],
            "positions": result["positions"],
            "chunks": result["chunks"],
            "matches": len(result["matches"]),
            "seconds": round(result["elapsed"], 6),
        })

        print()
        print("Target:", result["target"])
        print("Target length:", result["target_length"])
        print("Alignment positions:", result["positions"])
        print("Chunks:", result["chunks"], "x", result["chunk_size"], "bases")
        print("Matches found:", len(result["matches"]))
        print(f"{label} time:", f"{result['elapsed']:.6f}", "seconds")

        if args.mode == "gpu":
            print("Backend:", result["backend"])

    if len(targets) > 1:
        print("\nSummary")
        print("-" * 50)
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(f"\nTotal {label} time:", f"{total_time:.6f}", "seconds")
        print("Total matches:", len(all_matches))

    if not args.no_export:
        rows = export_matches(all_matches, args.output)
        print(f"\nMatches exported to: {args.output} ({rows} rows)")


if __name__ == "__main__":
    main()
