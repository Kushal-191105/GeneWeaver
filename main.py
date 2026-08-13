import argparse
import os

import pandas as pd

from src.parser import load_sequence_dataset, read_targets
from src.pipeline import run_alignment

DEFAULT_INPUT = "data/human_sequences.csv"
DEFAULT_TARGET = "data/target.txt"
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
        help=f"Human DNA dataset, CSV or TSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target file, one target per line (default: {DEFAULT_TARGET})",
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

    dataset = load_sequence_dataset(args.input, limit=limit)
    targets = read_targets(args.target)

    if dataset.empty:
        raise SystemExit("No sequences found in " + args.input)

    if not targets:
        raise SystemExit("No targets found in " + args.target)

    label = "CPU" if args.mode == "cpu" else "GPU"

    print(f"GeneWeaver - {label} Alignment")
    print(LINE)
    print("Dataset:", args.input)
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
        result = run_alignment(
            dataset,
            target,
            mode=args.mode,
            max_mismatches=args.max_mismatches,
        )

        all_matches.extend(result["matches"])
        total_time += result["elapsed"]

        summary_rows.append({
            "target": result["target"],
            "target_length": result["target_length"],
            "positions": result["positions"],
            "matches": len(result["matches"]),
            "seconds": round(result["elapsed"], 6),
        })

        print()
        print("Target:", result["target"])
        print("Target length:", result["target_length"])
        print("Alignment positions:", result["positions"])
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
