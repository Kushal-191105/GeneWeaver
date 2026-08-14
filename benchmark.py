import argparse

from src.parser import format_label, load_sequence_dataset, read_targets
from src.pipeline import run_alignment

DEFAULT_INPUT = "data/genome.fasta"
DEFAULT_TARGET = "data/targets.csv"
DEFAULT_LIMIT = 200
LINE = "=" * 50


def parse_args():
    parser = argparse.ArgumentParser(description="GeneWeaver benchmark")

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"FASTA (primary) or CSV/TSV dataset (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "fasta", "csv"],
        default="auto",
        help="Force the input format instead of detecting it (default: auto)",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"First N sequences (default: {DEFAULT_LIMIT}, 0 = all)",
    )
    parser.add_argument("--max-mismatches", type=int, default=2)

    return parser.parse_args()


def run_mode(dataset, targets, mode, max_mismatches):
    positions = 0
    matches = 0
    elapsed = 0.0
    backend = mode

    for target in targets:
        result = run_alignment(
            dataset,
            target,
            mode=mode,
            max_mismatches=max_mismatches,
        )

        positions += result["positions"]
        matches += len(result["matches"])
        elapsed += result["elapsed"]
        backend = result["backend"]

    return {
        "positions": positions,
        "matches": matches,
        "elapsed": elapsed,
        "backend": backend,
    }


def report(title, result):
    print(f"\n{title}")
    print("-" * 50)
    print("Backend:", result["backend"])
    print("Alignment positions:", result["positions"])
    print("Matches found:", result["matches"])
    print("Execution time:", f"{result['elapsed']:.6f}", "seconds")


def main():
    args = parse_args()

    limit = args.limit if args.limit and args.limit > 0 else None

    file_format = None if args.format == "auto" else args.format

    dataset = load_sequence_dataset(
        args.input,
        limit=limit,
        file_format=file_format,
    )
    targets = read_targets(args.target)

    if dataset.empty:
        raise SystemExit("No sequences found in " + args.input)

    if not targets:
        raise SystemExit("No targets found in " + args.target)

    print("GeneWeaver Performance Benchmark")
    print(LINE)
    print("Dataset:", args.input)
    print("Format:", format_label(dataset.attrs.get("format")))
    print("Sequences:", len(dataset))
    print("Total bases:", int(dataset["length"].sum()))
    print("Targets:", len(targets))

    cpu_result = run_mode(dataset, targets, "cpu", args.max_mismatches)
    report("CPU", cpu_result)

    gpu_result = run_mode(dataset, targets, "gpu", args.max_mismatches)
    report("GPU", gpu_result)

    print("\nPerformance")
    print("-" * 50)

    if gpu_result["elapsed"] > 0:
        speedup = cpu_result["elapsed"] / gpu_result["elapsed"]
        print("GPU speedup:", f"{speedup:.2f}x")
    else:
        print("GPU speedup: n/a")

    if cpu_result["matches"] == gpu_result["matches"]:
        print("Result check: PASSED (CPU and GPU agree)")
    else:
        print("Result check: FAILED")
        print("CPU matches:", cpu_result["matches"])
        print("GPU matches:", gpu_result["matches"])


if __name__ == "__main__":
    main()
