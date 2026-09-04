import argparse
import os

import pandas as pd

from src.parser import format_label, load_sequence_dataset, read_targets
from src.chunking import DEFAULT_CHUNK_SIZE
from src.distributed import cluster_summary, run_distributed_alignment
from src.pipeline import attach_scores, run_chunked_alignment
from src.scoring import severity_counts

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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "Week 3: spread chunks across every available GPU with Dask, "
            "balanced by base count"
        ),
    )
    parser.add_argument(
        "--kernel",
        choices=["auto", "shared", "naive"],
        default="auto",
        help=(
            "Week 4: CUDA kernel to launch - 'shared' stages each tile in "
            "shared memory, 'naive' is the Week 2 global-memory kernel "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip the Week 3 biological scoring and report raw hits only",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many of the worst-scoring hits to print (default: 10)",
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


def print_top_hits(matches, top):
    """The worst off-targets, ranked, with their base pairing."""
    if not matches or top <= 0:
        return

    print("\nMost dangerous off-targets")
    print("-" * 50)

    header = (
        f"{'severity':>9}  {'score':>7}  {'sequence':<14} {'pos':>7}  "
        f"{'mm':>2} {'seed':>4}  {'pam':<4} pairing"
    )
    print(header)

    for match in matches[:top]:
        print(
            f"{match['severity']:>9}  {match['score']:>7.2f}  "
            f"{str(match['sequence_id'])[:14]:<14} {match['position']:>7}  "
            f"{match['mismatches']:>2} {match['seed_mismatches']:>4}  "
            f"{(match['pam'] or '-'):<4} {match['alignment']}"
        )

    print("\n('x' marks a mismatched base; the last 12 positions are the "
          "PAM-proximal seed.)")


def print_next_steps(args):
    """Point at the dashboards, which are separate commands from this one."""
    print("\nDashboards")
    print("-" * 50)
    print("  Web:      python -m web.server --open")
    print("  Terminal: python -m src.dashboard --target", args.target)

    if args.target.endswith("targets.csv"):
        print(
            "\nEvery hit above scored 'low' because the guides in "
            f"{args.target} have no NGG PAM,"
        )
        print(
            "  so SpCas9 could not cut any of those sites. For guides whose "
            "sites really are"
        )
        print("  followed by a PAM, generate a realistic guide list first:")
        print("\n    python scripts/pick_guides.py")
        print("    python main.py --target data/guides.csv --max-mismatches 4")


def print_severity(matches):
    counts = severity_counts(matches)

    print("\nOff-target severity")
    print("-" * 50)

    for label in ("critical", "high", "moderate", "low"):
        print(f"{label:>9}: {counts[label]}")


def export_matches(matches, output_file):
    directory = os.path.dirname(output_file)

    if directory:
        os.makedirs(directory, exist_ok=True)

    columns = [
        "sequence_id",
        "target",
        "position",
        "sequence",
        "mismatches",
    ]

    # Scored runs carry the biology alongside the coordinates.
    if matches and "score" in matches[0]:
        columns += ["score", "severity", "seed_mismatches", "pam", "pam_status"]

    results = pd.DataFrame(matches, columns=columns)

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

    if args.distributed:
        cluster = cluster_summary(args.mode)
        print(
            "Scheduler:", cluster["scheduler"],
            "| devices:", cluster["device_count"],
        )

    if limit is not None:
        print(f"(limited to the first {limit} sequences, use --limit 0 for all)")

    if args.stats:
        print_dataset_stats(dataset)

    all_matches = []
    summary_rows = []
    total_time = 0.0

    for target in targets:
        if args.distributed:
            result = run_distributed_alignment(
                dataset,
                target,
                mode=args.mode,
                max_mismatches=args.max_mismatches,
                chunk_size=args.chunk_size,
                kernel=args.kernel,
            )
        else:
            result = run_chunked_alignment(
                dataset,
                target,
                mode=args.mode,
                max_mismatches=args.max_mismatches,
                chunk_size=args.chunk_size,
                kernel=args.kernel,
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
            print("Backend:", result["backend"], "| kernel:", result["kernel"])

        if args.distributed and result.get("devices"):
            print("Device split:")

            for row in result["devices"]:
                print(
                    f"  gpu{row['id']} {row['name']}: {row['chunks']} chunks, "
                    f"{row['bases']} bases ({row['share']:.2f}%), "
                    f"{row['seconds']:.3f}s"
                )

            print(
                "  load imbalance: "
                f"{result['balance']['imbalance'] * 100:.3f}%"
            )

    if len(targets) > 1:
        print("\nSummary")
        print("-" * 50)
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(f"\nTotal {label} time:", f"{total_time:.6f}", "seconds")
        print("Total matches:", len(all_matches))

    if not args.no_score:
        all_matches = attach_scores(all_matches, dataset)

        print_severity(all_matches)
        print_top_hits(all_matches, args.top)

    if not args.no_export:
        rows = export_matches(all_matches, args.output)
        print(f"\nMatches exported to: {args.output} ({rows} rows)")

    print_next_steps(args)


if __name__ == "__main__":
    main()
