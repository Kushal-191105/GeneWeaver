import argparse

from src.parser import format_label, load_sequence_dataset, read_targets
from src.chunking import DEFAULT_CHUNK_SIZE
from src.distributed import cluster_summary, run_distributed_alignment
from src.gpu_alignment import cuda_available
from src.kernel_config import global_reads_per_base, traffic_reduction
from src.pipeline import run_chunked_alignment

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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Bases per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Also time the Week 3 multi-device Dask path",
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help=(
            "Also time the Week 2 global-memory kernel against the Week 4 "
            "shared-memory kernel (needs a CUDA device)"
        ),
    )

    return parser.parse_args()


def run_mode(dataset, targets, mode, max_mismatches,
             chunk_size=DEFAULT_CHUNK_SIZE, kernel="auto", distributed=False):
    positions = 0
    matches = 0
    elapsed = 0.0
    chunks = 0
    backend = mode
    kernel_name = kernel
    devices = []

    for target in targets:
        if distributed:
            result = run_distributed_alignment(
                dataset,
                target,
                mode=mode,
                max_mismatches=max_mismatches,
                chunk_size=chunk_size,
                kernel=kernel,
            )
            devices = result["devices"]
        else:
            result = run_chunked_alignment(
                dataset,
                target,
                mode=mode,
                max_mismatches=max_mismatches,
                chunk_size=chunk_size,
                kernel=kernel,
            )

        chunks += result["chunks"]
        positions += result["positions"]
        matches += len(result["matches"])
        elapsed += result["elapsed"]
        backend = result["backend"]
        kernel_name = result.get("kernel", kernel)

    return {
        "positions": positions,
        "matches": matches,
        "elapsed": elapsed,
        "chunks": chunks,
        "backend": backend,
        "kernel": kernel_name,
        "devices": devices,
    }


def report(title, result):
    print(f"\n{title}")
    print("-" * 50)
    print("Backend:", result["backend"])
    print("Kernel:", result.get("kernel", "-"))
    print("Chunks processed:", result["chunks"])
    print("Alignment positions:", result["positions"])
    print("Matches found:", result["matches"])
    print("Execution time:", f"{result['elapsed']:.6f}", "seconds")

    for row in result.get("devices") or []:
        print(
            f"  gpu{row['id']} {row['name']}: {row['chunks']} chunks, "
            f"{row['share']:.2f}% of the bases, {row['seconds']:.3f}s"
        )


def speedup(baseline, candidate, label):
    if candidate["elapsed"] > 0:
        print(f"{label}:", f"{baseline['elapsed'] / candidate['elapsed']:.2f}x")
    else:
        print(f"{label}: n/a")


def agrees(left, right, label):
    if left["matches"] == right["matches"]:
        print(f"{label}: PASSED ({left['matches']} matches both ways)")
    else:
        print(f"{label}: FAILED ({left['matches']} vs {right['matches']})")


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
    print("Chunk size:", args.chunk_size, "bases")
    print("Total bases:", int(dataset["length"].sum()))
    print("Targets:", len(targets))

    cluster = cluster_summary("gpu")
    print("Scheduler:", cluster["scheduler"], "| devices:", cluster["device_count"])

    cpu_result = run_mode(
        dataset, targets, "cpu", args.max_mismatches, args.chunk_size)
    report("CPU", cpu_result)

    gpu_result = run_mode(
        dataset, targets, "gpu", args.max_mismatches, args.chunk_size)
    report("GPU (serial, Week 2)", gpu_result)

    results = [("CPU baseline", cpu_result), ("GPU serial", gpu_result)]

    if args.distributed:
        dist_result = run_mode(
            dataset, targets, "gpu", args.max_mismatches, args.chunk_size,
            distributed=True)
        report("GPU (distributed, Week 3)", dist_result)
        results.append(("GPU distributed", dist_result))

    kernel_results = None

    if args.kernels:
        if not cuda_available():
            print("\nKernel comparison skipped: no CUDA device on this machine.")
        else:
            naive = run_mode(
                dataset, targets, "gpu", args.max_mismatches,
                args.chunk_size, kernel="naive")
            report("CUDA global-memory kernel (Week 2)", naive)

            shared = run_mode(
                dataset, targets, "gpu", args.max_mismatches,
                args.chunk_size, kernel="shared")
            report("CUDA shared-memory kernel (Week 4)", shared)

            kernel_results = (naive, shared)

    print("\nPerformance")
    print("-" * 50)

    speedup(cpu_result, gpu_result, "GPU speedup over CPU")

    if args.distributed:
        speedup(cpu_result, dist_result, "Distributed speedup over CPU")
        speedup(gpu_result, dist_result, "Distributed speedup over serial GPU")

    if kernel_results:
        naive, shared = kernel_results
        speedup(naive, shared, "Shared-memory kernel speedup")

    guide_length = len(targets[0])

    print(
        "Global reads per base:",
        f"{global_reads_per_base(guide_length, 'naive'):.0f} (naive) ->",
        f"{global_reads_per_base(guide_length, 'shared'):.3f} (shared) =",
        f"{traffic_reduction(guide_length):.2f}x less VRAM traffic",
    )

    print("\nCorrectness")
    print("-" * 50)

    agrees(cpu_result, gpu_result, "CPU vs GPU")

    if args.distributed:
        agrees(gpu_result, dist_result, "Serial vs distributed")

    if kernel_results:
        agrees(kernel_results[0], kernel_results[1], "Naive vs shared kernel")


if __name__ == "__main__":
    main()
