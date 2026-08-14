import argparse

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from src.chunking import DEFAULT_CHUNK_SIZE
from src.parser import format_label, load_sequence_dataset, read_targets
from src.run_state import AlignmentRun, device_status, format_bases

DEFAULT_INPUT = "data/genome.fasta"
DEFAULT_TARGET = "data/targets.csv"
DEFAULT_LIMIT = 200
MAX_TABLE_ROWS = 100


class GeneWeaverDashboard(App):
    """Live view of a GeneWeaver alignment run."""

    TITLE = "GeneWeaver - GPU CRISPR Alignment"
    SUB_TITLE = "off-target scan"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    CSS = """
    #panels {
        height: auto;
    }

    #dataset, #device {
        width: 1fr;
        border: round $accent;
        padding: 0 1;
        height: 7;
    }

    #status {
        padding: 0 1;
        height: 2;
    }

    #progress {
        padding: 0 1;
    }

    #metrics {
        padding: 0 1;
        height: 3;
    }

    #matches {
        height: 1fr;
        border: round $accent;
    }
    """

    def __init__(self, run=None, **kwargs):
        super().__init__(**kwargs)

        self.run_state = run
        self._rows = 0

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical():
            with Horizontal(id="panels"):
                yield Static(self._dataset_text(), id="dataset")
                yield Static(self._device_text(), id="device")

            yield Static("Waiting to start...", id="status")
            yield ProgressBar(total=100, show_eta=True, id="progress")
            yield Static("", id="metrics")
            yield DataTable(id="matches")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#matches", DataTable)
        table.cursor_type = "row"
        table.add_columns("sequence_id", "target", "position", "mismatches")

        if self.run_state is not None:
            self.run_state.start(callback=self._on_progress)

    # ------------------------------------------------------------------
    # panels
    # ------------------------------------------------------------------

    def _dataset_text(self):
        if self.run_state is None:
            return "[b]Dataset[/b]\nno run attached"

        plan = self.run_state.plan

        return (
            "[b]Dataset[/b]\n"
            f"{self.run_state.source}\n"
            f"sequences: {plan['sequences']}\n"
            f"bases: {format_bases(plan['total_bases'])}\n"
            f"chunks: {plan['chunks']} x "
            f"{format_bases(plan['chunk_size'])} "
            f"(overlap {plan['overlap']})"
        )

    def _device_text(self):
        backend = self.run_state.backend if self.run_state else "unknown"
        status = device_status(backend)

        lines = ["[b]Device[/b]", f"backend: {status['backend']}"]

        if status["cores"]:
            lines.append(f"SM cores: {status['cores']}")

        if status["memory"]:
            memory = status["memory"]
            used = memory["used"] / 1024 ** 2
            total = memory["total"] / 1024 ** 2
            share = 100.0 * memory["used"] / memory["total"]

            lines.append(f"VRAM: {used:.0f} / {total:.0f} MiB ({share:.1f}%)")
        else:
            lines.append("VRAM: n/a (no CUDA device)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # progress
    # ------------------------------------------------------------------

    def _on_progress(self, snapshot):
        """Called from the worker thread after every chunk."""
        self.call_from_thread(self.apply_snapshot, snapshot)

    def apply_snapshot(self, snapshot):
        """Render a run snapshot. Safe to call directly in tests."""
        self.query_one("#progress", ProgressBar).update(
            total=100,
            progress=snapshot["progress"],
        )

        target_number = snapshot["target_index"] + 1

        self.query_one("#status", Static).update(
            f"[b]Target {target_number}/{snapshot['targets_total']}[/b]  "
            f"{snapshot['target']}"
        )

        self.query_one("#metrics", Static).update(
            f"chunks {snapshot['chunks_done']}/{snapshot['chunks_total']}  |  "
            f"scanned {format_bases(snapshot['bases_done'])}  |  "
            f"matches {snapshot['matches']}\n"
            f"elapsed {snapshot['elapsed']:.2f}s  |  "
            f"{format_bases(snapshot['throughput'])}/s  |  "
            f"backend {snapshot['backend']}"
        )

        self.query_one("#device", Static).update(self._device_text())

        self._fill_table(snapshot["recent_matches"])

        if snapshot["finished"]:
            note = "done"

            if snapshot["error"]:
                note = f"failed: {snapshot['error']}"

            self.query_one("#status", Static).update(
                f"[b]Scan {note}[/b]  "
                f"{snapshot['matches']} matches in "
                f"{snapshot['elapsed']:.2f}s"
            )

    def _fill_table(self, matches):
        table = self.query_one("#matches", DataTable)

        for match in matches[self._rows:MAX_TABLE_ROWS]:
            table.add_row(
                str(match["sequence_id"]),
                str(match["target"]),
                str(match["position"]),
                str(match["mismatches"]),
            )

            self._rows += 1

    def update_progress(self, progress: int, message: str):
        """Manual progress update (kept for scripted use and tests)."""
        self.query_one("#progress", ProgressBar).update(
            total=100,
            progress=progress,
        )
        self.query_one("#metrics", Static).update(message)


def parse_args():
    parser = argparse.ArgumentParser(description="GeneWeaver TUI dashboard")

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--format",
        choices=["auto", "fasta", "csv"],
        default="auto",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-mismatches", type=int, default=2)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Bases per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )

    return parser.parse_args()


def build_run(args):
    """Load the dataset and targets and prepare the run."""
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

    label = format_label(dataset.attrs.get("format"))

    return AlignmentRun(
        dataset,
        targets,
        mode=args.mode,
        max_mismatches=args.max_mismatches,
        chunk_size=args.chunk_size,
        source=f"{args.input} [{label}]",
    )


def main():
    args = parse_args()

    GeneWeaverDashboard(run=build_run(args)).run()


if __name__ == "__main__":
    main()
