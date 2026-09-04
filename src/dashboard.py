"""Textual dashboard for a GeneWeaver run.

Week 4 - Refine & Polish.

Week 2's dashboard proved the pipeline was alive: a progress bar and a
list of hits. That is a monitor, not an instrument. What a person
actually needs to decide whether an off-target matters is the base
pairing itself - which positions mismatch, how close they sit to the
PAM, and whether there is a PAM at all - so the polished dashboard adds
an alignment inspector underneath the table. Select any row and the
guide is drawn over the genomic site with every mismatched base printed
in red, the PAM-proximal seed region marked, and the MIT score and
severity tier alongside.

The rest of the polish is in the same spirit: severity-coloured rows so
the eye lands on the dangerous hits first, a severity tally, live
throughput and ETA, and a device panel that shows the scheduler and the
per-GPU split when the run is distributed.
"""

import argparse

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from src.chunking import DEFAULT_CHUNK_SIZE
from src.parser import format_label, load_sequence_dataset, read_targets
from src.run_state import (
    AlignmentRun,
    device_status,
    format_bases,
    format_seconds,
)
from src.scoring import SEED_LENGTH, describe

DEFAULT_INPUT = "data/genome.fasta"
DEFAULT_TARGET = "data/targets.csv"
DEFAULT_LIMIT = 200
MAX_TABLE_ROWS = 200

# Rich markup colours, one per severity tier.
SEVERITY_MARKUP = {
    "critical": "bold white on red",
    "high": "bold black on dark_orange",
    "moderate": "bold black on yellow",
    "low": "green",
}


def severity_tag(severity):
    style = SEVERITY_MARKUP.get(severity, "dim")

    return f"[{style}] {severity.upper():8s} [/]"


def render_alignment(match):
    """Draw a guide against its genomic site, mismatches in red.

    Three stacked rows, the way an alignment is read on paper:

        guide   5'-GCCGGAGCTGACCCCGGAGG-3'  AGG
        pairing    ||||||||||x|||||||||
        site    5'-GCCGGAGCTGTCCCCGGAGG-3'

    Every base that differs is printed in red on both sequence rows, so
    the mismatch is visible whichever row the eye is on.
    """
    if not match:
        return "[dim]Select a hit to inspect its base pairing.[/dim]"

    target = str(match["target"]).upper()
    found = str(match["sequence"]).upper()
    positions = set(match.get("mismatch_positions", []))
    length = len(target)
    seed_start = max(0, length - SEED_LENGTH)

    guide_row = []
    site_row = []
    pair_row = []

    for index in range(length):
        expected = target[index]
        actual = found[index] if index < len(found) else "-"

        if index in positions:
            guide_row.append(f"[bold red]{expected}[/bold red]")
            site_row.append(f"[bold red]{actual}[/bold red]")
            pair_row.append("[bold red]x[/bold red]")
        else:
            shade = "white" if index >= seed_start else "grey62"
            guide_row.append(f"[{shade}]{expected}[/{shade}]")
            site_row.append(f"[{shade}]{actual}[/{shade}]")
            pair_row.append(f"[{shade}]|[/{shade}]")

    # The seed ruler: which stretch of the guide Cas9 will not forgive.
    ruler = (
        f"[grey42]{'-' * seed_start}[/grey42]"
        f"[cyan]{'^' * (length - seed_start)}[/cyan]"
    )

    pam = match.get("pam") or "---"
    status = match.get("pam_status", "absent")

    if status == "ngg":
        pam_markup = f"[bold cyan]{pam}[/bold cyan] [dim](NGG, cuttable)[/dim]"
    elif status == "none":
        pam_markup = f"[red]{pam}[/red] [dim](no NGG - Cas9 cannot cut)[/dim]"
    else:
        pam_markup = "[dim]--- (record ends)[/dim]"

    score = match.get("score", 0.0)
    seed_hits = match.get("seed_mismatches", 0)

    return (
        f"[b]{match['sequence_id']}[/b]  position {match['position']}   "
        f"{severity_tag(match.get('severity', 'low'))}  "
        f"MIT score [b]{score:.2f}[/b] / 100\n"
        f"  guide   5'-{''.join(guide_row)}-3'  PAM {pam_markup}\n"
        f"  pairing    {''.join(pair_row)}\n"
        f"  site    5'-{''.join(site_row)}-3'\n"
        f"  seed       {ruler}  "
        f"[dim]{len(positions)} mismatch(es), {seed_hits} in the "
        f"{SEED_LENGTH} nt seed[/dim]"
    )


class GeneWeaverDashboard(App):
    """Live view of a GeneWeaver alignment run."""

    TITLE = "GeneWeaver - GPU CRISPR Alignment"
    SUB_TITLE = "off-target scan"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "stop_run", "Stop run"),
        ("h", "toggle_help", "Scoring help"),
    ]

    CSS = """
    #panels {
        height: auto;
    }

    #dataset, #device, #severity {
        width: 1fr;
        border: round $accent;
        padding: 0 1;
        height: 8;
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
        min-height: 8;
        border: round $accent;
    }

    #alignment {
        height: 8;
        border: round $success;
        padding: 0 1;
    }

    #help {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, run=None, **kwargs):
        super().__init__(**kwargs)

        self.run_state = run
        self._rows = 0
        self._matches = []
        self._selected = None
        self._show_help = False

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical():
            with Horizontal(id="panels"):
                yield Static(self._dataset_text(), id="dataset")
                yield Static(self._device_text(), id="device")
                yield Static(self._severity_text({}), id="severity")

            yield Static("Waiting to start...", id="status")
            yield ProgressBar(total=100, show_eta=True, id="progress")
            yield Static("", id="metrics")
            yield DataTable(id="matches")
            yield Static(render_alignment(None), id="alignment")
            yield Static("", id="help")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#matches", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(
            "severity", "score", "sequence_id", "position",
            "mm", "seed", "pam", "pairing",
        )

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
            f"(overlap {plan['overlap']})\n"
            f"targets: {len(self.run_state.targets)}"
        )

    def _device_text(self, snapshot=None):
        backend = self.run_state.backend if self.run_state else "unknown"
        status = device_status(backend)

        lines = ["[b]Device[/b]", f"backend: {status['backend']}"]

        if status["name"]:
            lines.append(f"gpu: {status['name']}")

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

        if snapshot:
            lines.append(
                f"kernel: {snapshot.get('kernel', '-')}  "
                f"scheduler: {snapshot.get('scheduler', '-')}"
            )

            devices = snapshot.get("devices") or []

            if len(devices) > 1:
                split = "  ".join(
                    f"gpu{row['id']} {row['share']:.1f}%" for row in devices
                )
                lines.append(f"split: {split}")

        return "\n".join(lines)

    def _severity_text(self, counts):
        counts = counts or {}

        return (
            "[b]Off-target severity[/b]\n"
            f"[bold red]critical[/bold red]  {counts.get('critical', 0)}\n"
            f"[dark_orange]high[/dark_orange]      {counts.get('high', 0)}\n"
            f"[yellow]moderate[/yellow]  {counts.get('moderate', 0)}\n"
            f"[green]low[/green]       {counts.get('low', 0)}\n"
            "[dim]MIT score, PAM-weighted[/dim]"
        )

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
            f"elapsed {format_seconds(snapshot['elapsed'])}  |  "
            f"eta {format_seconds(snapshot['eta'])}  |  "
            f"{format_bases(snapshot['throughput'])}/s  |  "
            f"backend {snapshot['backend']}"
        )

        self.query_one("#device", Static).update(self._device_text(snapshot))
        self.query_one("#severity", Static).update(
            self._severity_text(snapshot.get("severity")))

        self._fill_table(snapshot["recent_matches"])

        if snapshot["finished"]:
            note = "stopped" if snapshot.get("stopped") else "done"

            if snapshot["error"]:
                note = f"failed: {snapshot['error']}"

            self.query_one("#status", Static).update(
                f"[b]Scan {note}[/b]  "
                f"{snapshot['matches']} matches in "
                f"{format_seconds(snapshot['elapsed'])}"
            )

    def _fill_table(self, matches):
        """Redraw the ranked table.

        Ranking can reorder rows as worse hits arrive, so the table is
        rebuilt rather than appended to; it is capped at MAX_TABLE_ROWS,
        which keeps the redraw cheap.
        """
        if matches == self._matches:
            return

        self._matches = list(matches)

        table = self.query_one("#matches", DataTable)
        table.clear()

        for match in self._matches[:MAX_TABLE_ROWS]:
            table.add_row(
                severity_tag(match.get("severity", "low")),
                f"{match.get('score', 0.0):.2f}",
                str(match["sequence_id"]),
                str(match["position"]),
                str(match["mismatches"]),
                str(match.get("seed_mismatches", 0)),
                str(match.get("pam") or "-"),
                match.get("alignment", ""),
            )

        if self._selected is None and self._matches:
            self._select(0)

    def _select(self, row):
        if 0 <= row < len(self._matches):
            self._selected = row
            self.query_one("#alignment", Static).update(
                render_alignment(self._matches[row]))

    def on_data_table_row_highlighted(self, event) -> None:
        self._select(event.cursor_row)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def action_stop_run(self) -> None:
        if self.run_state is not None:
            self.run_state.stop()
            self.query_one("#status", Static).update("[b]Stopping...[/b]")

    def action_toggle_help(self) -> None:
        self._show_help = not self._show_help
        self.query_one("#help", Static).update(describe() if self._show_help else "")

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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Spread chunks across every GPU with Dask (Week 3)",
    )
    parser.add_argument(
        "--kernel",
        choices=["auto", "shared", "naive"],
        default="auto",
        help="CUDA kernel: shared memory (Week 4) or the Week 2 baseline",
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
        distributed=getattr(args, "distributed", False),
        kernel=getattr(args, "kernel", "auto"),
    )


def main():
    args = parse_args()

    GeneWeaverDashboard(run=build_run(args)).run()


if __name__ == "__main__":
    main()
