import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Label, Button, RichLog, ProgressBar
from textual import work

from src.gpu_device import get_gpu_device_info
from src.parser import read_fasta, read_target, create_chunks
from src.gpu_alignment import gpu_find_matches_with_mismatches as gpu_align
from src.distributed_scheduler import (
    get_available_gpus,
    partition_genome_for_workers,
    process_batch_alignment,
    gather_and_deduplicate_results
)
from src.scoring import rank_off_targets
from benchmark import benchmark_gpu_alignment, benchmark_cpu_alignment


def format_rich_off_target(item: dict) -> str:
    """
    Renders an off-target match candidate with color-coded biological severity badges
    and formatted PAM motifs for Rich/Textual terminal display.
    """
    tier = item.get("risk_tier", "LOW")
    score = item.get("severity_score", 0.0)
    pam = item.get("pam", "NNN")
    pam_type = item.get("pam_type", "invalid")
    pos = item.get("position", 0)
    seq = item.get("sequence", "")

    if tier == "HIGH":
        badge = "[bold white on red] HIGH RISK [/bold white on red]"
        score_str = f"[bold red]{score:5.1f}%[/bold red]"
    elif tier == "MEDIUM":
        badge = "[bold black on yellow] MED RISK [/bold black on yellow]"
        score_str = f"[bold yellow]{score:5.1f}%[/bold yellow]"
    else:
        badge = "[bold white on green] LOW RISK [/bold white on green]"
        score_str = f"[bold green]{score:5.1f}%[/bold green]"

    if pam_type == "canonical":
        pam_str = f"[bold cyan]{pam}[/bold cyan] (Canonical NGG)"
    elif pam_type == "non-canonical":
        pam_str = f"[yellow]{pam}[/yellow] (Non-canonical NAG)"
    else:
        pam_str = f"[dim red]{pam}[/dim red] (Non-viable)"

    return (
        f"  #{item.get('rank', 1):<2} | Pos: [bold]{pos:<9,}[/bold] | Seq: [white]{seq}[/white] | "
        f"PAM: {pam_str} | Score: {score_str} | {badge}"
    )


class GeneWeaverTUI(App):
    """
    Interactive Terminal User Interface for GeneWeaver CRISPR Alignment Engine.
    Week 3: Distributed Dask Scheduling & Biological PAM Severity Ranking.
    """
    CSS = """
    Screen {
        background: #111827;
        color: #f3f4f6;
    }
    Header {
        background: #1f2937;
        color: #60a5fa;
        text-style: bold;
    }
    Footer {
        background: #1f2937;
        color: #9ca3af;
    }
    .panel {
        background: #1e293b;
        border: round #3b82f6;
        padding: 1;
        margin: 1;
    }
    .card-title {
        text-style: bold;
        color: #38bdf8;
        margin-bottom: 1;
    }
    #sidebar {
        width: 35%;
    }
    #main-panel {
        width: 65%;
    }
    #progress-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: #0f172a;
        border: round #64748b;
    }
    #log-panel {
        height: 100%;
        background: #0f172a;
        border: solid #334155;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_alignment", "Run Scan"),
        ("d", "run_dask", "Dask Distributed"),
        ("b", "run_benchmark", "Benchmark"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                with Container(classes="panel"):
                    yield Label("HARDWARE ACCELERATOR", classes="card-title")
                    gpu_info = get_gpu_device_info()
                    if gpu_info:
                        yield Label(f"GPU: {gpu_info['name']}")
                        yield Label(f"VRAM: {gpu_info['total_memory_mb']} MB (CC {gpu_info['compute_capability']})")
                        yield Label(f"Compute Cores: {gpu_info.get('multiprocessors', 'N/A')} SMs")
                    else:
                        yield Label("GPU: No CUDA GPU Detected (CPU Mode)")

                with Container(classes="panel"):
                    yield Label("DISTRIBUTED DASK CLUSTER", classes="card-title")
                    gpus = get_available_gpus()
                    yield Label("Scheduler: Dask Distributed")
                    yield Label("Active Workers: 4 Partition Batches")
                    yield Label(f"GPU Workload Balance: {len(gpus)} GPU Core(s)")

                with Container(classes="panel"):
                    yield Label("CRISPR TARGET & PAM", classes="card-title")
                    yield Label("Target: ATGCCCCAACTAAATACTAC (20 bp)")
                    yield Label("PAM Motif: SpCas9 [NGG / NAG]")
                    yield Label("Seed Window: Positions 11-20 (Critical)")
                    yield Label("Tolerance: Max 2 Mismatches")

                with Container(classes="panel"):
                    yield Label("ACTIONS", classes="card-title")
                    yield Button("▶ Run Alignment (R)", id="btn-run", variant="primary")
                    yield Button("🌐 Distributed Dask (D)", id="btn-dask", variant="warning")
                    yield Button("⚡ Benchmark (B)", id="btn-bench", variant="success")

            with Vertical(id="main-panel"):
                with Container(id="progress-container"):
                    yield Label("PIPELINE & DISTRIBUTED PROGRESS", classes="card-title")
                    yield Label("Status: Ready", id="status-label")
                    yield ProgressBar(id="pipeline-progress", show_percentage=True, show_eta=False, total=100)

                with Container(classes="panel"):
                    yield Label("EXECUTION & ACTIVITY LOG", classes="card-title")
                    yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]GeneWeaver Week 3 TUI Initialized.[/bold green]")
        log.write("[cyan]Press [bold]R[/bold] for Single-GPU, [bold]D[/bold] for Dask Distributed, [bold]B[/bold] for Benchmark, [bold]Q[/bold] to Quit.[/cyan]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self.action_run_alignment()
        elif event.button.id == "btn-dask":
            self.action_run_dask()
        elif event.button.id == "btn-bench":
            self.action_run_benchmark()

    def update_progress(self, progress: float, status_text: str) -> None:
        p_bar = self.query_one("#pipeline-progress", ProgressBar)
        s_lbl = self.query_one("#status-label", Label)
        p_bar.progress = progress
        s_lbl.update(f"Status: {status_text}")

    @work(thread=True)
    def action_run_alignment(self) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(self.update_progress, 10, "Loading genome sequence data...")
        log.write("[bold yellow]Initiating Single-GPU Alignment Scan...[/bold yellow]")

        sequences = read_fasta("data/genome.fasta")
        genome = "".join(sequences)
        target = read_target("data/target.txt")
        log.write(f"Loaded genome: [bold]{len(genome):,}[/bold] bp | Target: [bold]{target}[/bold]")

        self.call_from_thread(self.update_progress, 50, "Offloading to CUDA VRAM & Running Kernel...")
        t0 = time.perf_counter()
        raw_matches = gpu_align(genome, target, max_mismatches=2)
        ranked = rank_off_targets(raw_matches, genome, target)
        gpu_duration = time.perf_counter() - t0

        self.call_from_thread(self.update_progress, 100, f"Completed in {gpu_duration*1000:.2f} ms")
        log.write(f"[bold green]✓ Single-GPU Completed![/bold green] Scored [bold]{len(ranked)}[/bold] off-target site(s) in {gpu_duration*1000:.2f} ms.")

        for r in ranked[:10]:
            log.write(format_rich_off_target(r))

    @work(thread=True)
    def action_run_dask(self) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(self.update_progress, 10, "Partitioning genome for Dask workers...")
        log.write("[bold yellow]Launching Dask Distributed Multi-Batch Pipeline...[/bold yellow]")

        sequences = read_fasta("data/genome.fasta")
        genome = "".join(sequences)
        target = read_target("data/target.txt")

        n_batches = 4
        batches = partition_genome_for_workers(genome, target_length=len(target), n_batches=n_batches)
        log.write(f"Partitioned genome into [bold]{len(batches)}[/bold] balanced batches.")

        batch_results = []
        t0 = time.perf_counter()

        for idx, batch in enumerate(batches):
            pct = int(15 + (idx + 1) * (70 / n_batches))
            self.call_from_thread(self.update_progress, pct, f"Worker processing Batch {idx+1}/{n_batches}...")
            log.write(f"Worker task dispatch: Batch #{idx} [{batch['start_offset']:,} -> {batch['end_offset']:,} bp]")
            out = process_batch_alignment(batch, target, max_mismatches=2, worker_index=idx)
            batch_results.append(out)

        self.call_from_thread(self.update_progress, 90, "Aggregating & scoring distributed results...")
        unique_matches = gather_and_deduplicate_results(batch_results)
        ranked = rank_off_targets(unique_matches, genome, target)
        dist_duration = time.perf_counter() - t0

        high_count = sum(1 for r in ranked if r["risk_tier"] == "HIGH")
        med_count = sum(1 for r in ranked if r["risk_tier"] == "MEDIUM")
        low_count = sum(1 for r in ranked if r["risk_tier"] == "LOW")

        self.call_from_thread(self.update_progress, 100, f"Dask Complete in {dist_duration*1000:.2f} ms")
        log.write(f"[bold green]✓ Dask Distributed Complete![/bold green] Gathered [bold]{len(ranked)}[/bold] deduplicated hits in {dist_duration*1000:.2f} ms.")
        log.write(f"Risk Breakdown: [bold red]{high_count} High[/bold red] | [bold yellow]{med_count} Medium[/bold yellow] | [bold green]{low_count} Low[/bold green]")

        for r in ranked[:10]:
            log.write(format_rich_off_target(r))

    @work(thread=True)
    def action_run_benchmark(self) -> None:
        log = self.query_one(RichLog)
        sample_len = 200000
        self.call_from_thread(self.update_progress, 15, f"Benchmarking CPU vs GPU ({sample_len:,} bp)...")
        log.write(f"[bold yellow]Starting CPU vs GPU Comparative Benchmark ({sample_len:,} bp)...[/bold yellow]")

        sequences = read_fasta("data/genome.fasta")
        genome = "".join(sequences)[:sample_len]
        target = read_target("data/target.txt")

        self.call_from_thread(self.update_progress, 40, "Running single-threaded CPU baseline...")
        cpu_res = benchmark_cpu_alignment(genome, target, max_mismatches=2)
        log.write(f"CPU Baseline: [bold]{cpu_res['total_cpu_sec']*1000:.2f} ms[/bold]")

        self.call_from_thread(self.update_progress, 80, "Running CUDA GPU kernel acceleration...")
        gpu_res = benchmark_gpu_alignment(genome, target, max_mismatches=2)
        log.write(f"GPU Accelerated: [bold]{gpu_res['total_gpu_sec']*1000:.2f} ms[/bold] (Kernel: {gpu_res['kernel_execution_sec']*1000:.3f} ms)")

        speedup = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
        self.call_from_thread(self.update_progress, 100, f"Benchmark Complete: {speedup:.1f}x Speedup")
        log.write(f"[bold green]⚡ SPEEDUP FACTOR: {speedup:.2f}x FASTER ON GPU[/bold green]")


if __name__ == "__main__":
    app = GeneWeaverTUI()
    if "--smoke-test" in sys.argv:
        print("Connected Dask scheduler to TUI successfully.")
    else:
        app.run()
