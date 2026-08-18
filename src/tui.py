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
from src.cpu_alignment import find_matches_with_mismatches as cpu_align
from src.gpu_alignment import gpu_find_matches_with_mismatches as gpu_align
from benchmark import benchmark_gpu_alignment, benchmark_cpu_alignment


class GeneWeaverTUI(App):
    """
    Interactive Terminal User Interface for GeneWeaver CRISPR Alignment Engine.
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
                    yield Label("CRISPR TARGET SPEC", classes="card-title")
                    yield Label("Target: ATGCCCCAACTAAATACTAC")
                    yield Label("Length: 20 bp")
                    yield Label("Max Mismatches: 2")
                    yield Label("Chunk Size: 1,000 bp")

                with Container(classes="panel"):
                    yield Label("ACTIONS", classes="card-title")
                    yield Button("▶ Run Alignment (R)", id="btn-run", variant="primary")
                    yield Button("⚡ Benchmark (B)", id="btn-bench", variant="success")

            with Vertical(id="main-panel"):
                with Container(id="progress-container"):
                    yield Label("PIPELINE PROGRESS", classes="card-title")
                    yield Label("Status: Ready", id="status-label")
                    yield ProgressBar(id="pipeline-progress", show_percentage=True, show_eta=False, total=100)

                with Container(classes="panel"):
                    yield Label("EXECUTION & ACTIVITY LOG", classes="card-title")
                    yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]GeneWeaver TUI Initialized.[/bold green]")
        log.write("[cyan]Press [bold]R[/bold] to run sequence alignment, [bold]B[/bold] to run benchmark, [bold]Q[/bold] to quit.[/cyan]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self.action_run_alignment()
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
        log.write("[bold yellow]Initiating CRISPR Off-Target Scan...[/bold yellow]")

        sequences = read_fasta("data/genome.fasta")
        genome = "".join(sequences)
        target = read_target("data/target.txt")
        log.write(f"Loaded genome: [bold]{len(genome):,}[/bold] bp | Target: [bold]{target}[/bold]")

        # Chunking
        self.call_from_thread(self.update_progress, 30, "Chunking genomic sequence...")
        chunks = create_chunks(genome, chunk_size=1000)
        log.write(f"Partitioned into [bold]{len(chunks):,}[/bold] chunks of 1,000 bp.")

        # GPU Offloading and Alignment
        self.call_from_thread(self.update_progress, 60, "Offloading to CUDA VRAM & Executing Kernel...")
        log.write("[cyan]Launching JIT-compiled CUDA alignment kernel on GPU...[/cyan]")
        t0 = time.perf_counter()
        matches = gpu_align(genome, target, max_mismatches=2)
        gpu_duration = time.perf_counter() - t0

        self.call_from_thread(self.update_progress, 100, f"Completed in {gpu_duration*1000:.2f} ms")
        log.write(f"[bold green]✓ Alignment Completed![/bold green] Found [bold]{len(matches)}[/bold] off-target site(s).")
        log.write(f"[magenta]Total GPU Execution Time: {gpu_duration*1000:.2f} ms[/magenta]")

        for m in matches[:5]:
            log.write(f"  → Pos [bold]{m['position']:,}[/bold]: Seq [bold]{m['sequence']}[/bold] (Mismatches: {m['mismatches']})")

    @work(thread=True)
    def action_run_benchmark(self) -> None:
        log = self.query_one(RichLog)
        sample_len = 200000
        self.call_from_thread(self.update_progress, 15, f"Benchmarking CPU vs GPU ({sample_len:,} bp)...")
        log.write(f"[bold yellow]Starting CPU vs GPU Comparative Benchmark ({sample_len:,} bp)...[/bold yellow]")

        sequences = read_fasta("data/genome.fasta")
        genome = "".join(sequences)[:sample_len]
        target = read_target("data/target.txt")

        # CPU baseline
        self.call_from_thread(self.update_progress, 40, "Running single-threaded CPU baseline...")
        cpu_res = benchmark_cpu_alignment(genome, target, max_mismatches=2)
        log.write(f"CPU Baseline: [bold]{cpu_res['total_cpu_sec']*1000:.2f} ms[/bold]")

        # GPU accelerated
        self.call_from_thread(self.update_progress, 80, "Running CUDA GPU kernel acceleration...")
        gpu_res = benchmark_gpu_alignment(genome, target, max_mismatches=2)
        log.write(f"GPU Accelerated: [bold]{gpu_res['total_gpu_sec']*1000:.2f} ms[/bold] (Kernel: {gpu_res['kernel_execution_sec']*1000:.3f} ms)")

        speedup = cpu_res["total_cpu_sec"] / gpu_res["total_gpu_sec"]
        self.call_from_thread(self.update_progress, 100, f"Benchmark Complete: {speedup:.1f}x Speedup")
        log.write(f"[bold green]⚡ SPEEDUP FACTOR: {speedup:.2f}x FASTER ON GPU[/bold green]")


if __name__ == "__main__":
    app = GeneWeaverTUI()
    if "--smoke-test" in sys.argv:
        print("Connected Alignment to TUI successfully.")
    else:
        app.run()
