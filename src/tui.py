import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Label, Button, RichLog, ProgressBar
from src.gpu_device import get_gpu_device_info


class GeneWeaverTUI(App):
    """
    Textual Terminal User Interface for GeneWeaver CRISPR Alignment Engine.
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

    def update_progress(self, progress: float, status_text: str) -> None:
        """Updates the TUI progress bar and current status text."""
        p_bar = self.query_one("#pipeline-progress", ProgressBar)
        s_lbl = self.query_one("#status-label", Label)
        p_bar.progress = progress
        s_lbl.update(f"Status: {status_text}")


if __name__ == "__main__":
    app = GeneWeaverTUI()
    if "--smoke-test" in sys.argv:
        print("GeneWeaver TUI with Progress Bar compiled successfully.")
    else:
        app.run()
