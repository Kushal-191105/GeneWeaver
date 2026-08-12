from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Static,
    ProgressBar,
)


class GeneWeaverDashboard(App):
    """
    Basic Week 2 GeneWeaver Textual dashboard.
    """

    TITLE = "GeneWeaver - GPU CRISPR Alignment"

    def compose(self) -> ComposeResult:

        yield Header()

        yield Static(
            "GeneWeaver GPU Alignment Engine",
            id="title",
        )

        yield Static(
            "Genome Processing",
            id="status",
        )

        yield ProgressBar(
            total=100,
            show_eta=True,
            id="progress",
        )

        yield Static(
            "Initializing...",
            id="metrics",
        )

        yield Footer()

    def update_progress(
        self,
        progress: int,
        message: str,
    ):
        """
        Update progress information.
        """

        progress_bar = self.query_one(
            "#progress",
            ProgressBar,
        )

        metrics = self.query_one(
            "#metrics",
            Static,
        )

        progress_bar.update(progress)

        metrics.update(message)


if __name__ == "__main__":
    app = GeneWeaverDashboard()
    app.run()
