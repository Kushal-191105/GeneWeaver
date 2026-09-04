"""Run orchestration and live progress, independent of any interface.

`AlignmentRun` drives every target over a dataset and keeps a snapshot
of what has happened so far. Both front ends - the Textual TUI and the
web dashboard - do nothing but render `snapshot()`, which is why the run
logic is testable without a terminal or a browser.

Week 3 and 4 additions:

* a run can take the distributed path (`distributed=True`), so the
  snapshot carries per-device rows and the balance report;
* every match is scored by `src/scoring.py`, so the snapshot carries
  severity tiers and a ranked "worst hits" list rather than raw hits;
* a run can be stopped from another thread, which the web UI needs.
"""

import threading
import time

from src.chunking import DEFAULT_CHUNK_SIZE, chunk_summary, count_dataset_chunks
from src.pipeline import run_chunked_alignment, sequence_index
from src.scoring import rank_matches, score_match, severity_counts

MAX_TRACKED_MATCHES = 500

# How many ranked hits the UIs show by default.
TOP_MATCHES = 50


def format_bases(count):
    """Human-readable base-pair count (1_500_000 -> '1.50 Mbp')."""
    count = int(count)

    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f} Gbp"

    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} Mbp"

    if count >= 1_000:
        return f"{count / 1_000:.2f} kbp"

    return f"{count} bp"


def format_seconds(value):
    value = float(value)

    if value < 60:
        return f"{value:.2f}s"

    minutes, seconds = divmod(value, 60)

    return f"{int(minutes)}m {seconds:04.1f}s"


def device_status(backend):
    """Live device information for the dashboard panel.

    Returns VRAM use and SM count on a real CUDA device, and a plain
    description of the active fallback backend otherwise.
    """
    status = {"backend": backend, "memory": None, "cores": None, "name": None}

    try:
        from numba import cuda

        if not cuda.is_available():
            return status

        device = cuda.get_current_device()
        free, total = cuda.current_context().get_memory_info()

        name = device.name

        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")

        status["name"] = name
        status["cores"] = int(getattr(device, "MULTIPROCESSOR_COUNT", 0))
        status["memory"] = {
            "free": int(free),
            "total": int(total),
            "used": int(total - free),
        }
    except Exception:
        return status

    return status


class AlignmentRun:
    """Runs every target over a dataset and tracks progress as it goes.

    The run happens on a background thread so a UI can stay responsive;
    `snapshot()` is safe to call from another thread at any time.
    """

    def __init__(self, dataset, targets, mode="gpu", max_mismatches=2,
                 chunk_size=DEFAULT_CHUNK_SIZE, source="", distributed=False,
                 kernel="auto", scheduler="auto"):
        self.dataset = dataset
        self.targets = list(targets)
        self.mode = mode
        self.max_mismatches = max_mismatches
        self.chunk_size = chunk_size
        self.source = source
        self.distributed = bool(distributed)
        self.kernel = kernel
        self.scheduler = scheduler

        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._callback = None
        self._sequences = sequence_index(dataset)

        self.plan = chunk_summary(
            dataset,
            chunk_size=chunk_size,
            target_length=len(self.targets[0]) if self.targets else 0,
        )

        # Total work for the whole run, known up front so the progress bar
        # rises once instead of resetting at every target.
        self.chunks_total = sum(
            count_dataset_chunks(
                dataset,
                chunk_size=chunk_size,
                target_length=len(target),
            )
            for target in self.targets
        )

        self.backend = mode
        self.kernel_name = kernel
        self.scheduler_name = "serial"
        self.devices = []
        self.balance = {}
        self.target_index = 0
        self.current_target = self.targets[0] if self.targets else ""
        self.chunks_done = 0
        self.bases_done = 0
        self.bases_total = self.plan["total_bases"] * max(len(self.targets), 1)
        self.matches_found = 0
        self.recent_matches = []
        self.severity = severity_counts([])
        self.results = []
        self.elapsed = 0.0
        self.started = False
        self.finished = False
        self.stopped = False
        self.error = None

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _score(self, match):
        """Give one hit its PAM, score and severity tier."""
        return score_match(match, self._sequences.get(match.get("sequence_id")))

    def _record(self, matches):
        """Fold a finished target's hits into the tracked state."""
        scored = [self._score(match) for match in matches]

        self.recent_matches = rank_matches(
            self.recent_matches + scored)[:MAX_TRACKED_MATCHES]
        self.severity = severity_counts(self.recent_matches)

        return scored

    # ------------------------------------------------------------------
    # progress accounting
    # ------------------------------------------------------------------

    def _on_chunk(self, status):
        with self._lock:
            self.backend = status["backend"]
            self.chunks_done = self._completed_chunks + status["chunks_done"]
            self.bases_done = self._completed_bases + status["bases_done"]
            self.matches_found = self._completed_matches + status["matches"]
            self.elapsed = time.perf_counter() - self._start_time

            if "scheduler" in status:
                self.scheduler_name = status["scheduler"]

        if self._callback is not None:
            self._callback(self.snapshot())

    def snapshot(self, top=TOP_MATCHES):
        """Consistent view of the run, safe to read from another thread."""
        with self._lock:
            chunk_fraction = (
                self.chunks_done / self.chunks_total
                if self.chunks_total else 0.0
            )
            throughput = (
                self.bases_done / self.elapsed if self.elapsed > 0 else 0.0
            )
            remaining = max(0, self.chunks_total - self.chunks_done)
            eta = (
                remaining * (self.elapsed / self.chunks_done)
                if self.chunks_done else 0.0
            )

            return {
                "source": self.source,
                "mode": self.mode,
                "backend": self.backend,
                "kernel": self.kernel_name,
                "scheduler": self.scheduler_name,
                "distributed": self.distributed,
                "devices": list(self.devices),
                "balance": dict(self.balance),
                "sequences": self.plan["sequences"],
                "chunk_size": self.chunk_size,
                "overlap": self.plan["overlap"],
                "target": self.current_target,
                "target_index": self.target_index,
                "targets_total": len(self.targets),
                "chunks_done": self.chunks_done,
                "chunks_total": self.chunks_total,
                "progress": 100.0 * chunk_fraction,
                "bases_done": self.bases_done,
                "bases_total": self.bases_total,
                "matches": self.matches_found,
                "severity": dict(self.severity),
                "recent_matches": self.recent_matches[:top],
                "elapsed": self.elapsed,
                "eta": eta,
                "throughput": throughput,
                "started": self.started,
                "finished": self.finished,
                "stopped": self.stopped,
                "error": self.error,
            }

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def _align(self, target):
        """Dispatch one target to the serial or the distributed path."""
        if self.distributed:
            from src.distributed import run_distributed_alignment

            return run_distributed_alignment(
                self.dataset,
                target,
                mode=self.mode,
                max_mismatches=self.max_mismatches,
                chunk_size=self.chunk_size,
                progress=self._on_chunk,
                scheduler=self.scheduler,
                kernel=self.kernel,
                stop_event=self._stop,
            )

        return run_chunked_alignment(
            self.dataset,
            target,
            mode=self.mode,
            max_mismatches=self.max_mismatches,
            chunk_size=self.chunk_size,
            progress=self._on_chunk,
            kernel=self.kernel,
            stop_event=self._stop,
        )

    def run(self, callback=None):
        """Run every target in the current thread."""
        self._callback = callback
        self._start_time = time.perf_counter()
        self._completed_chunks = 0
        self._completed_bases = 0
        self._completed_matches = 0

        with self._lock:
            self.started = True

        try:
            for index, target in enumerate(self.targets):
                if self._stop.is_set():
                    break

                with self._lock:
                    self.target_index = index
                    self.current_target = target

                result = self._align(target)

                with self._lock:
                    self.results.append(result)
                    self._completed_chunks += result["chunks"]
                    self._completed_bases += result["bases"]
                    self._completed_matches += len(result["matches"])
                    self.matches_found = self._completed_matches
                    self.kernel_name = result.get("kernel", self.kernel_name)

                    if result.get("scheduler"):
                        self.scheduler_name = result["scheduler"]

                    if result.get("devices"):
                        self.devices = result["devices"]

                    if result.get("balance"):
                        self.balance = result["balance"]

                    self._record(result["matches"])
        except Exception as error:  # surfaced in the UI instead of a crash
            with self._lock:
                self.error = str(error)
            raise
        finally:
            with self._lock:
                self.elapsed = time.perf_counter() - self._start_time
                self.finished = True
                self.stopped = self._stop.is_set()

            if callback is not None:
                callback(self.snapshot())

        return self.results

    def start(self, callback=None):
        """Run every target on a background thread."""
        self._thread = threading.Thread(
            target=self.run,
            kwargs={"callback": callback},
            daemon=True,
        )
        self._thread.start()

        return self._thread

    def stop(self):
        """Ask the run to finish early; workers notice at chunk boundaries."""
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def all_matches(self):
        matches = []

        for result in self.results:
            matches.extend(result["matches"])

        return matches

    def scored_matches(self):
        """Every hit, scored and ranked worst-first."""
        return rank_matches([self._score(m) for m in self.all_matches()])
