# GeneWeaver
GeneWeaver is a GPU-accelerated CRISPR DNA sequence alignment tool designed to identify potential off-target mutations in large genome sequences. It uses Python, BioPython, Numba (CUDA), Dask, and Textual to process DNA data efficiently and display alignment results through a terminal dashboard.

## Input data

The pipeline reads two interchangeable sequence formats and the format is detected automatically from the file, so the same commands work either way.

**Primary dataset — FASTA:** `data/genome.fasta` (4380 sequences, ~5.5 M bases). This is the default input. Sequence ids come from the FASTA headers (`>sequence_0`), multi-line records and lowercase bases are handled, and BioPython's `SeqIO` is used when installed with a built-in reader as fallback.

**Alternative dataset — CSV/TSV:** `data/human_sequences.csv`, the same 4380 sequences with columns `sequence_id`, `sequence`, `class`. Only `sequence` is required — ids are generated if missing, and the separator is auto-detected. The `class` column is what `--stats` uses for the class distribution, so that section only appears for CSV/TSV runs; FASTA carries no labels.

```bash
python main.py                                     # FASTA (default)
python main.py --input data/human_sequences.csv    # CSV
python main.py --input data/genome.fasta --format fasta   # skip detection
```

Regenerate the FASTA from the CSV at any time:

```bash
python -m src.parser --input data/human_sequences.csv --output data/genome.fasta
```

**Chromosome-scale input.** For a genuinely massive run, pull a real chromosome from Ensembl — gzipped FASTA is read directly, no unpacking needed:

```bash
./scripts/download_chromosome.sh 21          # ~46.7 Mbp, ~15 MB gzipped
python main.py --input data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz --limit 0
```

Targets live in `data/targets.csv`, one target per row under a `target` column; blank values and `#` lines are ignored.

Runs are limited to the first 200 sequences by default so they finish quickly; `--limit 0` uses all 4380.

## Chunking

A chromosome does not fit comfortably in GPU VRAM, so `src/chunking.py` slices every record into fixed-size `numpy` arrays (1 Mbp by default, `--chunk-size` to change it) and the pipeline streams them through the kernels one at a time.

Two rules keep this scientifically correct:

- **Chunks never span two records.** Joining separate sequences end to end would invent matches that do not exist in the biology.
- **Consecutive chunks of the same record overlap by `target_length - 1` bases**, so a match straddling a chunk boundary is still fully inside one chunk. Each chunk only reports the first `chunk_size - (target_length - 1)` positions it covers, so the overlap never reports the same match twice.

Chunked results are identical to unchunked results — verified against matches planted deliberately across chunk boundaries.

## TUI dashboard

```bash
python -m src.dashboard                                  # FASTA default, 200 sequences
python -m src.dashboard --limit 0 --chunk-size 2000000    # whole dataset
```

Live view of chunking progress, throughput, backend and device VRAM, plus off-target matches as they are found. The run happens on a background thread so the interface stays responsive; the run logic itself lives in `src/run_state.py` (`AlignmentRun`) so it can be tested without a terminal.

## How to run Week 2

CPU:

```bash
python main.py --mode cpu
```

GPU (NVIDIA/CUDA machine; falls back to the vectorized NumPy backend elsewhere):

```bash
python main.py --mode gpu
```

Output:

```
GeneWeaver - CPU Alignment
==================================================
Dataset: data/genome.fasta
Format: FASTA
Sequences: 200
Total bases: 264902
Targets: 5

Target: ATGCCCCAACTAAATACTAC
Target length: 20
Alignment positions: 261102
Matches found: 1
CPU time: 0.310114 seconds
...
```

All matches are written to `results/matches.csv` (`sequence_id, target, position, sequence, mismatches`).

Benchmark CPU vs GPU:

```bash
python benchmark.py
```

```
GeneWeaver Performance Benchmark
==================================================
Dataset: data/genome.fasta
Format: FASTA
Sequences: 200
Targets: 5

CPU
--------------------------------------------------
Alignment positions: 1305510
Execution time: 1.572752 seconds

GPU
--------------------------------------------------
Alignment positions: 1305510
Execution time: 0.349657 seconds

Performance
--------------------------------------------------
GPU speedup: 4.50x
Result check: PASSED (CPU and GPU agree)
```

### Useful options

Both scripts accept `--input`, `--format`, `--target`, `--limit` and `--max-mismatches`; `main.py` adds `--output`, `--no-export` and `--stats`.

```bash
python main.py --mode gpu --limit 0 --max-mismatches 3   # whole genome.fasta
python main.py --input data/human_sequences.csv --stats  # CSV run with class stats
python benchmark.py --limit 500
```
