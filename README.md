# GeneWeaver
GeneWeaver is a GPU-accelerated CRISPR DNA sequence alignment tool designed to identify potential off-target mutations in large genome sequences. It uses Python, BioPython, Numba (CUDA), Dask, and Textual to process DNA data efficiently and display alignment results through a terminal dashboard.

## Input data

Alignment runs against the **human DNA sequence dataset** `data/human_sequences.csv` (4380 sequences, ~5.5 M bases) — the only sequence input the pipeline uses.

Columns: `sequence_id`, `sequence`, `class`. Only `sequence` is required — ids are generated if missing, and CSV or TSV both work (separator auto-detected).

Targets live in `data/target.txt`, one target per line; `#` lines are ignored.

Runs are limited to the first 200 sequences by default so they finish quickly; `--limit 0` uses all 4380.

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
Dataset: data/human_sequences.csv
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
Dataset: data/human_sequences.csv
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

Both scripts accept `--input`, `--target`, `--limit` and `--max-mismatches`; `main.py` adds `--output`, `--no-export` and `--stats`.

```bash
python main.py --mode gpu --limit 0 --max-mismatches 3   # whole human dataset
python benchmark.py --limit 500
```
