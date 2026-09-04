# GeneWeaver
GeneWeaver is a GPU-accelerated CRISPR DNA sequence alignment tool designed to identify potential off-target mutations in large genome sequences. It uses Python, BioPython, Numba (CUDA), Dask, and Textual to process DNA data efficiently, and presents the results through either a terminal dashboard or a web dashboard.

Week 1 established a CPU baseline and the BioPython data pipeline. Week 2 moved the alignment onto CUDA and added chunking plus the TUI. Week 3 spread the work across every available GPU with Dask and added the biological scoring that turns an alignment into a ranked risk assessment. Week 4 cut the kernel's memory traffic with CUDA shared memory and polished both front ends around the base pairing itself.

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

## Week 3 - Distributed scaling

`src/distributed.py` puts Dask in front of the chunk stream so a second
GPU is not left idle.

The scheduling problem is that chunks are **not uniform**: the last chunk
of every FASTA record is a remainder and can be a few hundred bases next
to a neighbour's million. Handing chunks out round-robin balances chunk
*counts* while leaving the base counts lopsided, so assignment is instead
greedy longest-processing-time-first - every chunk goes to whichever
device currently holds the fewest bases.

```bash
python main.py --mode gpu --distributed --target data/guides.csv
```

```
Scheduler: dask | devices: 2
...
Device split:
  gpu0 NVIDIA RTX A4000: 634 chunks, 272478 bases (49.99%), 1.204s
  gpu1 NVIDIA RTX A4000: 651 chunks, 272589 bases (50.01%), 1.198s
  load imbalance: 0.041%
```

Note the chunk counts differ while the base counts do not - that is the
balancer doing its job. Every layer degrades instead of failing: no Dask
installed falls back to a `ThreadPoolExecutor` running the same
partitions, and no CUDA falls back to the numpy backend, so
`--distributed` is testable on a laptop. Results are sorted back into
file order before returning, so a distributed run and a serial run
produce byte-identical CSVs - `benchmark.py --distributed` asserts it.

## Week 3 - Biological scoring

An alignment says *where* a guide could bind. It does not say whether
that binding matters. `src/scoring.py` scores every hit with the MIT
(Hsu et al. 2013) specificity score, so hits can be ranked by how likely
Cas9 actually is to cut there:

- **PAM dependence.** SpCas9 cannot cut without a 5'-NGG-3' PAM
  immediately 3' of the protospacer, so a flawless 20/20 alignment with
  no PAM is capped at `low` severity.
- **PAM-proximal seed.** The 12 nt closest to the PAM are the seed. A
  mismatch there usually abolishes cleavage; the same mismatch at the
  distal 5' end is often tolerated completely.
- **Substitution identity.** Transitions (A<->G, C<->T) form wobble
  pairs and are tolerated far better than transversions.

```
Off-target severity
--------------------------------------------------
 critical: 27
     high: 0
 moderate: 0
      low: 13

Most dangerous off-targets
--------------------------------------------------
 severity    score  sequence           pos  mm seed  pam  pairing
 critical   100.00  sequence_37         35   0    0  AGG  ....................
      low     4.82  sequence_394        70   2    1  AAA  ...x.........x......
```

The score runs 0-100 and reads as "how much on-target cutting activity
survives", so a **high score is a dangerous off-target**. A single
mismatch at position 1 still scores 100; the same mismatch at position 20
scores 33; two in the seed scores 0.14.

The hand-written `data/targets.csv` from Week 1 mostly has no PAM, so
every hit against it is correctly scored `low`. To exercise the scoring
on realistic guides, generate a guide list whose sites really are
followed by NGG:

```bash
python scripts/pick_guides.py --count 5      # writes data/guides.csv
```

## Week 4 - CUDA shared memory

The Week 2 kernel gives one thread one alignment position, and each
thread reads `target_length` bases straight out of global VRAM. With a
20 nt guide, thread `p` and thread `p+1` overlap on 19 of their 20 reads,
so **every base crosses the memory bus about 20 times**. That traffic,
not arithmetic, is what the kernel spends its time waiting on.

`alignment_kernel_shared` stages the work on-chip instead. Each block
cooperatively copies the guide and its tile of sequence
(`blockDim.x + target_length - 1` bases, because the last thread's window
runs past the last thread's start position) into shared memory,
synchronises, then does every comparison against on-chip memory:

```
Global reads per base: 20 (naive) -> 1.074 (shared) = 18.62x less VRAM traffic
```

Shared memory must be sized at compile time, which fixes the block width
at 256 threads and the guide limit at 64 nt; longer guides fall back to
the Week 2 kernel automatically. `src/kernel_config.py` holds that launch
geometry in a numba-free module so the CLI and both dashboards can reason
about kernel choice on a machine with no CUDA installed.

```bash
python main.py --mode gpu --kernel shared     # Week 4 kernel
python main.py --mode gpu --kernel naive      # Week 2 kernel
python benchmark.py --kernels --distributed   # time all of them
```

A faster kernel that returns different answers is not an optimisation, so
the shared-memory kernel is checked against both the Week 2 kernel and a
numpy reference - on block boundaries, partial final blocks and guides
whose halo is larger than a warp:

```bash
python scripts/verify_kernels.py                       # on a CUDA machine
NUMBA_ENABLE_CUDASIM=1 python scripts/verify_kernels.py  # anywhere
```

The simulator form runs the kernels in Python, which is slow but
exercises the same indexing, the same halo arithmetic and the same
`__syncthreads()` placement, so the tiling is genuinely under test with
no GPU present.

## Week 4 - The dashboards

Both front ends render the same `AlignmentRun.snapshot()` dict from
`src/run_state.py`, so no analysis logic is duplicated between them.

A progress bar and a list of hits is a monitor, not an instrument. What
you need in order to judge an off-target is the base pairing itself, so
both dashboards put an alignment inspector under the results table:
select any hit and the guide is drawn over the genomic site with **every
mismatched base in red**, the seed region marked, the PAM called out, and
the MIT score alongside.

```
  guide   5'-AAGGCAAAAGCAAGAAATGG-3'  PAM ATG (no NGG - Cas9 cannot cut)
  pairing    x||||||||||||||x|||xx
  site    5'-CAGGCAAAAGCAAGGAATCT-3'
  seed       --------^^^^^^^^^^^^  4 mismatch(es), 3 in the seed
```

### Terminal

```bash
python -m src.dashboard --target data/guides.csv --distributed
```

Severity-coloured rows, a severity tally, live throughput and ETA, and a
device panel showing the scheduler and the per-GPU split. `s` stops a run
early, `h` explains the scoring, `q` quits.

### Web

The TUI is the right tool at a workstation with the data on local disk.
It is the wrong tool when the GPUs live in a server room and the person
who needs the report is a biologist with a laptop. `web/server.py`
exposes the same engine over HTTP:

```bash
python -m web.server --open           # http://127.0.0.1:8000
python -m web.server --host 0.0.0.0 --port 9000
```

The page has the scan setup on the left (dataset, guide list, backend,
kernel, chunk size, distributed toggle), and live progress, per-device
balance bars, the severity breakdown, the alignment inspector and a
sortable ranked table on the right. Results export to CSV with the score,
severity, seed-mismatch count and PAM included.

This deliberately uses **nothing but the standard library**. Adding Flask
or FastAPI to a project whose point is GPU throughput would put a web
framework, an ASGI server and their dependency trees between the user and
a `python` command that already works; `http.server` with a threading
mixin is adequate for a handful of analysts and runs anywhere the
pipeline already runs.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | datasets, guide files, cluster and device info |
| `GET /api/scoring` | the position-weight matrix |
| `POST /api/run` | start a scan |
| `GET /api/status` | live snapshot, polled by the page |
| `POST /api/stop` | stop the active scan |
| `GET /api/matches.csv` | scored hits of a finished scan |

Client-supplied paths are resolved against the project root and rejected
if they escape it, static serving is confined to `web/static`, and one
scan runs at a time - a second request gets a 409 rather than queueing
onto the same GPUs.

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

Both scripts accept `--input`, `--format`, `--target`, `--limit`,
`--chunk-size`, `--max-mismatches`, `--distributed` and `--kernel`;
`main.py` adds `--output`, `--no-export`, `--stats`, `--no-score` and
`--top`, and `benchmark.py` adds `--kernels`.

```bash
python main.py --mode gpu --limit 0 --max-mismatches 3      # whole genome.fasta
python main.py --input data/human_sequences.csv --stats     # CSV run with class stats
python main.py --target data/guides.csv --distributed --top 20
python main.py --no-score                                   # raw hits, Week 2 output
python benchmark.py --limit 500 --distributed --kernels
```

## Layout

```
main.py                 CLI scan, scored and exported
benchmark.py            CPU vs GPU vs distributed vs kernel timings
scripts/pick_guides.py  pull PAM-adjacent guides out of a genome
scripts/verify_kernels.py  check the shared kernel against the baseline
src/parser.py           FASTA / CSV loading (BioPython, with a fallback)
src/chunking.py         record-safe, overlap-correct chunking
src/cpu_alignment.py    Week 1 pure-Python baseline
src/cuda_kernels.py     both CUDA kernels
src/kernel_config.py    launch geometry, importable without numba
src/gpu_alignment.py    backend dispatch and kernel selection
src/gpu_memory.py       host <-> device transfers
src/distributed.py      Week 3 Dask scheduling and device balancing
src/scoring.py          Week 3 MIT/PAM-weighted off-target scoring
src/pipeline.py         chunked alignment, serial path
src/run_state.py        run orchestration and the snapshot both UIs render
src/dashboard.py        Textual dashboard
web/server.py           stdlib HTTP API
web/static/             the web dashboard
```
