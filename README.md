# GeneWeaver

GeneWeaver is a GPU-accelerated CRISPR DNA sequence alignment tool designed to identify potential off-target mutations in large genome sequences. It uses Python, BioPython, Numba (CUDA), Dask, and Textual to process DNA data efficiently and display alignment results through a terminal dashboard.

---

## Module 2 — CUDA/Numba GPU Alignment Engine

Module 2 provides high-performance, GPU-accelerated DNA sequence alignment and mismatch detection using Numba CUDA kernels, with automated fallback to CPU execution when an NVIDIA GPU is not available.

### Features
1. **GPU Device Detection**:
   - Automated detection of CUDA drivers and NVIDIA GPU devices.
   - Device querying for compute capability, multiprocessors, and VRAM.
   - Clean `is_cuda_available()` API with non-crashing CPU fallback.

2. **DNA Numeric Encoding**:
   - Converts standard IUPAC DNA strings (`A, C, G, T, N`) into compact 8-bit unsigned integer arrays (`uint8`):
     - `A / a` → `0`
     - `C / c` → `1`
     - `G / g` → `2`
     - `T / t` → `3`
     - `N / n` → `4`
   - High-throughput lookup table (LUT) encoding with project-level character validation.

3. **CUDA Alignment Kernel & Mismatch Counting**:
   - Numba CUDA kernel (`@cuda.jit`) evaluating all candidate start positions in parallel across GPU threads.
   - Parallel mismatch counting against configurable mismatch threshold (`max_mismatches`).
   - Output normalized to identical structure as the CPU matcher (`position`, `sequence`, `mismatches`, `mismatch_positions`, `chunk_id`).

4. **Chunk-Based GPU Processing**:
   - Integrates with existing genome chunker (`create_chunks`).
   - Streams chunks to GPU memory sequentially without allocating full gigabyte genomes on device.
   - Reusable device memory buffers (`GPUMemoryBuffer`) to minimize host-to-device allocation churn.
   - Preserves global genomic position across all chunk boundaries.

5. **CPU Fallback & Multi-Backend Support**:
   - Supports execution backends: `'auto'` (CUDA if available, else CPU), `'cuda'` (GPU only), and `'cpu'` (CPU baseline).

6. **Benchmarking & Performance Measurement**:
   - Records real execution times, speedup factors, data transfer, and kernel durations.
   - Distinguishes Numba JIT compilation warmup from steady-state execution.

---

## Hardware & Software Requirements

- **Python**: 3.9+
- **Packages**: `numba>=0.60.0`, `numpy>=2.0.0`, `biopython>=1.85`, `pandas>=2.2.0`, `pytest>=8.0.0`
- **GPU Acceleration**: NVIDIA GPU with CUDA Compute Capability 5.0+ and compatible NVIDIA CUDA Driver (Optional: runs in CPU fallback mode if unavailable).

---

## Usage Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Running Alignment in Python
```python
from src.gpu import GPUAlignmentEngine, is_cuda_available

# Initialize engine ('auto', 'cuda', or 'cpu')
engine = GPUAlignmentEngine(
    threads_per_block=128,
    chunk_size=100_000,
    backend="auto"
)

genome = "ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCC"
target = "ATGCCCCAACTAAATACTAC"

matches = engine.align(genome, target, max_mismatches=2)

for match in matches:
    print(f"Position: {match['position']}, Mismatches: {match['mismatches']}, Seq: {match['sequence']}")
```

### 3. Running Benchmarks
To run the full CPU vs GPU alignment benchmark:
```bash
python benchmark.py
```

### 4. Running the Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run with Numba CUDA Simulator enabled (for verification on non-NVIDIA systems)
NUMBA_ENABLE_CUDASIM=1 pytest tests/ -v
```
