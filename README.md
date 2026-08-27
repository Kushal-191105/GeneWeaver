# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

**Domain:** Bioinformatics & High-Performance Computing (HPC)  
**Technology Stack:** Python, BioPython, Numba (CUDA JIT), Dask Distributed, NumPy, Textual TUI

---

## Overview
GeneWeaver is a high-performance bioinformatics engine designed to rapidly search human genome sequences (millions/billions of base pairs) for unintended CRISPR "off-target" mutations. By bypassing Python's Global Interpreter Lock (GIL) and offloading sequence alignments directly to NVIDIA CUDA GPU hardware via custom JIT-compiled kernels coordinated through Dask distributed scheduling, GeneWeaver achieves massive speedups compared to traditional CPU implementations.

---

## Architecture Evolution

### Week 1: Data Pipeline & CPU Baseline
- Ingestion of 4,380 raw human sequences (5.53 million base pairs) via BioPython.
- DNA sequence quality cleansing and nucleotide distribution analysis.
- Genome chunking into 1,000 bp blocks with lossless reconstruction validation.
- Pure Python sliding-window Hamming distance alignment algorithm to establish baseline metrics.

### Week 2: CUDA GPU Acceleration & TUI Scaffolding
- Hardware introspection (NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM, 20 SMs, CC 8.6).
- Contiguous `uint8` ASCII byte encoding for zero-overhead DMA Host-to-Device (H2D) transfers.
- Custom `@cuda.jit` exact match and mismatch-tolerant alignment kernels with early bounds termination.
- 100% mathematical parity verification between CPU baseline and GPU hardware results.
- **Over 220x total speedup** and **>780x pure kernel speedup**.
- Interactive Terminal User Interface (TUI) built with `Textual`.

### Week 3: Distributed Scaling (Dask) & Biological Scoring Matrix
- **Distributed Dask Coordination:** Partitioning genomic sequences into balanced batches across parallel worker processes with boundary overlap handling (`target_len + 3` bp).
- **GPU Worker Device Affinity:** Automatic mapping of Dask workers to available CUDA devices (`cuda.select_device(worker_id)`).
- **CRISPR-Cas9 Biological Scoring Matrix:**
  - **PAM Recognition:** Extraction and classification of adjacent 3-bp Protospacer Adjacent Motifs (Canonical `NGG`, Non-canonical `NAG`, and Non-viable).
  - **PAM-Proximity Weighting:** Position-weight matrix penalizing mismatches in the critical proximal 3' seed region (positions 11–20) while capturing tolerated distal mismatches (positions 1–10).
  - **Severity Scoring:** Biological cutting probability score ($0.0 - 100.0\%$).
  - **Risk Stratification:** Categorization into **High Risk** ($\ge 60\%$), **Medium Risk** ($20\% - 59\%$), and **Low Risk** ($< 20\%$).
  - **Off-Target Ranking:** Automated sorting of candidate sites by biological cleavage severity.
- **Enhanced TUI Dashboard:** Multi-worker progress bars, Dask cluster status cards, and color-coded risk badges.

---

## Installation

1. Activate your Python virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirement.txt
```

---

## Usage Guide

### 1. Run Complete Week 3 Distributed Pipeline
```powershell
python main.py
```

### 2. Launch Interactive Terminal UI (Textual Dashboard)
```powershell
python main.py --tui
# or
python src/tui.py
```
*(Inside the TUI: Press **R** for Single-GPU, **D** for Distributed Dask, **B** for Benchmark, or **Q** to Quit)*

### 3. Run Benchmark Suite (CPU vs GPU vs Dask Distributed)
```powershell
python benchmark.py
```

### 4. Run 3-Way Parity Validation (CPU vs GPU vs Dask)
```powershell
python validate_gpu.py
```

### 5. Run Biological Scoring Test Suite
```powershell
python test_scoring.py
```

### 6. Inspect GPU Hardware Status
```powershell
python gpu_check.py
```
