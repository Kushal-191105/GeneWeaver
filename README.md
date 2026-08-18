# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

**Domain:** Bioinformatics & High-Performance Computing (HPC)  
**Technology Stack:** Python, BioPython, Numba (CUDA JIT), NumPy, Textual TUI

---

## Overview
GeneWeaver is a high-performance bioinformatics engine designed to rapidly search human genome sequences (millions/billions of base pairs) for unintended CRISPR "off-target" mutations. By bypassing Python's Global Interpreter Lock (GIL) and offloading sequence alignments directly to NVIDIA CUDA GPU hardware via custom JIT-compiled kernels, GeneWeaver achieves massive speedups compared to single-threaded CPU implementations.

---

##  Features & Architecture

- **GPU Device Querying:** Automatic hardware introspection (VRAM, SM count, Compute Capability).
- **Zero-Overhead Memory Transfers:** Contiguous uint8 ASCII byte representation for DMA host-to-device transfers.
- **Custom CUDA Alignment Kernels:**
  - `@cuda.jit` exact match kernel.
  - `@cuda.jit` mismatch-tolerant kernel with early bounds and threshold checking.
- **100% Parity Validation:** Exact matching verification between CPU baseline and GPU hardware results.
- **Performance Benchmarks:** Over **150x to 220x total speedup** (and **>400x kernel speedup**) on NVIDIA GeForce RTX 3050.
- **Interactive Textual TUI:** Real-time terminal dashboard with progress bars, live execution logging, and GPU memory metrics.

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

### 1. Run Complete Pipeline
```powershell
python main.py
```

### 2. Launch Interactive Terminal UI (Textual)
```powershell
python main.py --tui
# or
python src/tui.py
```

### 3. Run CPU vs GPU Benchmarks
```powershell
python benchmark.py
```

### 4. Run Parity Validation
```powershell
python validate_gpu.py
```

### 5. Check GPU Hardware Status
```powershell
python gpu_check.py
```
