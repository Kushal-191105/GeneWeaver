import math
import numpy as np

# Ensure NumPy 2.x compatibility with Numba CUDA
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

from numba import cuda, uint8


def dna_to_gpu_array(sequence: str) -> np.ndarray:
    """
    Converts a DNA nucleotide string into a contiguous uint8 NumPy array
    representing ASCII byte values for efficient GPU memory transfer.
    """
    if isinstance(sequence, str):
        cleaned_seq = sequence.strip().upper().encode("ascii")
    elif isinstance(sequence, bytes):
        cleaned_seq = sequence.strip().upper()
    else:
        raise ValueError("Sequence must be a string or bytes.")

    return np.ascontiguousarray(np.frombuffer(cleaned_seq, dtype=np.uint8))


def gpu_array_to_dna(array: np.ndarray) -> str:
    """
    Converts a uint8 NumPy array back to a DNA nucleotide string.
    """
    return bytes(array).decode("ascii")


def transfer_target_to_gpu(target: str):
    """
    Transfers a CRISPR target sequence from Host (CPU RAM) to Device (GPU VRAM).
    Returns (d_target, target_length).
    """
    target_arr = dna_to_gpu_array(target)
    d_target = cuda.to_device(target_arr)
    return d_target, len(target_arr)


def transfer_genome_to_gpu(genome: str):
    """
    Transfers a full genomic sequence from Host (CPU RAM) to Device (GPU VRAM).
    Returns (d_genome, genome_length).
    """
    genome_arr = dna_to_gpu_array(genome)
    d_genome = cuda.to_device(genome_arr)
    return d_genome, len(genome_arr)


@cuda.jit
def cuda_exact_match_kernel(genome, target, match_flags, total_positions, target_len):
    """
    CUDA Kernel: Checks exact string matching in parallel across GPU threads.
    """
    pos = cuda.grid(1)
    if pos < total_positions:
        is_match = 1
        for i in range(target_len):
            if genome[pos + i] != target[i]:
                is_match = 0
                break
        match_flags[pos] = is_match


@cuda.jit
def cuda_mismatch_count_kernel(genome, target, mismatch_counts, total_positions, target_len, max_mismatches):
    """
    CUDA Kernel: Counts base pair mismatches for CRISPR off-target identification (Global Memory).
    """
    pos = cuda.grid(1)
    if pos < total_positions:
        mismatches = 0
        for i in range(target_len):
            if genome[pos + i] != target[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            mismatch_counts[pos] = mismatches
        else:
            mismatch_counts[pos] = 255


@cuda.jit
def cuda_shared_mem_alignment_kernel(genome, target, mismatch_counts, total_positions, target_len, max_mismatches, genome_len):
    """
    Optimized CUDA Kernel: Dual Shared Memory Architecture (Target Buffer + Genomic Tile Cache).
    - Cooperative loading of target sequence into on-chip SRAM (s_target).
    - Cooperative loading of genomic block window + halo into on-chip SRAM (s_genome).
    - Block barrier cuda.syncthreads() ensures SRAM data integrity.
    - Zero global memory access during the 20-bp sliding window comparisons.
    """
    s_target = cuda.shared.array(32, dtype=uint8)
    s_genome = cuda.shared.array(320, dtype=uint8)

    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    block_start = cuda.blockIdx.x * bdim

    # 1. Cooperative Target Load
    if tid < target_len:
        s_target[tid] = target[tid]

    # 2. Cooperative Main Tile Load
    global_idx = block_start + tid
    if global_idx < genome_len:
        s_genome[tid] = genome[global_idx]
    else:
        s_genome[tid] = 0

    # 3. Cooperative Halo Boundary Load
    halo_len = target_len - 1
    if tid < halo_len:
        halo_global_idx = block_start + bdim + tid
        if halo_global_idx < genome_len:
            s_genome[bdim + tid] = genome[halo_global_idx]
        else:
            s_genome[bdim + tid] = 0

    # 4. Synchronization Barrier
    cuda.syncthreads()

    # 5. Ultra-Fast SRAM Sliding Window Comparison
    pos = block_start + tid
    if pos < total_positions:
        mismatches = 0
        for i in range(target_len):
            if s_genome[tid + i] != s_target[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            mismatch_counts[pos] = mismatches
        else:
            mismatch_counts[pos] = 255


def gpu_exact_match(genome: str, target: str, threads_per_block: int = 256):
    """
    Performs exact DNA sequence alignment on the GPU.
    """
    genome_len = len(genome)
    target_len = len(target)
    total_positions = genome_len - target_len + 1

    if total_positions <= 0:
        return []

    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_match_flags = cuda.device_array(total_positions, dtype=np.uint8)

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_exact_match_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_match_flags, total_positions, target_len
    )
    cuda.synchronize()

    h_match_flags = d_match_flags.copy_to_host()
    return np.where(h_match_flags == 1)[0].tolist()


def gpu_count_mismatches_global(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Executes baseline global-memory mismatch counting on GPU.
    """
    genome_len = len(genome)
    target_len = len(target)
    total_positions = genome_len - target_len + 1

    if total_positions <= 0:
        return np.array([], dtype=np.uint8)

    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_mismatch_counts = cuda.device_array(total_positions, dtype=np.uint8)

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_mismatch_count_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_mismatch_counts, total_positions, target_len, max_mismatches
    )
    cuda.synchronize()

    return d_mismatch_counts.copy_to_host()


def gpu_count_mismatches_shared_mem(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Executes dual shared-memory optimized mismatch counting on GPU.
    """
    genome_len = len(genome)
    target_len = len(target)
    total_positions = genome_len - target_len + 1

    if total_positions <= 0:
        return np.array([], dtype=np.uint8)

    d_genome, _ = transfer_genome_to_gpu(genome)
    d_target, _ = transfer_target_to_gpu(target)
    d_mismatch_counts = cuda.device_array(total_positions, dtype=np.uint8)

    blocks_per_grid = math.ceil(total_positions / threads_per_block)
    cuda_shared_mem_alignment_kernel[blocks_per_grid, threads_per_block](
        d_genome, d_target, d_mismatch_counts, total_positions, target_len, max_mismatches, genome_len
    )
    cuda.synchronize()

    return d_mismatch_counts.copy_to_host()


def gpu_find_matches_with_mismatches(genome: str, target: str, max_mismatches: int = 2, threads_per_block: int = 256):
    """
    Orchestrates GPU alignment using on-chip Shared Memory and collects structured results.
    """
    target_len = len(target)
    mismatch_array = gpu_count_mismatches_shared_mem(genome, target, max_mismatches=max_mismatches, threads_per_block=threads_per_block)

    valid_positions = np.where(mismatch_array <= max_mismatches)[0]
    matches = []

    for pos in valid_positions:
        pos_int = int(pos)
        curr_seq = genome[pos_int:pos_int + target_len]
        mismatch_count = int(mismatch_array[pos_int])
        mismatch_positions = [i for i in range(target_len) if curr_seq[i] != target[i]]

        matches.append({
            "position": pos_int,
            "sequence": curr_seq,
            "mismatches": mismatch_count,
            "mismatch_positions": mismatch_positions
        })

    return matches


if __name__ == "__main__":
    test_genome = "ATGCCCCAACTAAATACTAC" + "TGG" + "GATC" * 50
    test_target = "ATGCCCCAACTAAATACTAC"
    res = gpu_find_matches_with_mismatches(test_genome, test_target, max_mismatches=2)
    print(f"Shared Memory Kernel found {len(res)} matches:")
    for r in res:
        print(f"  Pos: {r['position']} | Seq: {r['sequence']} | Mismatches: {r['mismatches']}")
    assert len(res) > 0
    print("CUDA Shared Memory Mismatch Kernel verified successfully!")
