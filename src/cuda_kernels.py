from numba import cuda


@cuda.jit
def alignment_kernel(sequence, target, mismatch_counts):
    """
    Compare the target against every possible position
    in the genome sequence.

    One CUDA thread handles one alignment position.
    """

    position = cuda.grid(1)

    target_length = target.size

    if position + target_length > sequence.size:
        return

    mismatch_count = 0

    for i in range(target_length):

        if sequence[position + i] != target[i]:
            mismatch_count += 1

    mismatch_counts[position] = mismatch_count
