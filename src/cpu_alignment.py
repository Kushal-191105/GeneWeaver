def compare_sequences(sequence, target):
    mismatches = 0
    mismatch_positions = []

    for i in range(len(target)):
        if sequence[i] != target[i]:
            mismatches += 1
            mismatch_positions.append(i)

    return mismatches, mismatch_positions


def find_matches_with_mismatches(genome, target, max_mismatches=2):
    matches = []

    genome_length = len(genome)
    target_length = len(target)

    for position in range(genome_length - target_length + 1):

        current_sequence = genome[
            position:position + target_length
        ]

        mismatches, mismatch_positions = compare_sequences(
            current_sequence,
            target
        )

        if mismatches <= max_mismatches:
            matches.append({
                "position": position,
                "sequence": current_sequence,
                "mismatches": mismatches,
                "mismatch_positions": mismatch_positions
            })

    return matches