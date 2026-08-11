def calculate_mismatches(sequence, target):
    mismatches = 0

    for i in range(len(target)):
        if sequence[i] != target[i]:
            mismatches += 1

    return mismatches


def find_matches_with_mismatches(genome, target, max_mismatches=2):
    matches = []

    genome_length = len(genome)
    target_length = len(target)

    for position in range(genome_length - target_length + 1):

        current_sequence = genome[
            position:position + target_length
        ]

        mismatches = calculate_mismatches(
            current_sequence,
            target
        )

        if mismatches <= max_mismatches:
            matches.append({
                "position": position,
                "sequence": current_sequence,
                "mismatches": mismatches
            })

    return matches