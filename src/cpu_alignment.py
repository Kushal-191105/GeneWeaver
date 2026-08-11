def find_exact_matches(genome, target):
    matches = []

    genome_length = len(genome)
    target_length = len(target)

    for position in range(genome_length - target_length + 1):

        current_sequence = genome[
            position:position + target_length
        ]

        if current_sequence == target:
            matches.append({
                "position": position,
                "sequence": current_sequence
            })

    return matches