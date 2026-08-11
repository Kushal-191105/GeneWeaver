# CPU alignment module
def align_sequence(genome, target, max_mismatches=2):
    results = []

    genome_length = len(genome)
    target_length = len(target)

    for position in range(genome_length - target_length + 1):

        current_sequence = genome[position:position + target_length]

        mismatches = 0

        for i in range(target_length):
            if genome[position + i] != target[i]:
                mismatches += 1

        if mismatches <= max_mismatches:
            results.append({
                "position": position,
                "sequence": current_sequence,
                "mismatches": mismatches
            })

    return results