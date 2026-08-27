def extract_pam(genome: str, match_position: int, target_length: int = 20) -> str:
    """
    Extracts the 3-bp Protospacer Adjacent Motif (PAM) sequence immediately
    adjacent to the 3' end of the matched CRISPR target site.
    """
    pam_start = match_position + target_length
    pam_end = pam_start + 3

    if pam_end <= len(genome):
        return genome[pam_start:pam_end].upper()
    elif pam_start < len(genome):
        # Partial PAM near chromosome boundary
        return genome[pam_start:].upper()
    else:
        return "NNN"


if __name__ == "__main__":
    # Test PAM sequence extraction
    sample_genome = "ATGCGATCGATCGATCGATC" + "CGG" + "AATTCC"
    match_pos = 0
    target_len = 20
    extracted_pam = extract_pam(sample_genome, match_pos, target_len)
    print(f"Extracted PAM at position {match_pos + target_len}: {extracted_pam}")
    assert extracted_pam == "CGG", f"Expected 'CGG', got '{extracted_pam}'"
    print("PAM sequence extraction verified successfully!")
