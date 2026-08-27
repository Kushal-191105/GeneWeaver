# Biological Position-Weight Matrix for SpCas9 (Hsu et al. / CFD Model Principles)
# Positions 0..9 (5' Distal region): Mismatches are tolerated by Cas9 -> lower penalty factor
# Positions 10..19 (3' Seed region, proximal to PAM): Mismatches disrupt cleavage -> high penalty factor
PAM_PROXIMITY_WEIGHTS = {
    # Distal non-seed region (Positions 1 to 10, 0-indexed 0 to 9)
    0: 0.10, 1: 0.12, 2: 0.15, 3: 0.18, 4: 0.20,
    5: 0.22, 6: 0.25, 7: 0.28, 8: 0.32, 9: 0.38,
    # Proximal seed region (Positions 11 to 20, 0-indexed 10 to 19, adjacent to PAM)
    10: 0.55, 11: 0.65, 12: 0.72, 13: 0.80, 14: 0.85,
    15: 0.88, 16: 0.92, 17: 0.95, 18: 0.97, 19: 0.98
}


def get_position_weight(position: int) -> float:
    """
    Returns the mismatch penalty weight for a given position in the 20-bp protospacer.
    Positions near the 3' PAM (seed region) carry high penalties, whereas 5' distal
    positions carry lower penalties.
    """
    return PAM_PROXIMITY_WEIGHTS.get(position, 0.50)


def is_seed_region(position: int) -> bool:
    """
    Returns True if the position is within the critical 3' seed region (positions 10-19).
    """
    return position >= 10


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
        return genome[pam_start:].upper()
    else:
        return "NNN"


def validate_pam(pam_seq: str) -> dict:
    """
    Validates the PAM motif for SpCas9 endonuclease:
    - Canonical PAM (NGG): Full binding & cleavage affinity (factor = 1.0)
    - Non-canonical PAM (NAG): Low affinity / weak cleavage (factor = 0.25)
    - Invalid PAM (other): Non-viable site, cleavage prohibited (factor = 0.0)
    """
    pam = pam_seq.upper().strip()

    if len(pam) == 3 and pam[1:] == "GG":
        return {
            "pam": pam,
            "type": "canonical",
            "is_viable": True,
            "pam_factor": 1.0,
            "description": "Canonical SpCas9 PAM (NGG)"
        }
    elif len(pam) == 3 and pam[1:] == "AG":
        return {
            "pam": pam,
            "type": "non-canonical",
            "is_viable": True,
            "pam_factor": 0.25,
            "description": "Non-canonical SpCas9 PAM (NAG)"
        }
    else:
        return {
            "pam": pam,
            "type": "invalid",
            "is_viable": False,
            "pam_factor": 0.0,
            "description": "Non-viable PAM (cleavage absent)"
        }


if __name__ == "__main__":
    print("Testing biological position weights...")
    for pos in [0, 5, 9, 10, 15, 19]:
        weight = get_position_weight(pos)
        seed = "SEED (Critical)" if is_seed_region(pos) else "DISTAL (Tolerated)"
        print(f"Position {pos + 1:2d} (idx {pos:2d}): Penalty {weight:.2f} [{seed}]")

    assert get_position_weight(0) < get_position_weight(19)
    assert not is_seed_region(4)
    assert is_seed_region(15)
    print("Biological position weights verified successfully!")
