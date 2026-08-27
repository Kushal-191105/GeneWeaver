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
    # Test PAM validation
    test_cases = ["CGG", "TGG", "AAG", "CAG", "ATC", "NNN"]
    for p in test_cases:
        res = validate_pam(p)
        print(f"PAM '{p}': {res['type']} (Factor: {res['pam_factor']}) - {res['description']}")

    assert validate_pam("AGG")["type"] == "canonical"
    assert validate_pam("GAG")["type"] == "non-canonical"
    assert validate_pam("ATC")["type"] == "invalid"
    print("PAM motif validation verified successfully!")
