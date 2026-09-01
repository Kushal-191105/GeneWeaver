import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def format_visual_alignment(target: str, match_seq: str, pam: str = "", mismatch_positions: list = None, use_rich: bool = True) -> dict:
    """
    Constructs a visual representation of the DNA sequence alignment:
    - Matching base pairs are rendered in Green/Cyan.
    - Mutated / Mismatched base pairs are highlighted in Bold Red.
    - PAM motif is highlighted in Bold Magenta.
    """
    if mismatch_positions is None:
        mismatch_positions = [i for i in range(min(len(target), len(match_seq))) if target[i] != match_seq[i]]

    target_formatted = []
    match_formatted = []

    for i in range(len(match_seq)):
        t_base = target[i] if i < len(target) else "-"
        m_base = match_seq[i]

        if i in mismatch_positions:
            # Mutated base pair highlighted in bold red
            if use_rich:
                target_formatted.append(f"[dim]{t_base}[/dim]")
                match_formatted.append(f"[bold red]{m_base}[/bold red]")
            else:
                target_formatted.append(t_base)
                match_formatted.append(f"\033[1;31m{m_base}\033[0m")
        else:
            # Preserved matching base pair
            if use_rich:
                target_formatted.append(f"[cyan]{t_base}[/cyan]")
                match_formatted.append(f"[bold green]{m_base}[/bold green]")
            else:
                target_formatted.append(t_base)
                match_formatted.append(f"\033[1;32m{m_base}\033[0m")

    # PAM formatting
    if pam:
        if use_rich:
            pam_str = f" [bold magenta]{pam}[/bold magenta]"
        else:
            pam_str = f" \033[1;35m{pam}\033[0m"
    else:
        pam_str = ""

    return {
        "target_display": "".join(target_formatted) + (" [dim]PAM[/dim]" if pam else ""),
        "match_display": "".join(match_formatted) + pam_str,
        "mismatch_count": len(mismatch_positions),
        "mismatch_positions": mismatch_positions
    }


if __name__ == "__main__":
    test_target = "ATGCCCCAACTAAATACTAC"
    test_match = "ATGCTCCAACTAAATCCTAC"
    test_pam = "CGG"

    res = format_visual_alignment(test_target, test_match, pam=test_pam, use_rich=False)
    print("Visual DNA Mismatch Formatter Test:")
    print(f"Target: {res['target_display']}")
    print(f"Match:  {res['match_display']}")
    print(f"Mismatches at positions: {res['mismatch_positions']}")
    assert len(res["mismatch_positions"]) == 2
    print("Visual mismatch formatter verified successfully!")
