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
            if use_rich:
                target_formatted.append(f"[dim]{t_base}[/dim]")
                match_formatted.append(f"[bold red]{m_base}[/bold red]")
            else:
                target_formatted.append(t_base)
                match_formatted.append(f"\033[1;31m{m_base}\033[0m")
        else:
            if use_rich:
                target_formatted.append(f"[cyan]{t_base}[/cyan]")
                match_formatted.append(f"[bold green]{m_base}[/bold green]")
            else:
                target_formatted.append(t_base)
                match_formatted.append(f"\033[1;32m{m_base}\033[0m")

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


def generate_alignment_track(target: str, match_seq: str, pam: str = "", use_rich: bool = True) -> str:
    """
    Generates a full 3-line visual DNA alignment track highlighting matches ('|')
    and mutated base pairs ('.' highlighted in red).
    """
    min_len = min(len(target), len(match_seq))
    track_chars = []

    for i in range(min_len):
        if target[i] == match_seq[i]:
            if use_rich:
                track_chars.append("[bold green]|[/bold green]")
            else:
                track_chars.append("\033[1;32m|\033[0m")
        else:
            if use_rich:
                track_chars.append("[bold red].[[/bold red]") if False else track_chars.append("[bold red].[/bold red]")
            else:
                track_chars.append("\033[1;31m.\033[0m")

    target_line = " ".join(list(target))
    track_line = " ".join(track_chars)
    match_line = " ".join([
        f"[bold red]{b}[/bold red]" if use_rich and i < len(target) and target[i] != b else
        f"\033[1;31m{b}\033[0m" if not use_rich and i < len(target) and target[i] != b else
        f"[bold green]{b}[/bold green]" if use_rich else
        f"\033[1;32m{b}\033[0m"
        for i, b in enumerate(match_seq)
    ])

    if pam:
        pam_spaced = " ".join(list(pam))
        target_line += "   " + ("[dim]N G G[/dim]" if use_rich else "N G G")
        track_line += "   " + ("[bold magenta]: : :[/bold magenta]" if use_rich else ": : :")
        match_line += "   " + (f"[bold magenta]{pam_spaced}[/bold magenta]" if use_rich else f"\033[1;35m{pam_spaced}\033[0m")

    block = (
        f"Target  (5'->3'): {target_line}\n"
        f"Alignment Track : {track_line}\n"
        f"Off-Target Site : {match_line}"
    )
    return block


if __name__ == "__main__":
    t = "ATGCCCCAACTAAATACTAC"
    m = "ATGCTCCAACTAAATCCTAC"
    p = "CGG"

    print("Alignment Track Output:")
    print(generate_alignment_track(t, m, pam=p, use_rich=False))
