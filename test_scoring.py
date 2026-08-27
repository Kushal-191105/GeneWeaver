import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import read_fasta, read_target
from src.gpu_alignment import gpu_find_matches_with_mismatches
from src.scoring import rank_off_targets, validate_pam, extract_pam, calculate_severity_score


def test_biological_scoring_suite():
    print("=" * 65)
    print("      Testing Biological Scoring & Off-Target Ranking Engine      ")
    print("=" * 65)

    # 1. Unit Tests on Core Biological Rules
    print("\n--- Test 1: Canonical SpCas9 PAM Motif Recognition ---")
    assert validate_pam("CGG")["type"] == "canonical"
    assert validate_pam("TGG")["type"] == "canonical"
    assert validate_pam("AGG")["type"] == "canonical"
    assert validate_pam("GGG")["type"] == "canonical"
    print("[OK] Canonical NGG PAMs correctly recognized.")

    print("\n--- Test 2: Non-Canonical & Non-Viable PAM Validation ---")
    assert validate_pam("AAG")["type"] == "non-canonical"
    assert validate_pam("CAG")["type"] == "non-canonical"
    assert validate_pam("ATC")["type"] == "invalid"
    assert validate_pam("TTT")["type"] == "invalid"
    print("[OK] Non-canonical NAG and non-viable PAMs correctly classified.")

    print("\n--- Test 3: Seed vs Distal Region Mismatch Sensitivity ---")
    # Mismatch at distal position 1 (furthest from PAM)
    distal_score = calculate_severity_score([0], "CGG")["severity_score"]
    # Mismatch at proximal seed position 19 (closest to PAM)
    seed_score = calculate_severity_score([18], "CGG")["severity_score"]
    print(f"Distal mismatch score: {distal_score}% vs Seed mismatch score: {seed_score}%")
    assert distal_score > seed_score, "Distal mismatch should have higher cleavage tolerance than seed mismatch!"
    print("[OK] Seed region penalty dynamics verified.")

    # 2. Integration Test on Real Genomic Dataset
    print("\n--- Test 4: Real Genomic Dataset Off-Target Ranking ---")
    sequences = read_fasta("data/genome.fasta")
    genome = "".join(sequences)
    target = read_target("data/target.txt")
    print(f"Loaded Genome: {len(genome):,} bp | Target: {target}")

    raw_matches = gpu_find_matches_with_mismatches(genome, target, max_mismatches=2)
    print(f"Raw Matches Identified by GPU: {len(raw_matches)}")

    ranked_results = rank_off_targets(raw_matches, genome, target)
    print(f"Biological Off-Targets Scored & Ranked: {len(ranked_results)}")

    print("\n" + "-" * 75)
    print(f"{'Rank':<5} | {'Pos':<9} | {'Sequence':<22} | {'PAM':<5} | {'Type':<12} | {'Score':<8} | {'Risk':<12}")
    print("-" * 75)
    for r in ranked_results[:10]:
        print(f"#{r['rank']:<4} | {r['position']:<9} | {r['sequence']:<22} | {r['pam']:<5} | {r['pam_type']:<12} | {r['severity_score']:<7.1f}% | {r['risk_badge']:<12}")
    print("-" * 75)

    for i in range(len(ranked_results) - 1):
        assert ranked_results[i]["severity_score"] >= ranked_results[i + 1]["severity_score"], "Ranking order violated!"

    print("\n[PASSED] Biological Scoring Engine fully verified on real genomic data!")
    return True


if __name__ == "__main__":
    test_biological_scoring_suite()
