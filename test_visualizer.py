import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualizer import format_visual_alignment, generate_alignment_track, describe_mutations, format_off_target_summary_card


def test_visualizer_suite():
    print("=" * 65)
    print("      Testing Visual DNA Mismatch & Alignment Track Suite      ")
    print("=" * 65)

    target = "ATGCCCCAACTAAATACTAC"

    # 1. Exact match test
    print("\n--- Test 1: Exact Match Visual Track ---")
    res_exact = format_visual_alignment(target, target, pam="TGG", use_rich=False)
    assert res_exact["mismatch_count"] == 0
    track_exact = generate_alignment_track(target, target, pam="TGG", use_rich=False)
    assert "." not in track_exact
    print("[OK] Exact match visual track verified (all '|', 0 mutations).")

    # 2. Single Distal Mutation Test
    print("\n--- Test 2: Single Distal Mutation (Position 1) ---")
    distal_match = "CTGCCCCAACTAAATACTAC"
    muts_distal = describe_mutations(target, distal_match, use_rich=False)
    assert len(muts_distal) == 1
    assert muts_distal[0]["position_1"] == 1
    assert muts_distal[0]["region"] == "Distal Region"
    print(f"  {muts_distal[0]['description']}")
    print("[OK] Distal mutation accurately classified.")

    # 3. Single Seed Mutation Test
    print("\n--- Test 3: Single Seed Mutation (Position 19) ---")
    seed_match = "ATGCCCCAACTAAATACTTC"
    muts_seed = describe_mutations(target, seed_match, use_rich=False)
    assert len(muts_seed) == 1
    assert muts_seed[0]["position_1"] == 19
    assert muts_seed[0]["region"] == "Seed Region"
    print(f"  {muts_seed[0]['description']}")
    print("[OK] Seed region mutation accurately classified.")

    # 4. Multi-mismatch sequence test
    print("\n--- Test 4: Multi-Mismatch Alignment Track ---")
    multi_match = "ATGCTCCAACTAAATCCTAC"
    muts_multi = describe_mutations(target, multi_match, use_rich=False)
    assert len(muts_multi) == 2
    track_multi = generate_alignment_track(target, multi_match, pam="CGG", use_rich=False)
    print(track_multi)
    assert "." in track_multi
    print("[OK] Multi-mismatch alignment track generated correctly.")

    # 5. Full Summary Card Test
    print("\n--- Test 5: Off-Target Analysis Card Rendering ---")
    sample_item = {
        "position": 54200,
        "sequence": multi_match,
        "pam": "AGG",
        "severity_score": 52.4,
        "risk_tier": "MEDIUM",
        "pam_type": "canonical"
    }
    card = format_off_target_summary_card(sample_item, target, use_rich=False)
    assert "Off-Target Site @ Position 54,200" in card
    print(card)
    print("[OK] Analysis card verified.")

    print("\n[PASSED] Visual Mismatch Engine fully verified!")
    return True


if __name__ == "__main__":
    test_visualizer_suite()
