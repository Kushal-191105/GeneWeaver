import os
import sys
import json
import csv

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualizer import describe_mutations, format_visual_alignment


def export_results_to_json(results: list, target: str, filepath: str = "data/off_target_report.json") -> str:
    """
    Exports full CRISPR off-target alignment and biological severity ranking
    to a structured JSON document.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    export_payload = {
        "engine": "GeneWeaver CRISPR Alignment Engine",
        "version": "4.0.0",
        "target_sequence": target,
        "target_length": len(target),
        "total_hits_found": len(results),
        "summary": {
            "high_risk_count": sum(1 for r in results if r.get("risk_tier") == "HIGH"),
            "med_risk_count": sum(1 for r in results if r.get("risk_tier") == "MEDIUM"),
            "low_risk_count": sum(1 for r in results if r.get("risk_tier") == "LOW")
        },
        "off_targets": []
    }

    for r in results:
        muts = describe_mutations(target, r["sequence"], use_rich=False)
        record = {
            "rank": r.get("rank"),
            "genomic_position": r.get("position"),
            "sequence": r.get("sequence"),
            "pam_motif": r.get("pam"),
            "pam_type": r.get("pam_type"),
            "pam_viability": r.get("pam_type") != "invalid",
            "severity_score_pct": r.get("severity_score"),
            "risk_tier": r.get("risk_tier"),
            "mismatch_count": r.get("mismatches"),
            "mismatch_positions": r.get("mismatch_positions"),
            "point_mutations": muts
        }
        export_payload["off_targets"].append(record)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    return filepath


def export_results_to_csv(results: list, filepath: str = "data/off_target_summary.csv") -> str:
    """
    Exports tabular summary of off-target candidates to CSV format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    fieldnames = [
        "rank", "position", "sequence", "pam", "pam_type",
        "severity_score", "risk_tier", "mismatches", "mismatch_positions"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "rank": r.get("rank"),
                "position": r.get("position"),
                "sequence": r.get("sequence"),
                "pam": r.get("pam"),
                "pam_type": r.get("pam_type"),
                "severity_score": r.get("severity_score"),
                "risk_tier": r.get("risk_tier"),
                "mismatches": r.get("mismatches"),
                "mismatch_positions": str(r.get("mismatch_positions"))
            }
            writer.writerow(row)

    return filepath


if __name__ == "__main__":
    test_target = "ATGCCCCAACTAAATACTAC"
    dummy_results = [
        {
            "rank": 1,
            "position": 1420,
            "sequence": "ATGCTCCAACTAAATCCTAC",
            "pam": "CGG",
            "pam_type": "canonical",
            "severity_score": 52.4,
            "risk_tier": "MEDIUM",
            "mismatches": 2,
            "mismatch_positions": [4, 15]
        },
        {
            "rank": 2,
            "position": 58310,
            "sequence": "ATGCCCCAACTAAATACTAC",
            "pam": "CGT",
            "pam_type": "invalid",
            "severity_score": 0.0,
            "risk_tier": "LOW",
            "mismatches": 0,
            "mismatch_positions": []
        }
    ]

    json_path = export_results_to_json(dummy_results, test_target, "data/test_report.json")
    csv_path = export_results_to_csv(dummy_results, "data/test_summary.csv")

    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)
    print(f"Exported JSON: {json_path}")
    print(f"Exported CSV:  {csv_path}")
    print("Report export module verified successfully!")
