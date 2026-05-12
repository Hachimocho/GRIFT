"""Test-time uncertainty scoring helpers.

These functions are intentionally post-hoc: they consume final-test prediction
records after the detector has finished training and do not affect training.
"""

import csv
import json
from datetime import datetime
from pathlib import Path


def compute_msp_scores(prediction_records):
    """Compute MSP uncertainty from final-test prediction records."""
    score_records = []
    for record in prediction_records:
        confidence = float(record.get("confidence", 0.0))
        uncertainty = 1.0 - confidence
        score_records.append({
            "node_id": record.get("node_id", ""),
            "label": int(record.get("label", 0)),
            "prediction": int(record.get("prediction", 0)),
            "confidence": confidence,
            "msp_uncertainty": uncertainty,
            "correct": int(record.get("correct", 0)),
            "false_negative": int(record.get("false_negative", 0)),
        })
    return score_records


def summarize_uncertainty_as_fake_policy(score_records, score_key):
    """Summarize fixed policies that treat the most uncertain samples as fake."""
    total = len(score_records)
    total_fake = sum(1 for record in score_records if int(record.get("label", 0)) == 1)
    original_false_negatives = sum(
        1
        for record in score_records
        if int(record.get("label", 0)) == 1 and int(record.get("prediction", 0)) == 0
    )
    original_correct = sum(1 for record in score_records if int(record.get("correct", 0)) == 1)

    policies = []
    sorted_records = sorted(score_records, key=lambda record: float(record.get(score_key, 0.0)), reverse=True)
    for fraction in (0.05, 0.10, 0.20):
        num_flagged = max(1, int(round(total * fraction))) if total else 0
        flagged_ids = {record.get("node_id", "") for record in sorted_records[:num_flagged]}

        adjusted_correct = 0
        adjusted_false_negatives = 0
        for record in score_records:
            label = int(record.get("label", 0))
            prediction = int(record.get("prediction", 0))
            if record.get("node_id", "") in flagged_ids:
                prediction = 1

            if prediction == label:
                adjusted_correct += 1
            if label == 1 and prediction == 0:
                adjusted_false_negatives += 1

        threshold = float(sorted_records[num_flagged - 1].get(score_key, 0.0)) if num_flagged else None
        fake_recall = ((total_fake - adjusted_false_negatives) / total_fake) if total_fake else None
        policies.append({
            "top_uncertain_fraction": fraction,
            "num_flagged_as_fake": num_flagged,
            "uncertainty_threshold": threshold,
            "remaining_false_negatives": adjusted_false_negatives,
            "false_negative_reduction": original_false_negatives - adjusted_false_negatives,
            "accuracy_after_fake_override": (adjusted_correct / total) if total else None,
            "fake_recall_after_fake_override": fake_recall,
        })

    return {
        "num_records": total,
        "original_accuracy": (original_correct / total) if total else None,
        "original_false_negatives": original_false_negatives,
        "original_fake_recall": ((total_fake - original_false_negatives) / total_fake) if total_fake else None,
        "treat_uncertain_as_fake_policies": policies,
    }


def save_score_records(score_records, output_path, fieldnames):
    """Write score records to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in score_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def run_msp_uncertainty(prediction_records, output_dir):
    """Compute and save MSP uncertainty scores and summary."""
    uncertainty_dir = Path(output_dir) / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    score_records = compute_msp_scores(prediction_records)
    method_summary = summarize_uncertainty_as_fake_policy(score_records, "msp_uncertainty")

    scores_path = uncertainty_dir / "msp_scores.csv"
    summary_path = uncertainty_dir / "msp_summary.json"

    save_score_records(
        score_records=score_records,
        output_path=scores_path,
        fieldnames=[
            "node_id",
            "label",
            "prediction",
            "confidence",
            "msp_uncertainty",
            "correct",
            "false_negative",
        ],
    )

    summary = {
        "method": "msp",
        "status": "completed",
        "generated_at": datetime.now().isoformat(),
        "scores_file": str(scores_path),
        **method_summary,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "status": "completed",
        "scores_file": str(scores_path),
        "summary_file": str(summary_path),
        "summary": summary,
    }
