"""Test-time uncertainty scoring helpers.

These functions are intentionally post-hoc: they consume final-test prediction
records after the detector has finished training and do not affect training.

Implemented methods:
  - MSP          : Maximum Softmax Probability (simple baseline)
  - DDU          : Deterministic Uncertainty (per-class Gaussian on train logits)
  - Trust Score  : kNN trust ratio in logit space (train-fitted)
  - Graph UQ     : Neighbor prediction consistency in the test graph
"""

import csv
import json
import numpy as np
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def summarize_uncertainty_as_fake_policy(score_records, score_key):
    """Summarize fixed policies that treat the most uncertain samples as fake."""
    total = len(score_records)
    total_fake = sum(1 for r in score_records if int(r.get("label", 0)) == 1)
    original_false_negatives = sum(
        1 for r in score_records
        if int(r.get("label", 0)) == 1 and int(r.get("prediction", 0)) == 0
    )
    original_correct = sum(1 for r in score_records if int(r.get("correct", 0)) == 1)

    policies = []
    sorted_records = sorted(score_records, key=lambda r: float(r.get(score_key, 0.0)), reverse=True)
    for fraction in (0.05, 0.10, 0.20):
        num_flagged = max(1, int(round(total * fraction))) if total else 0
        flagged_ids = {r.get("node_id", "") for r in sorted_records[:num_flagged]}

        adjusted_correct = 0
        adjusted_false_negatives = 0
        for r in score_records:
            label = int(r.get("label", 0))
            pred = int(r.get("prediction", 0))
            if r.get("node_id", "") in flagged_ids:
                pred = 1
            if pred == label:
                adjusted_correct += 1
            if label == 1 and pred == 0:
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
        "original_fake_recall": (
            (total_fake - original_false_negatives) / total_fake if total_fake else None
        ),
        "treat_uncertain_as_fake_policies": policies,
    }


def save_score_records(score_records, output_path, fieldnames):
    """Write score records to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in score_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def _rank_normalize(values):
    """Map an array of floats to [0, 1] via rank-based normalization."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n)
    return ranks / max(n - 1, 1)


# ---------------------------------------------------------------------------
# MSP
# ---------------------------------------------------------------------------

def compute_msp_scores(prediction_records):
    """Compute MSP uncertainty from final-test prediction records."""
    score_records = []
    for record in prediction_records:
        confidence = float(record.get("confidence", 0.0))
        score_records.append({
            "node_id": record.get("node_id", ""),
            "label": int(record.get("label", 0)),
            "prediction": int(record.get("prediction", 0)),
            "confidence": confidence,
            "msp_uncertainty": 1.0 - confidence,
            "correct": int(record.get("correct", 0)),
            "false_negative": int(record.get("false_negative", 0)),
        })
    return score_records


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
        fieldnames=["node_id", "label", "prediction", "confidence",
                    "msp_uncertainty", "correct", "false_negative"],
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


# ---------------------------------------------------------------------------
# DDU  (per-class Gaussian on training logits)
# ---------------------------------------------------------------------------

def run_ddu_uncertainty(test_prediction_records, train_prediction_records, output_dir):
    """Compute DDU uncertainty using per-class Gaussian fit on training logits.

    Simplified logit-space DDU: fit N(mu_fake, sigma_fake) and N(mu_real, sigma_real)
    on training data, then score test samples by their max log-density across
    the two class Gaussians.  Samples far from both class modes get high uncertainty.

    Full DDU (Mukhoti et al. 2023) uses spectral-normalised penultimate features;
    the logit is used here as a proxy since embedding hooks would require
    per-architecture changes.
    """
    uncertainty_dir = Path(output_dir) / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    def _skip(reason):
        summary = {"method": "ddu", "status": "skipped", "reason": reason,
                   "generated_at": datetime.now().isoformat()}
        sp = uncertainty_dir / "ddu_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "skipped", "summary_file": str(sp), "summary": summary}

    if not train_prediction_records:
        return _skip("no train prediction records provided")

    train_fake = np.array([float(r["logit"]) for r in train_prediction_records
                           if int(r.get("label", 0)) == 1], dtype=float)
    train_real = np.array([float(r["logit"]) for r in train_prediction_records
                           if int(r.get("label", 0)) == 0], dtype=float)

    if len(train_fake) == 0 or len(train_real) == 0:
        return _skip("insufficient class-separated training data")

    fake_mean, fake_std = float(np.mean(train_fake)), float(np.std(train_fake)) + 1e-8
    real_mean, real_std = float(np.mean(train_real)), float(np.std(train_real)) + 1e-8

    from scipy.stats import norm as sp_norm

    raw_scores = []
    score_records = []
    for record in test_prediction_records:
        logit = float(record.get("logit", 0.0))
        p_fake = float(sp_norm.pdf(logit, fake_mean, fake_std))
        p_real = float(sp_norm.pdf(logit, real_mean, real_std))
        in_dist_score = max(p_fake, p_real)
        raw_scores.append(in_dist_score)
        score_records.append({
            "node_id": record.get("node_id", ""),
            "label": int(record.get("label", 0)),
            "prediction": int(record.get("prediction", 0)),
            "logit": logit,
            "confidence": float(record.get("confidence", 0.0)),
            "p_fake": p_fake,
            "p_real": p_real,
            "in_dist_score": in_dist_score,
            "ddu_uncertainty": None,
            "correct": int(record.get("correct", 0)),
            "false_negative": int(record.get("false_negative", 0)),
        })

    norm_in_dist = _rank_normalize(raw_scores)
    for i, rec in enumerate(score_records):
        rec["ddu_uncertainty"] = float(1.0 - norm_in_dist[i])

    method_summary = summarize_uncertainty_as_fake_policy(score_records, "ddu_uncertainty")

    scores_path = uncertainty_dir / "ddu_scores.csv"
    summary_path = uncertainty_dir / "ddu_summary.json"

    save_score_records(
        score_records=score_records,
        output_path=scores_path,
        fieldnames=["node_id", "label", "prediction", "logit", "confidence",
                    "p_fake", "p_real", "in_dist_score", "ddu_uncertainty",
                    "correct", "false_negative"],
    )

    summary = {
        "method": "ddu",
        "status": "completed",
        "generated_at": datetime.now().isoformat(),
        "scores_file": str(scores_path),
        "train_fake_gaussian": {"mean": fake_mean, "std": fake_std},
        "train_real_gaussian": {"mean": real_mean, "std": real_std},
        "num_train_fake": int(len(train_fake)),
        "num_train_real": int(len(train_real)),
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


# ---------------------------------------------------------------------------
# Trust Score  (1-D kNN trust ratio in logit space)
# ---------------------------------------------------------------------------

def run_trust_score_uncertainty(test_prediction_records, train_prediction_records, output_dir):
    """Compute Trust Score uncertainty using nearest-neighbour logit distances.

    Trust Score (Jiang et al. 2018) = dist_to_nearest_other_class_neighbour
                                      / dist_to_nearest_same_class_neighbour

    Low trust score -> prediction is not supported by nearby training data -> uncertain.
    This implementation works in 1-D logit space (original uses full embeddings).
    """
    uncertainty_dir = Path(output_dir) / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    def _skip(reason):
        summary = {"method": "trust_score", "status": "skipped", "reason": reason,
                   "generated_at": datetime.now().isoformat()}
        sp = uncertainty_dir / "trust_score_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "skipped", "summary_file": str(sp), "summary": summary}

    if not train_prediction_records:
        return _skip("no train prediction records provided")

    train_fake = np.array([float(r["logit"]) for r in train_prediction_records
                           if int(r.get("label", 0)) == 1], dtype=float)
    train_real = np.array([float(r["logit"]) for r in train_prediction_records
                           if int(r.get("label", 0)) == 0], dtype=float)

    if len(train_fake) == 0 or len(train_real) == 0:
        return _skip("insufficient class-separated training data")

    raw_trust = []
    score_records = []
    for record in test_prediction_records:
        logit = float(record.get("logit", 0.0))
        predicted = int(record.get("prediction", 0))

        same_class = train_fake if predicted == 1 else train_real
        other_class = train_real if predicted == 1 else train_fake

        d_same = float(np.min(np.abs(same_class - logit))) if len(same_class) else 1.0
        d_other = float(np.min(np.abs(other_class - logit))) if len(other_class) else 0.0
        trust_score = d_other / (d_same + 1e-8)

        raw_trust.append(trust_score)
        score_records.append({
            "node_id": record.get("node_id", ""),
            "label": int(record.get("label", 0)),
            "prediction": predicted,
            "logit": logit,
            "confidence": float(record.get("confidence", 0.0)),
            "d_same_class": d_same,
            "d_other_class": d_other,
            "trust_score": trust_score,
            "trust_score_uncertainty": None,
            "correct": int(record.get("correct", 0)),
            "false_negative": int(record.get("false_negative", 0)),
        })

    norm_trust = _rank_normalize(raw_trust)
    for i, rec in enumerate(score_records):
        rec["trust_score_uncertainty"] = float(1.0 - norm_trust[i])

    method_summary = summarize_uncertainty_as_fake_policy(score_records, "trust_score_uncertainty")

    scores_path = uncertainty_dir / "trust_score_scores.csv"
    summary_path = uncertainty_dir / "trust_score_summary.json"

    save_score_records(
        score_records=score_records,
        output_path=scores_path,
        fieldnames=["node_id", "label", "prediction", "logit", "confidence",
                    "d_same_class", "d_other_class", "trust_score",
                    "trust_score_uncertainty", "correct", "false_negative"],
    )

    summary = {
        "method": "trust_score",
        "status": "completed",
        "generated_at": datetime.now().isoformat(),
        "scores_file": str(scores_path),
        "num_train_fake": int(len(train_fake)),
        "num_train_real": int(len(train_real)),
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


# ---------------------------------------------------------------------------
# Graph Uncertainty  (neighbor prediction consistency)
# ---------------------------------------------------------------------------

def run_graph_uncertainty(test_prediction_records, neighbor_map, output_dir,
                          bridge_node_ids=None):
    """Compute uncertainty from local neighbor consistency in the test graph.

    For each test node, look at predictions of its graph neighbors.
    Disagreement (low homophily) signals high uncertainty.

    Nodes added by ensure_graph_connected() have arbitrary (non-similarity) edges,
    so their uncertainty results are dropped and flagged as bridge nodes.

    Args:
        test_prediction_records: list of per-node prediction dicts.
        neighbor_map: {node_id: [neighbor_node_id, ...]} from the test graph.
        output_dir: directory to write artifacts.
        bridge_node_ids: optional set of node IDs inserted as bridge connections.
    """
    uncertainty_dir = Path(output_dir) / "uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    if not neighbor_map:
        summary = {"method": "graph", "status": "skipped",
                   "reason": "no neighbor map provided",
                   "generated_at": datetime.now().isoformat()}
        sp = uncertainty_dir / "graph_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        return {"status": "skipped", "summary_file": str(sp), "summary": summary}

    bridge_node_ids = set(bridge_node_ids or [])
    pred_by_id = {r.get("node_id", ""): r for r in test_prediction_records}

    score_records = []
    for record in test_prediction_records:
        node_id = record.get("node_id", "")
        label = int(record.get("label", 0))
        prediction = int(record.get("prediction", 0))
        confidence = float(record.get("confidence", 0.0))
        correct = int(record.get("correct", 0))
        false_neg = int(record.get("false_negative", 0))

        if node_id in bridge_node_ids:
            score_records.append({
                "node_id": node_id, "label": label, "prediction": prediction,
                "confidence": confidence, "neighbor_count": 0,
                "neighbor_homophily": None, "neighbor_prob_variance": None,
                "graph_uncertainty": None, "is_bridge_node": 1,
                "correct": correct, "false_negative": false_neg,
            })
            continue

        raw_neighbors = neighbor_map.get(node_id, [])
        valid_neighbors = [n for n in raw_neighbors
                           if n not in bridge_node_ids and n in pred_by_id]

        if not valid_neighbors:
            score_records.append({
                "node_id": node_id, "label": label, "prediction": prediction,
                "confidence": confidence, "neighbor_count": 0,
                "neighbor_homophily": None, "neighbor_prob_variance": None,
                "graph_uncertainty": 0.5, "is_bridge_node": 0,
                "correct": correct, "false_negative": false_neg,
            })
            continue

        n_preds = [int(pred_by_id[n].get("prediction", 0)) for n in valid_neighbors]
        n_probs = [float(pred_by_id[n].get("probability_fake", 0.5)) for n in valid_neighbors]
        homophily = float(sum(1 for p in n_preds if p == prediction) / len(n_preds))
        prob_var = float(np.var(n_probs)) if len(n_probs) > 1 else 0.0

        score_records.append({
            "node_id": node_id, "label": label, "prediction": prediction,
            "confidence": confidence, "neighbor_count": len(valid_neighbors),
            "neighbor_homophily": homophily, "neighbor_prob_variance": prob_var,
            "graph_uncertainty": float(1.0 - homophily), "is_bridge_node": 0,
            "correct": correct, "false_negative": false_neg,
        })

    scoreable = [r for r in score_records
                 if r.get("graph_uncertainty") is not None and not r.get("is_bridge_node")]
    method_summary = (
        summarize_uncertainty_as_fake_policy(scoreable, "graph_uncertainty")
        if scoreable else {}
    )

    scores_path = uncertainty_dir / "graph_scores.csv"
    summary_path = uncertainty_dir / "graph_summary.json"

    save_score_records(
        score_records=score_records,
        output_path=scores_path,
        fieldnames=["node_id", "label", "prediction", "confidence",
                    "neighbor_count", "neighbor_homophily", "neighbor_prob_variance",
                    "graph_uncertainty", "is_bridge_node", "correct", "false_negative"],
    )

    summary = {
        "method": "graph",
        "status": "completed",
        "generated_at": datetime.now().isoformat(),
        "scores_file": str(scores_path),
        "num_nodes_with_neighbors": sum(1 for r in score_records
                                        if r.get("neighbor_count", 0) > 0),
        "num_bridge_nodes_dropped": int(len(bridge_node_ids)),
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
