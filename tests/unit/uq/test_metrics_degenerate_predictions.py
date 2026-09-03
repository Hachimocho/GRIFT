"""A model that emits one class for every sample must be flagged.

`clf_accuracy` then equals the majority-class prior -- about 0.87 on this dataset -- which
reads as a respectable result and is not a measurement at all. It is distinct from the two
degeneracies already covered: the labels are both present (so `SINGLE_CLASS_LABELS` does
not fire) and the uncertainty scores vary (so `DEGENERATE_CONSTANT_SCORE` does not). Before
this flag existed the row came back `ok` with no flag, and identical accuracy across three
different traversals was the only visible symptom.
"""

import numpy as np

from evaluation.uq.metrics import (
    DEGENERATE_CONSTANT_SCORE, SINGLE_CLASS_LABELS, SINGLE_CLASS_PREDICTIONS,
    discrimination_metrics,
)


def imbalanced_labels(n=5000, positive_fraction=0.8672):
    """The real smoke-run split: 4336 of 5000 test samples are class 1."""
    labels = np.zeros(n, dtype=int)
    labels[: int(round(n * positive_fraction))] = 1
    return labels


def test_majority_class_collapse_is_flagged():
    labels = imbalanced_labels()
    # Every probability above 0.5, but varying -- exactly what a collapsed model produces.
    probabilities = np.linspace(0.67, 0.99, labels.size)

    result, flags = discrimination_metrics(labels, probabilities)

    assert SINGLE_CLASS_PREDICTIONS in flags
    # The two existing flags deliberately do not fire here.
    assert SINGLE_CLASS_LABELS not in flags
    assert DEGENERATE_CONSTANT_SCORE not in flags
    # Accuracy is the prior, and balanced accuracy pins to 0.5.
    assert result["accuracy"] == 0.8672
    assert result["balanced_accuracy"] == 0.5


def test_all_negative_collapse_is_also_flagged():
    labels = imbalanced_labels()
    probabilities = np.linspace(0.01, 0.49, labels.size)
    _result, flags = discrimination_metrics(labels, probabilities)
    assert SINGLE_CLASS_PREDICTIONS in flags


def test_a_discriminating_model_is_not_flagged():
    labels = imbalanced_labels()
    probabilities = np.where(labels == 1, 0.9, 0.1)
    result, flags = discrimination_metrics(labels, probabilities)
    assert SINGLE_CLASS_PREDICTIONS not in flags
    assert result["balanced_accuracy"] == 1.0


def test_the_flag_reaches_the_scored_row():
    import pandas as pd

    from evaluation.uq.scoring import Cell, score_cells

    labels = imbalanced_labels(n=400)
    probabilities = np.linspace(0.67, 0.99, labels.size)
    frame = pd.DataFrame({
        "record_id": [f"r{i:04d}" for i in range(labels.size)],
        "label": labels,
        "pred": (probabilities > 0.5).astype(int),
        "correct": ((probabilities > 0.5).astype(int) == labels).astype(int),
        "prob": probabilities,
    })
    results = score_cells(
        [Cell(detector="tiny", method_id="baseline_maxprob",
              score_column="u_maxprob", frame=frame)],
        require_comparable=False,
    )
    assert SINGLE_CLASS_PREDICTIONS in str(results.iloc[0]["status_flags"])
