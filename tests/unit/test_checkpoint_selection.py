"""Which validation metric decides the best epoch.

Selection used to be `current_val_accuracy > best_val_accuracy`. On an imbalanced split
that is close to useless: a model emitting one class for every sample scores the
majority-class prior (~87% on AI-Face) at epoch 1, cannot strictly beat it afterwards, and
so freezes `best_epoch` at 1. Everything later -- further training, graph rewiring, node
reduction -- is then computed and discarded, which is exactly why a real sweep produced
byte-identical record tables for three differently-configured cells.

Balanced accuracy is prevalence-free but pins to *exactly* 0.5 for such a model, so it ties
rather than improving and freezes just the same. AUROC is threshold-free and moves whenever
the ranking changes, which is why it is the default.
"""

import math

import pytest

from test_hierarchical import _selection_score
from test_helpers.args_utils import parse_args
from web_ui.gpu_queue_manager import GPUQueueManager

METRICS = ("accuracy", "balanced_accuracy", "auroc")


def test_default_is_auroc():
    assert parse_args([]).checkpoint_metric == "auroc"


@pytest.mark.parametrize("metric", METRICS)
def test_every_choice_is_accepted(metric):
    assert parse_args(["--checkpoint-metric", metric]).checkpoint_metric == metric


def test_an_unknown_metric_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint-metric", "f1"])


def test_the_flag_reaches_the_cli():
    manager = GPUQueueManager.__new__(GPUQueueManager)
    command = manager._build_command_args({"checkpoint_metric": "balanced_accuracy"})
    assert "--checkpoint-metric" in command
    assert "balanced_accuracy" in command


@pytest.mark.parametrize("metric,expected", [
    ("accuracy", 87.5),
    ("balanced_accuracy", 50.0),
    ("auroc", 0.74),
])
def test_selection_score_reads_the_requested_metric(metric, expected):
    metrics = {"accuracy": 87.5, "balanced_accuracy": 50.0, "auroc": 0.74}
    assert _selection_score(metrics, metric) == pytest.approx(expected)


def test_a_collapsed_model_ties_on_accuracy_but_not_on_auroc():
    """The freeze, stated directly: accuracy is identical across epochs while the ranking
    improves, so only a threshold-free metric can tell the epochs apart."""
    epoch1 = {"accuracy": 87.55, "balanced_accuracy": 50.0, "auroc": 0.62}
    epoch2 = {"accuracy": 87.55, "balanced_accuracy": 50.0, "auroc": 0.74}

    for metric in ("accuracy", "balanced_accuracy"):
        assert not _selection_score(epoch2, metric) > _selection_score(epoch1, metric)
    assert _selection_score(epoch2, "auroc") > _selection_score(epoch1, "auroc")


def test_missing_metric_falls_back_to_accuracy():
    """A single-class validation subsample leaves AUROC undefined; refusing to checkpoint
    at all would be worse than checkpointing on the metric that still exists."""
    assert _selection_score({"accuracy": 91.0}, "auroc") == pytest.approx(91.0)


def test_nan_metric_falls_back_to_accuracy():
    metrics = {"accuracy": 91.0, "auroc": float("nan")}
    assert _selection_score(metrics, "auroc") == pytest.approx(91.0)


def test_fallback_is_silent_when_accuracy_was_requested(capsys):
    _selection_score({"accuracy": 50.0}, "accuracy")
    assert "falling back" not in capsys.readouterr().out


def test_fallback_says_so_when_it_happens(capsys):
    _selection_score({"accuracy": 50.0}, "auroc")
    assert "falling back to accuracy" in capsys.readouterr().out


def test_evaluate_model_reports_the_metrics_selection_needs():
    """The three keys `_selection_score` can be asked for must all be produced."""
    import inspect

    import test_hierarchical

    source = inspect.getsource(test_hierarchical.evaluate_model)
    for key in ("balanced_accuracy", "auroc", "n_positive"):
        assert f"final_metrics['{key}']" in source, key
    # Computed by the benchmark's own implementation, so the number logged per epoch and
    # the number in results.csv cannot drift apart.
    assert "discrimination_metrics" in source


def test_the_fingerprint_records_which_criterion_chose_the_checkpoint():
    """Two runs selected on different metrics are not comparable."""
    import inspect

    import test_hierarchical

    source = inspect.getsource(test_hierarchical.main)
    assert '"checkpoint_metric"' in source
    assert '"best_selection_score"' in source
