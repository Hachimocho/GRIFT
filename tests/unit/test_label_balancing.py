"""Equalizing the real/fake class counts.

`--fair-train`/`--fair-test` balance *demographic* subgroups (race x gender) and leave the
class prior untouched -- which is why they could not fix majority-class collapse, and why
this is a separate mechanism. The corrected AI-Face split is ~87% fake; at that prior BCE is
minimized substantially by raising the output bias, so models emit one class for every
sample.
"""

import pytest

from test_helpers.args_utils import parse_args
from test_helpers.data_graph_utils import _apply_label_balancing, balance_nodes_by_label
from web_ui.gpu_queue_manager import GPUQueueManager


class FakeNode:
    """Minimal stand-in: the balancer needs only `node_id` and `get_label()`."""

    def __init__(self, node_id, label):
        self.node_id = node_id
        self._label = label

    def get_label(self):
        return self._label


def imbalanced(n_fake=875, n_real=125):
    """The real prior: about 87.5% fake."""
    return (
        [FakeNode(f"fake{i:05d}", 1) for i in range(n_fake)]
        + [FakeNode(f"real{i:05d}", 0) for i in range(n_real)]
    )


def counts(nodes):
    result = {0: 0, 1: 0}
    for node in nodes:
        result[int(node.get_label())] += 1
    return result


# -- the balancer ---------------------------------------------------------------- #

def test_it_equalizes_the_classes():
    balanced = balance_nodes_by_label(imbalanced())
    assert counts(balanced) == {0: 125, 1: 125}


def test_the_minority_class_caps_the_result():
    """Only ~13% of the corpus is real, so most fakes are necessarily discarded."""
    balanced = balance_nodes_by_label(imbalanced(n_fake=875, n_real=125))
    assert len(balanced) == 250
    assert len(balanced) < len(imbalanced())


def test_a_target_size_is_respected():
    balanced = balance_nodes_by_label(imbalanced(), target_num_nodes=100)
    assert counts(balanced) == {0: 50, 1: 50}


def test_a_target_larger_than_the_minority_class_allows_is_capped(capsys):
    balanced = balance_nodes_by_label(imbalanced(n_real=30), target_num_nodes=400)
    assert counts(balanced) == {0: 30, 1: 30}
    # Said out loud, so a smaller-than-requested graph is not a silent surprise.
    assert "caps perfect balance" in capsys.readouterr().out


def test_the_result_is_interleaved_not_grouped_by_class():
    """A step-limited traversal or a truncated cache takes a prefix; in label order that
    prefix would be one class."""
    balanced = balance_nodes_by_label(imbalanced())
    prefix = counts(balanced[:50])
    assert prefix[0] > 0 and prefix[1] > 0


def test_selection_depends_only_on_the_nodes_offered():
    """Content-addressed seed, matching balance_nodes_by_subgroup: the choice must not
    depend on how much randomness anything upstream consumed."""
    import random

    nodes = imbalanced()
    random.seed(1)
    first = [node.node_id for node in balance_nodes_by_label(nodes)]
    for _ in range(500):
        random.random()
    second = [node.node_id for node in balance_nodes_by_label(nodes)]
    assert first == second


def test_a_single_class_list_is_refused():
    """Balancing is impossible and training on it would be meaningless."""
    only_fake = [FakeNode(f"f{i}", 1) for i in range(50)]
    with pytest.raises(ValueError, match="only class"):
        balance_nodes_by_label(only_fake)


def test_an_empty_list_returns_empty():
    assert balance_nodes_by_label([]) == []


# -- the flag -------------------------------------------------------------------- #

def test_the_default_changes_nothing():
    assert parse_args([]).balance_labels == "none"


@pytest.mark.parametrize("mode", ("none", "train", "all"))
def test_every_mode_is_accepted(mode):
    assert parse_args(["--balance-labels", mode]).balance_labels == mode


def test_an_unknown_mode_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--balance-labels", "test"])


def test_the_flag_reaches_the_cli():
    manager = GPUQueueManager.__new__(GPUQueueManager)
    command = manager._build_command_args({"balance_labels": "train"})
    assert "--balance-labels" in command
    assert "train" in command


class Args:
    def __init__(self, mode, cached_nodes=None):
        self.balance_labels = mode
        self.cached_nodes = cached_nodes


def test_mode_none_is_a_pass_through():
    train, val, test = imbalanced(), imbalanced(), imbalanced()
    out = _apply_label_balancing(Args("none"), train, val, test)
    assert out == (train, val, test)


def test_mode_train_leaves_the_evaluation_splits_alone():
    """Val and test keep the population's real distribution, so reported numbers stay on
    the data as it is."""
    train, val, test = imbalanced(), imbalanced(), imbalanced()
    balanced_train, out_val, out_test = _apply_label_balancing(
        Args("train"), train, val, test
    )
    assert counts(balanced_train) == {0: 125, 1: 125}
    assert out_val is val and out_test is test


def test_mode_all_balances_every_split():
    train, val, test = imbalanced(), imbalanced(), imbalanced()
    out = _apply_label_balancing(Args("all"), train, val, test)
    for split in out:
        assert counts(split) == {0: 125, 1: 125}


def test_the_cache_suffix_distinguishes_a_balanced_run():
    """A balanced and an unbalanced run must not share a graph cache entry. The node-set
    hash already guarantees that; the suffix makes the filename say so."""
    import inspect

    import test_hierarchical

    source = inspect.getsource(test_hierarchical.main)
    assert "label_part" in source
    assert "balance_labels" in source
