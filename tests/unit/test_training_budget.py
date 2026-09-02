"""The per-epoch training budget, which is the run's real sample count.

`--train-steps` bounds how far a traversal *walks*; `max_nodes_per_epoch` bounds how many
nodes are actually *trained on*. The two capabilities shipped different hardcoded defaults
-- 5000 basic, 10000 DQN -- so an i-value arm trained on twice the samples of a random arm
and any accuracy gap between them confounded sample selection with sample count. These
tests pin the override that makes arms comparable, and pin the differing defaults so the
confound cannot silently return.
"""

import pytest

from trainers.capabilities.BasicTrainingCapability import BasicTrainingCapability
from trainers.capabilities.DQNCapability import DQNCapability


class _StubTrainer:
    def __init__(self, **attrs):
        import torch
        self.device = torch.device('cpu')
        self.models = []
        self.attribute_metadata = None
        for key, value in attrs.items():
            setattr(self, key, value)


def _basic(**attrs):
    return BasicTrainingCapability(_StubTrainer(**attrs))


def _dqn(**attrs):
    return DQNCapability(_StubTrainer(**attrs))


def test_defaults_differ_which_is_the_confound():
    # Documented, not endorsed: this asymmetry is why the override exists.
    assert _basic().max_nodes_per_epoch == 5000
    assert _dqn().max_nodes_per_epoch == 10000


@pytest.mark.parametrize("budget", [1, 500, 8000, 20000])
def test_override_applies_to_both_capabilities(budget):
    assert _basic(max_nodes_per_epoch=budget).max_nodes_per_epoch == budget
    assert _dqn(max_nodes_per_epoch=budget).max_nodes_per_epoch == budget


def test_one_budget_makes_the_two_capabilities_agree():
    # The point of the flag: set it once and the arms become comparable.
    assert (_basic(max_nodes_per_epoch=7500).max_nodes_per_epoch
            == _dqn(max_nodes_per_epoch=7500).max_nodes_per_epoch)


def test_none_falls_back_to_the_capability_default():
    assert _basic(max_nodes_per_epoch=None).max_nodes_per_epoch == 5000
    assert _dqn(max_nodes_per_epoch=None).max_nodes_per_epoch == 10000


def test_flag_reaches_the_cli_and_the_queue():
    from test_helpers.args_utils import parse_args
    from web_ui.gpu_queue_manager import ARG_MAPPING, validate_config_keys

    assert parse_args(['--max-nodes-per-epoch', '9000']).max_nodes_per_epoch == 9000
    assert parse_args([]).max_nodes_per_epoch is None
    assert ARG_MAPPING['max_nodes_per_epoch'] == '--max-nodes-per-epoch'
    # A sweep config carrying the key must not be silently dropped on the way to the CLI.
    assert validate_config_keys({'max_nodes_per_epoch': 9000}) == []


def test_the_flag_survives_adaptive_trainer_construction():
    """The regression that hid for two sweeps.

    `AdaptiveTrainer.__init__` stored the kwarg and then overwrote it with a hardcoded
    10000 before the capabilities were built, so `--max-nodes-per-epoch` was discarded on
    every path. It went unnoticed because the value everyone passed *was* 10000, and because
    the tests above construct capabilities from a stub trainer and never traverse
    `AdaptiveTrainer` at all. This one does.
    """
    import torch

    from trainers.AdaptiveTrainer import AdaptiveTrainer

    class _Graph:
        def get_nodes(self):
            return []

    class _Manager:
        def get_graph(self):
            return _Graph()

    trainer = AdaptiveTrainer(
        graphmanager=_Manager(), models=[], device=torch.device('cpu'),
        attribute_metadata=None, loss_fn=torch.nn.BCEWithLogitsLoss(),
        max_nodes_per_epoch=777,
    )
    assert trainer.max_nodes_per_epoch == 777, "an explicit budget must not be overwritten"

    default = AdaptiveTrainer(
        graphmanager=_Manager(), models=[], device=torch.device('cpu'),
        attribute_metadata=None, loss_fn=torch.nn.BCEWithLogitsLoss(),
    )
    assert default.max_nodes_per_epoch == 10000, "the historical default must be unchanged"


def test_the_eagerly_built_capability_sees_the_budget():
    """Ordering, which is what actually broke this twice.

    `CapabilityManager.__init__` constructs `BasicTrainingCapability(trainer)` immediately,
    and that capability reads `trainer.max_nodes_per_epoch` in its own `__init__`. So the
    assignment has to happen *before* `CapabilityManager` is constructed. Setting it after
    left the basic path on its private 5000 default while the lazily-built DQN path saw
    10000 -- reintroducing the 2x asymmetry, and silently: both numbers are plausible.
    """
    import torch

    from trainers.AdaptiveTrainer import AdaptiveTrainer

    class _Graph:
        def get_nodes(self):
            return []

    class _Manager:
        def get_graph(self):
            return _Graph()

    trainer = AdaptiveTrainer(
        graphmanager=_Manager(), models=[], device=torch.device('cpu'),
        attribute_metadata=None, loss_fn=torch.nn.BCEWithLogitsLoss(),
        max_nodes_per_epoch=4321,
    )
    basic = trainer.capabilities.basic_training_capability
    assert basic.max_nodes_per_epoch == 4321, \
        "the eagerly-built basic capability must see the requested budget"
