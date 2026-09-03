"""`AdaptiveTrainer`'s traversal presence checks.

`if not self.current_traversal:` invokes `__len__`. `Traversal.__len__` raises
`NotImplementedError` and the four `Random*` traversals never override it, so every
training run using one died on the first epoch with a bare
"Subclass must implement __len__()" -- from a line whose visible intent is only
"is a traversal set?".

The traversals that *do* implement `__len__` had the opposite failure: a zero-length
traversal was silently indistinguishable from no traversal at all, so training would
raise "No traversal method set" for a traversal that was, in fact, set.

Both are the same fix -- `is None` -- and both are pinned here.
"""

import pytest

from tests.helpers.factories import build_ring_graph, build_traversal, get_traversal_classes


class NoLenTraversal:
    """The shape a Random* traversal presents: no `__len__`, no `__bool__`."""

    def __init__(self):
        self.calls = 0

    def __len__(self):
        raise NotImplementedError("Subclass must implement __len__()")


class EmptyTraversal:
    """A traversal that exists but has zero steps."""

    def __len__(self):
        return 0


def make_trainer(traversal, traversal_type="random"):
    """An AdaptiveTrainer with its collaborators stubbed out.

    Constructed via `__new__` so no capability manager, model, or graph is built: the
    guards under test read only `current_traversal` and `capabilities`.
    """
    from trainers.AdaptiveTrainer import AdaptiveTrainer

    trainer = AdaptiveTrainer.__new__(AdaptiveTrainer)
    trainer.current_traversal = traversal
    trainer.current_traversal_type = traversal_type
    trainer.capabilities = StubCapabilities()
    return trainer


class StubCapabilities:
    def __init__(self):
        self.enabled_capabilities = set()
        self.calls = []

    def train_with_traversal(self, traversal, epoch):
        self.calls.append((traversal, epoch))
        return {"loss": 0.5}


# --------------------------------------------------------------------------- #
# train()
# --------------------------------------------------------------------------- #

def test_train_accepts_a_traversal_without_len():
    """The regression: this raised NotImplementedError before the fix."""
    trainer = make_trainer(NoLenTraversal())
    result = trainer.train(epoch=0)
    assert result == {"loss": 0.5}
    assert len(trainer.capabilities.calls) == 1


def test_train_accepts_a_zero_length_traversal():
    """A set-but-empty traversal must reach the capability manager, not be refused."""
    traversal = EmptyTraversal()
    trainer = make_trainer(traversal)
    trainer.train(epoch=1)
    assert trainer.capabilities.calls == [(traversal, 1)]


def test_train_still_refuses_a_missing_traversal():
    trainer = make_trainer(None)
    with pytest.raises(ValueError, match="No traversal method set"):
        trainer.train(epoch=0)


def test_train_forwards_the_epoch():
    """SNGP's per-epoch precision reset depends on the epoch actually arriving."""
    trainer = make_trainer(NoLenTraversal())
    trainer.train(epoch=7)
    assert trainer.capabilities.calls[0][1] == 7


# --------------------------------------------------------------------------- #
# get_current_traversal_info()
# --------------------------------------------------------------------------- #

def test_info_works_on_a_traversal_without_len():
    trainer = make_trainer(NoLenTraversal(), traversal_type="random")
    info = trainer.get_current_traversal_info()
    assert info["type"] == "random"
    assert info["class"] == "NoLenTraversal"


def test_info_reports_a_missing_traversal():
    assert make_trainer(None).get_current_traversal_info() == "No traversal set"


# --------------------------------------------------------------------------- #
# The real traversal classes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(get_traversal_classes()))
def test_no_real_traversal_can_be_truth_tested(name):
    """Every traversal must survive being handed to `train()`.

    Parametrized over the classes by reflection rather than a list, so a new traversal
    is covered automatically. This is the assertion that would have caught the bug:
    it fails for RandomTraversal on the pre-fix code.
    """
    traversal_class = get_traversal_classes()[name]
    graph, _nodes, _edges = build_ring_graph(count=6)
    traversal = build_traversal(traversal_class, graph, num_pointers=1, num_steps=2)

    trainer = make_trainer(traversal, traversal_type=name)
    trainer.train(epoch=0)
    assert trainer.capabilities.calls[0][0] is traversal
    # And the info path, which is the other guard.
    assert trainer.get_current_traversal_info()["class"] == name
