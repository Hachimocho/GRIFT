"""Framework-wide class contracts, checked by reflection.

Cheap, broad regression net: a new node/edge/traversal that forgets part of the
convention fails here rather than at runtime deep inside a training run.

Note that `tags` is currently *vestigial* -- its only two consumers in the repo
(`utils/tag_list_updater.py`, which regenerates the already-stale `docs/tags.md`,
and `utils/import_utils.get_tagged_classes_from_module`) have no callers, and the
latter is buggy: it is documented to accept a list of tags but tests
`tags in obj.tags`, so passing a list never matches. These tests assert the
convention is *followed*, not that anything enforces it.
"""

import inspect

import pytest

from tests.helpers.factories import (
    get_edge_classes, get_node_classes, get_traversal_classes,
)

FRAMEWORK_PACKAGES = ("nodes", "edges", "traversals", "dataloaders", "managers")


def _iter_classes(package_name):
    import importlib
    package = importlib.import_module(package_name)
    for name, obj in inspect.getmembers(package, inspect.isclass):
        if getattr(obj, "__module__", "").startswith(package_name + "."):
            yield name, obj


# Abstract bases declare `tags = ["none"]` as a "do not use directly" marker but
# deliberately carry no `hyperparameters` -- there is nothing to sweep on a base.
ABSTRACT_BASES = {"Traversal", "Dataloader", "GraphManager", "Dataset", "Model", "Trainer"}


@pytest.mark.parametrize("package_name", FRAMEWORK_PACKAGES)
def test_classes_declare_tags(package_name):
    classes = list(_iter_classes(package_name))
    assert classes, f"no classes discovered in {package_name}"
    for name, cls in classes:
        assert hasattr(cls, "tags"), f"{package_name}.{name} is missing `tags`"


@pytest.mark.parametrize("package_name", FRAMEWORK_PACKAGES)
def test_concrete_classes_declare_hyperparameters(package_name):
    for name, cls in _iter_classes(package_name):
        if name in ABSTRACT_BASES:
            continue
        assert hasattr(cls, "hyperparameters"), (
            f"{package_name}.{name} is missing `hyperparameters`"
        )


def test_node_classes_discovered():
    classes = get_node_classes()
    assert "Node" in classes and "AttributeNode" in classes and "RandomNode" in classes


def test_edge_classes_discovered():
    assert "Edge" in get_edge_classes()


def test_traversal_classes_expose_the_traversal_interface():
    classes = get_traversal_classes()
    assert classes, "no traversal classes discovered"
    for name, cls in classes.items():
        for method in ("traverse", "get_pointers", "reset_pointers"):
            assert callable(getattr(cls, method, None)), f"{name} is missing {method}()"


def test_traversal_discovery_is_complete():
    """Reflection must find every concrete traversal.

    The upstream branch used a four-name allowlist and silently skipped five
    classes, so half the traversals were never tested.
    """
    discovered = set(get_traversal_classes())
    expected = {
        "RandomTraversal",
        "RandomWarpTraversal",
        "RandomNoReturnTraversal",
        "RandomNoReturnWarpTraversal",
        "ComprehensiveTraversal",
        "IValueTraversal",
        "IValueTraversalSubcluster",
        "IValueTraversalClusterHop",
    }
    missing = expected - discovered
    assert not missing, f"traversal classes not discovered by reflection: {sorted(missing)}"


def test_uncertainty_package_exports():
    """The uncertainty package's public surface, so accidental removals are caught."""
    import models.uncertainty as uq

    for name in (
        "PredictionBundle",
        "BatchEnsembleBinaryHead",
        "BinaryEvidentialHead",
        "EvidentialBinaryClassificationLoss",
        "SNGPBinaryHead",
        "compute_batch_graph_uncertainty",
        "mc_dropout_predict",
    ):
        assert hasattr(uq, name), f"models.uncertainty is missing {name}"
        assert name in uq.__all__, f"{name} is not in models.uncertainty.__all__"
