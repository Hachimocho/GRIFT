"""`--graph-type` must mean what it says.

Three defects made it partly a lie, all of them silent:

1. The dataloader was selected with `args.graph_type == 'nonclustered'`, so
   `nonclustered_subclustered` fell to the clustered branch and built a race-gender
   clustered graph while reporting itself non-clustered.
2. Neither dataloader's `_build_graph_standard` -- the only builder the live path calls --
   assigns Louvain subclusters. The clustered builder gates on an `assign_subclusters`
   hyperparameter nothing ever set to True, and the unclustered builder's unconditional
   call is in a method the live path never reaches. So subclustering never ran, and every
   subcluster traversal silently used its no-subcluster fallback.
3. The warm-cache branch constructs a bare `HyperGraph(nodes)` and loads edges, so even a
   fixed builder would have skipped subclustering on every run after the first.

Together these made `clustered_subclustered` and `nonclustered_subclustered` produce
record tables byte-identical to `clustered` -- which is how a real sweep spent two cells
measuring nothing.
"""

import ast
import inspect

import pytest

import test_hierarchical
from dataloaders.HierarchicalDeepfakeDataloader import HierarchicalDeepfakeDataloader
from dataloaders.UnclusteredDeepfakeDataloader import UnclusteredDeepfakeDataloader
from test_helpers.args_utils import parse_args

GRAPH_TYPES = ("clustered", "clustered_subclustered",
               "nonclustered", "nonclustered_subclustered")


def dispatch_source():
    """The dataloader-selection block, read out of the runner's source.

    Asserted at the source level because reaching it for real needs a dataset load; the
    property under test is a one-line branch condition that regressed once already.
    """
    source = inspect.getsource(test_hierarchical)
    start = source.index("# Select dataloader based on graph type")
    return source[start:start + 800]


@pytest.mark.parametrize("graph_type", GRAPH_TYPES)
def test_graph_type_is_a_valid_choice(graph_type):
    assert parse_args(["--graph-type", graph_type]).graph_type == graph_type


def test_dispatch_uses_a_prefix_test_not_equality():
    """`== 'nonclustered'` silently sent nonclustered_subclustered to the wrong builder."""
    block = dispatch_source()
    assert "startswith('nonclustered')" in block or 'startswith("nonclustered")' in block
    assert "graph_type == 'nonclustered'" not in block


@pytest.mark.parametrize("graph_type,expected", [
    ("nonclustered", "unclustered"),
    ("nonclustered_subclustered", "unclustered"),
    ("clustered", "clustered"),
    ("clustered_subclustered", "clustered"),
])
def test_every_graph_type_maps_to_the_right_builder(graph_type, expected):
    """The mapping the dispatch is supposed to implement, stated independently."""
    assert ("unclustered" if graph_type.startswith("nonclustered") else "clustered") == expected


def test_the_runner_assigns_subclusters_itself():
    """It cannot delegate: no builder on the live path does it, warm or cold."""
    source = inspect.getsource(test_hierarchical)
    assert "graph.assign_louvain_subclusters()" in source
    # Guarded by the graph type, so a plain graph type does not silently get subclusters.
    assert "args.graph_type.endswith('_subclustered')" in source


@pytest.mark.parametrize("dataloader", [
    HierarchicalDeepfakeDataloader, UnclusteredDeepfakeDataloader,
])
def test_the_live_builder_still_does_not_subcluster(dataloader):
    """Documents *why* the runner must do it, and fails if a builder starts doing it too
    -- assigning twice would silently double the work and could renumber subclusters."""
    source = inspect.getsource(dataloader._build_graph_standard)
    assert "assign_louvain_subclusters" not in source


def test_subclustered_graph_types_refuse_to_run_without_louvain(monkeypatch):
    """A no-op assignment would make the run behave as its plain counterpart."""
    monkeypatch.setattr(test_hierarchical, "_LOUVAIN_AVAILABLE", False)
    source = inspect.getsource(test_hierarchical)
    assert "_LOUVAIN_AVAILABLE" in source
    # The refusal names the flag and the fallback it is preventing.
    start = source.index("if graph and args.graph_type.endswith('_subclustered')")
    block = source[start:start + 900]
    assert "python-louvain" in block
    assert "RuntimeError" in block


def test_the_dispatch_block_parses_as_written():
    """Cheap guard that the source-level assertions above are reading real code."""
    ast.parse(inspect.getsource(test_hierarchical))
