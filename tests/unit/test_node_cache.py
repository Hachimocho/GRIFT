"""The node cache: `save_cached_nodes`, `load_cached_nodes`, and the cache-hit path.

Nothing tested these before, which is how `web_ui/app.py` shipped a writer producing
`{split: [nodes]}` -- a shape `load_cached_nodes` matches none of, so it returns `None`,
the caller reads that as "no cache", and every run silently paid a full 10-minute
dataset load. A rejected cache is indistinguishable from an absent one in the logs.

The reader is deliberately tolerant of three historical formats. That tolerance is
load-bearing (there are caches on disk in all three), so it is pinned here rather than
left to be discovered.
"""

import os

import dill
import pytest

from test_helpers.data_graph_utils import (
    load_and_prepare_data_splits,
    load_cached_nodes,
    save_cached_nodes,
)
from tests.helpers.factories import make_attr_nodes
from tests.helpers.node_cache import write_synthetic_node_cache

SPLITS = ("train", "val", "test")


@pytest.fixture
def node_lists():
    """Distinct nodes per split, with node_ids that identify their split."""
    lists = {}
    for offset, split in enumerate(SPLITS):
        nodes = make_attr_nodes(12, split=split)
        for index, node in enumerate(nodes):
            node.node_id = f"{split}/{offset * 100 + index}"
        lists[split] = nodes
    return lists


@pytest.fixture
def written_cache(node_lists, tmp_path):
    path = str(tmp_path / "node_cache" / "cached_nodes.pkl")
    save_cached_nodes(
        node_lists["train"], node_lists["val"], node_lists["test"], path,
        target_num_nodes=6,
    )
    return path


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #

def test_every_split_round_trips(written_cache, node_lists):
    for split in SPLITS:
        loaded = load_cached_nodes(written_cache, split)
        assert [node.node_id for node in loaded] == [
            node.node_id for node in node_lists[split]
        ], f"{split} did not round-trip in order"


def test_splits_do_not_bleed_into_each_other(written_cache):
    """A cache keyed per split must not hand back another split's nodes.

    The reader's second fallback returns a *split-agnostic* 'full' set, so a
    malformed-enough cache silently gives train nodes for a test request. Guard the
    happy path against that.
    """
    for split in SPLITS:
        loaded = load_cached_nodes(written_cache, split)
        assert {node.split for node in loaded} == {split}


def test_attributes_and_embeddings_survive(written_cache, node_lists):
    """dill must preserve the numpy embedding and the np.int64 demographics.

    The embedding is what the dataloaders' edge construction and graph-distance
    uncertainty both read; if it round-tripped as a list or a float64 copy, downstream
    dtype assumptions would shift without anything failing here.
    """
    import numpy as np

    loaded = load_cached_nodes(written_cache, "train")
    original = node_lists["train"]
    for before, after in zip(original, loaded):
        assert set(after.attributes) == set(before.attributes)
        embedding = after.attributes["face_embedding"]
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == before.attributes["face_embedding"].dtype
        assert np.array_equal(embedding, before.attributes["face_embedding"])
        gender = after.attributes["Ground Truth Gender"]
        assert isinstance(gender, np.integer)
        assert gender == before.attributes["Ground Truth Gender"]


def test_balanced_view_is_the_requested_size(written_cache):
    for split in SPLITS:
        assert len(load_cached_nodes(written_cache, split, balanced=True)) == 6


def test_balanced_view_is_a_subset_of_full(written_cache):
    for split in SPLITS:
        full = {node.node_id for node in load_cached_nodes(written_cache, split)}
        balanced = {
            node.node_id
            for node in load_cached_nodes(written_cache, split, balanced=True)
        }
        assert balanced <= full


def test_balancing_is_deterministic(node_lists, tmp_path):
    """Two caches written from the same nodes select the same balanced subset.

    `balance_nodes_by_subgroup` seeds itself from an md5 of the sorted node ids
    specifically so this holds without depending on the global RNG.
    """
    selections = []
    for run in range(2):
        path = str(tmp_path / f"run{run}" / "cached_nodes.pkl")
        save_cached_nodes(
            node_lists["train"], node_lists["val"], node_lists["test"], path,
            target_num_nodes=6,
        )
        selections.append(
            [node.node_id for node in load_cached_nodes(path, "train", balanced=True)]
        )
    assert selections[0] == selections[1]


def test_unbalanceable_target_falls_back_to_full(node_lists, tmp_path):
    """A target larger than a subgroup must not lose the split.

    `save_cached_nodes` catches the ValueError and stores the full list as the balanced
    view. That fallback matters: dropping the split would make the cache unloadable,
    and raising would make an over-large --cached-nodes fatal rather than merely
    unbalanced.
    """
    path = str(tmp_path / "big_target" / "cached_nodes.pkl")
    save_cached_nodes(
        node_lists["train"], node_lists["val"], node_lists["test"], path,
        target_num_nodes=10_000,
    )
    for split in SPLITS:
        balanced = load_cached_nodes(path, split, balanced=True)
        assert len(balanced) == len(node_lists[split])


# --------------------------------------------------------------------------- #
# The three formats the reader tolerates
# --------------------------------------------------------------------------- #

def write_raw(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        dill.dump(payload, handle)
    return path


def test_current_format_is_what_the_writer_produces(written_cache):
    with open(written_cache, "rb") as handle:
        payload = dill.load(handle)
    assert set(payload) == set(SPLITS)
    for split in SPLITS:
        assert set(payload[split]) == {"full", "balanced"}


def test_legacy_splitless_dict_still_loads(node_lists, tmp_path):
    """Format 2: one overall full/balanced pair, no per-split keys."""
    path = write_raw(
        str(tmp_path / "legacy" / "cached_nodes.pkl"),
        {"full": node_lists["train"], "balanced": node_lists["train"][:4]},
    )
    assert len(load_cached_nodes(path, "test")) == 12
    assert len(load_cached_nodes(path, "test", balanced=True)) == 4


def test_legacy_bare_list_still_loads(node_lists, tmp_path):
    """Format 3: a plain list of nodes."""
    path = write_raw(str(tmp_path / "verylegacy" / "cached_nodes.pkl"),
                     node_lists["train"])
    assert len(load_cached_nodes(path, "train")) == 12
    # A list cannot express a balanced view, so the reader returns the full list.
    assert len(load_cached_nodes(path, "train", balanced=True)) == 12


def test_the_shape_web_ui_used_to_write_is_rejected(node_lists, tmp_path):
    """`{split: [nodes]}` matches no accepted format -- the regression this pins.

    Kept as a test rather than deleted with the bug: the reader returning `None` here
    (instead of raising) is exactly why the bug was invisible, so if anyone ever makes
    this shape loadable, they should have to notice this test.
    """
    path = write_raw(
        str(tmp_path / "webui" / "cached_nodes.pkl"),
        {split: nodes for split, nodes in node_lists.items()},
    )
    assert load_cached_nodes(path, "train") is None


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_cached_nodes(str(tmp_path / "absent.pkl"), "train") is None


def test_a_corrupt_file_is_not_an_error(tmp_path):
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"not a pickle at all")
    assert load_cached_nodes(str(path), "train") is None


def test_an_unrecognized_structure_is_not_an_error(tmp_path):
    path = write_raw(str(tmp_path / "odd" / "cached_nodes.pkl"), {"nodes": 3})
    assert load_cached_nodes(path, "train") is None


# --------------------------------------------------------------------------- #
# The cache-hit path through load_and_prepare_data_splits
# --------------------------------------------------------------------------- #

class Args:
    """The subset of the CLI namespace the cache path reads."""

    def __init__(self, cache_file, data_root, **overrides):
        self.use_cached = True
        self.cache_file = cache_file
        self.cached_nodes = 6
        self.dynamic_cache_detection = False
        self.fair_train = False
        self.fair_test = False
        self.cache_nodes = False
        self.atr_threshold = 2
        # `load_and_prepare_data_splits` resolves the root *before* consulting the
        # cache and raises FileNotFoundError if resolution fails, so even a pure
        # cache-hit test needs a valid-looking root. The tiny fixture root is one, and
        # using it keeps these tests independent of the real dataset.
        self.data_root = data_root
        for name, value in overrides.items():
            setattr(self, name, value)


@pytest.fixture
def no_dataset(monkeypatch):
    """Make constructing AIFaceDataset a loud failure. Returns the call log.

    A cache-hit test can then assert the dataset was never built, and a fall-through
    test can assert it was -- neither of which a timing assertion could do reliably.
    """
    import test_helpers.data_graph_utils as module

    calls = []

    def explode(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("direct dataset load reached")

    monkeypatch.setattr(module, "AIFaceDataset", explode)
    return calls


def test_cache_hit_never_touches_the_dataset(written_cache, tiny_ai_face_root,
                                             no_dataset):
    """The whole point: a warm cache must not construct AIFaceDataset."""
    train, val, test = load_and_prepare_data_splits(
        Args(written_cache, tiny_ai_face_root), tiny_ai_face_root
    )[:3]
    assert (len(train), len(val), len(test)) == (12, 12, 12)
    assert not no_dataset, "AIFaceDataset was constructed despite a cache hit"


def test_cache_hit_aliases_full_to_the_loaded_lists(written_cache, tiny_ai_face_root,
                                                   no_dataset):
    """`*_nodes_full` is the *same object* as `*_nodes` on the cache-hit path.

    Documented here because it is surprising and load-bearing: callers that mutate a
    "full" list expecting the balanced one to be untouched would corrupt both. With
    `--fair-train`, `train_nodes` is the balanced view, so `train_nodes_full` is the
    balanced view too -- the full population is not returned at all.
    """
    train, val, test, train_full, val_full, test_full, _ = (
        load_and_prepare_data_splits(
            Args(written_cache, tiny_ai_face_root), tiny_ai_face_root
        )
    )
    assert train_full is train
    assert val_full is val
    assert test_full is test


def test_fair_flags_select_the_balanced_view(written_cache, tiny_ai_face_root,
                                             no_dataset):
    args = Args(written_cache, tiny_ai_face_root, fair_train=True, fair_test=True)
    train, val, test = load_and_prepare_data_splits(args, tiny_ai_face_root)[:3]
    assert (len(train), len(val), len(test)) == (6, 6, 6)


def test_dynamic_detection_reads_the_count_off_the_cache(written_cache,
                                                         tiny_ai_face_root,
                                                         no_dataset):
    args = Args(written_cache, tiny_ai_face_root, dynamic_cache_detection=True,
                cached_nodes=999)
    load_and_prepare_data_splits(args, tiny_ai_face_root)
    assert args.cached_nodes == 12, "should have detected the full train count"

    args = Args(written_cache, tiny_ai_face_root, dynamic_cache_detection=True,
                cached_nodes=999, fair_train=True, fair_test=True)
    load_and_prepare_data_splits(args, tiny_ai_face_root)
    assert args.cached_nodes == 6, "should have detected the balanced train count"


def test_a_rejected_cache_clears_use_cached(node_lists, tmp_path, tiny_ai_face_root,
                                            no_dataset):
    """A cache the reader refuses must fall through to a direct load, not proceed.

    This is the observable symptom of the web_ui bug, and it is why that bug cost
    minutes per run rather than failing.
    """
    path = write_raw(
        str(tmp_path / "webui" / "cached_nodes.pkl"),
        {split: nodes for split, nodes in node_lists.items()},
    )
    args = Args(path, tiny_ai_face_root)
    with pytest.raises(RuntimeError, match="direct dataset load reached"):
        load_and_prepare_data_splits(args, tiny_ai_face_root)
    assert no_dataset, "should have attempted a direct load"
    assert args.use_cached is False


def test_a_missing_cache_file_falls_through(tmp_path, tiny_ai_face_root, no_dataset):
    args = Args(str(tmp_path / "absent.pkl"), tiny_ai_face_root)
    with pytest.raises(RuntimeError, match="direct dataset load reached"):
        load_and_prepare_data_splits(args, tiny_ai_face_root)
    assert args.use_cached is False


# --------------------------------------------------------------------------- #
# The test-fixture builder
# --------------------------------------------------------------------------- #

def test_the_fixture_builder_writes_a_loadable_cache(tiny_node_cache):
    for split, expected in (("train", 120), ("val", 40), ("test", 40)):
        loaded = load_cached_nodes(tiny_node_cache, split)
        assert len(loaded) == expected
        assert {node.split for node in loaded} == {split}


def test_the_fixture_builder_produces_real_image_data(tiny_node_cache):
    """Nodes carry a real ImageFileData over a real file, so eval loops can run."""
    nodes = load_cached_nodes(tiny_node_cache, "val")
    for node in nodes[:5]:
        assert node.data is not None
        assert os.path.isfile(node.node_id)
        image = node.data.load_data()
        assert image is not None and image.shape[2] == 3


def test_the_fixture_builder_spreads_nodes_across_sources(tiny_node_cache):
    """Both classes and several source folders appear, as in the real data."""
    nodes = load_cached_nodes(tiny_node_cache, "train")
    assert {node.label for node in nodes} == {0, 1}
    sources = {node.node_id.split(os.sep)[-2] for node in nodes}
    assert len(sources) >= 3


def test_the_fixture_builder_is_reproducible(tmp_path):
    """Same arguments, same node ids and labels -- a fixture must not drift."""
    ids = []
    for run in range(2):
        path = write_synthetic_node_cache(
            tmp_path / f"c{run}" / "cached_nodes.pkl", tmp_path / f"i{run}",
            n_train=6, n_val=3, n_test=3, embedding_dim=8,
        )
        nodes = load_cached_nodes(path, "train")
        ids.append([(os.path.basename(node.node_id), node.label) for node in nodes])
    assert ids[0] == ids[1]
