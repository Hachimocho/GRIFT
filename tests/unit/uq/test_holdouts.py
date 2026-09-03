"""Held-out source groups: `evaluation/uq/holdouts.py`.

The protocol's correctness rests on four things, each of which produces a
plausible-looking wrong number if broken:

* held-out groups leave *train and val*, but test nodes are only **labelled** -- drop
  them and the held-out run's ID number is computed over a different sample set than
  the control's, so the two are not comparable;
* val is filtered too, because val picks the checkpoint -- leaving held-out generators
  in val selects the checkpoint that best fits data the experiment says was never seen;
* a real source can never be held out -- it is the entire negative class;
* the OOD partition is all-fake, so classification metrics there are refused and the
  shifted-classification partition (OOD fakes + ID reals) is what keeps them defined.
"""

import pandas as pd
import pytest

from evaluation.uq.holdouts import (
    FAKE_GROUP_COUNTS,
    HOLDOUTS,
    REAL_GROUPS,
    HoldoutSpec,
    apply_holdout,
    control_id,
    get_holdout,
    node_group,
    partition_records,
    shifted_classification_frame,
    summarize_available,
)

DATA_ROOT = "/shared/datasets/ai-face/ai-face"


class FakeNode:
    """Enough of an AttributeNode for the holdout filter."""

    def __init__(self, relative, label):
        self.node_id = f"{DATA_ROOT}{relative}"
        self.label = label
        self.attributes = {}
        self.split = "test"


def make_nodes(specs):
    return [FakeNode(relative, label) for relative, label in specs]


#: A miniature population spanning a real source, a mixed video source's two halves,
#: a diffusion generator, and a GAN.
POPULATION = [
    ("/FFHQ/0.png", 0),
    ("/FFHQ/1.png", 0),
    ("/wiki/00/2.jpg", 0),
    ("/dfdc/real/3.png", 0),
    ("/dfdc/fake/4.png", 1),
    ("/ff++/crop_img/5.png", 1),
    ("/ff++/real/6.png", 0),
    ("/StableDiffusion1.5/7.png", 1),
    ("/latent_diffusion/8.png", 1),
    ("/ProGAN/9.png", 1),
    ("/DCFACe/10/11.jpg", 1),
]


# --------------------------------------------------------------------------- #
# Group parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "relative,expected",
    [
        ("/FFHQ/0.png", "FFHQ"),
        ("/wiki/00/2.jpg", "wiki"),
        ("/ProGAN/9.png", "ProGAN"),
        ("/DCFACe/10/11.jpg", "DCFACe"),
        # Mixed sources keep their second component: without it, celebdf's reals and
        # fakes are one group and holding out "the celebdf fakes" would also remove
        # 9,309 real training images.
        ("/dfdc/real/3.png", "dfdc/real"),
        ("/dfdc/fake/4.png", "dfdc/fake"),
        ("/ff++/crop_img/5.png", "ff++/crop_img"),
        ("/celebdf/real/1.png", "celebdf/real"),
        ("/celebdf/crop_img/1.png", "celebdf/crop_img"),
    ],
)
def test_node_groups_are_parsed_from_the_path(relative, expected):
    assert node_group(FakeNode(relative, 0), DATA_ROOT) == expected


def test_the_colon_in_a_group_name_survives_parsing():
    """`taming_transformer:VQGAN` is a real directory name."""
    node = FakeNode("/taming_transformer:VQGAN/1.png", 1)
    assert node_group(node, DATA_ROOT) == "taming_transformer:VQGAN"


# --------------------------------------------------------------------------- #
# Spec validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("holdout_id", sorted(HOLDOUTS))
def test_every_shipped_holdout_validates(holdout_id):
    HOLDOUTS[holdout_id].validate()


@pytest.mark.parametrize("holdout_id", sorted(HOLDOUTS))
def test_every_shipped_holdout_removes_and_labels_something(holdout_id):
    counts = HOLDOUTS[holdout_id].counts()
    assert counts["train"] > 0
    assert counts["test"] > 0


@pytest.mark.parametrize("holdout_id", sorted(HOLDOUTS))
def test_no_shipped_holdout_touches_a_real_source(holdout_id):
    """The single most important invariant: reals are the whole negative class."""
    assert not (HOLDOUTS[holdout_id].groups & REAL_GROUPS)


def test_holding_out_a_real_source_is_refused():
    spec = HoldoutSpec("bad_real", {"FFHQ"}, "should not be allowed")
    with pytest.raises(ValueError, match="negative class"):
        spec.validate()


def test_holding_out_a_mixed_sources_real_half_is_refused():
    spec = HoldoutSpec("bad_mixed", {"dfdc/real"}, "should not be allowed")
    with pytest.raises(ValueError, match="negative class"):
        spec.validate()


def test_an_unknown_group_is_refused():
    """Otherwise the holdout silently removes nothing and looks like a null result."""
    spec = HoldoutSpec("typo", {"StableDifusion1.5"}, "typo in the group name")
    with pytest.raises(ValueError, match="not present in the dataset"):
        spec.validate()


def test_an_empty_holdout_is_refused():
    with pytest.raises(ValueError, match="no groups"):
        HoldoutSpec("empty", set(), "nothing").validate()


def test_get_holdout_resolves_none():
    for value in (None, "", "none", "None"):
        assert get_holdout(value) is None


def test_get_holdout_rejects_an_unknown_id():
    with pytest.raises(ValueError, match="unknown holdout"):
        get_holdout("H99_nonsense")


def test_the_cache_component_is_filename_safe():
    """`taming_transformer:VQGAN` would otherwise put a colon in a cache filename."""
    spec = HoldoutSpec("weird:id/with spaces", {"ProGAN"}, "x")
    component = spec.cache_component()
    for character in (":", "/", " "):
        assert character not in component


# --------------------------------------------------------------------------- #
# Measured counts
# --------------------------------------------------------------------------- #

def test_the_headline_holdout_counts_match_what_was_measured():
    """H1 is the paper's primary shift cell, so its size is pinned.

    Measured from the *corrected* manifests: 45,100 train rows removed and 22,916 test
    rows labelled ood. A change here means either the manifests moved or a group was
    added to the spec.
    """
    counts = HOLDOUTS["H1_diffusion_unseen"].counts()
    assert counts == {"train": 45100, "val": 19284, "test": 22916}


def test_the_video_holdout_counts_match_what_was_measured():
    counts = HOLDOUTS["H4_video_deepfakes"].counts()
    assert counts == {"train": 129578, "val": 55119, "test": 65973}


def test_the_small_holdout_is_deliberately_tiny():
    """H5 exists to exercise the small-N and confidence-interval paths."""
    assert HOLDOUTS["H5_commercial"].counts()["test"] == 159


def test_no_fake_group_is_also_listed_as_real():
    assert not (set(FAKE_GROUP_COUNTS) & REAL_GROUPS)


def test_there_are_six_real_groups():
    """FFHQ, wiki, and the real halves of the four mixed video sets."""
    assert len(REAL_GROUPS) == 6


# --------------------------------------------------------------------------- #
# Applying a holdout
# --------------------------------------------------------------------------- #

@pytest.fixture
def population():
    return (make_nodes(POPULATION), make_nodes(POPULATION), make_nodes(POPULATION))


def test_no_holdout_labels_everything_id(population):
    train, val, test = population
    kept_train, kept_val, kept_test, stats = apply_holdout(
        train, val, test, None, DATA_ROOT, verbose=False
    )
    assert len(kept_train) == len(train)
    assert len(kept_val) == len(val)
    assert {node.attributes["domain"] for node in kept_test} == {"id"}
    assert stats["test_ood"] == 0


def test_held_out_groups_leave_train(population):
    train, val, test = population
    spec = HOLDOUTS["H1_diffusion_unseen"]
    kept_train, _kept_val, _kept_test, stats = apply_holdout(
        train, val, test, spec, DATA_ROOT, verbose=False
    )
    groups = {node_group(node, DATA_ROOT) for node in kept_train}
    assert not (groups & spec.groups)
    assert stats["train_removed"] == 2  # StableDiffusion1.5 and latent_diffusion


def test_held_out_groups_leave_val_too(population):
    """Val selects the checkpoint, so leaving OOD data in val is a leak.

    It would pick whichever epoch happens to fit the held-out generators best, which
    inflates the OOD number for a reason that has nothing to do with the method.
    """
    train, val, test = population
    spec = HOLDOUTS["H1_diffusion_unseen"]
    _kept_train, kept_val, _kept_test, stats = apply_holdout(
        train, val, test, spec, DATA_ROOT, verbose=False
    )
    groups = {node_group(node, DATA_ROOT) for node in kept_val}
    assert not (groups & spec.groups)
    assert stats["val_removed"] == 2


def test_test_nodes_are_labelled_not_dropped(population):
    """The whole set must survive, or held-out and control runs are incomparable."""
    train, val, test = population
    original = len(test)
    _t, _v, kept_test, stats = apply_holdout(
        train, val, test, HOLDOUTS["H1_diffusion_unseen"], DATA_ROOT, verbose=False
    )
    assert len(kept_test) == original
    assert stats["test_ood"] == 2
    assert stats["test_id"] == original - 2


def test_the_domain_label_lands_on_the_right_nodes(population):
    train, val, test = population
    spec = HOLDOUTS["H1_diffusion_unseen"]
    _t, _v, kept_test, _stats = apply_holdout(
        train, val, test, spec, DATA_ROOT, verbose=False
    )
    for node in kept_test:
        expected = "ood" if node_group(node, DATA_ROOT) in spec.groups else "id"
        assert node.attributes["domain"] == expected


def test_the_ood_partition_is_all_fake(population):
    """Which is why classification metrics there must be refused, not computed."""
    train, val, test = population
    _t, _v, kept_test, _stats = apply_holdout(
        train, val, test, HOLDOUTS["H1_diffusion_unseen"], DATA_ROOT, verbose=False
    )
    ood = [node for node in kept_test if node.attributes["domain"] == "ood"]
    assert ood
    assert {node.label for node in ood} == {1}


def test_the_class_prior_shift_is_reported(population):
    """The number that makes a paired control necessary rather than optional."""
    train, val, test = population
    _t, _v, _te, stats = apply_holdout(
        train, val, test, HOLDOUTS["H1_diffusion_unseen"], DATA_ROOT, verbose=False
    )
    assert stats["train_real_fraction"] > stats["train_real_fraction_before"]
    # 5 real of 11 before, 5 of 9 after.
    assert stats["train_real_fraction_before"] == pytest.approx(5 / 11)
    assert stats["train_real_fraction"] == pytest.approx(5 / 9)


def test_the_real_halves_of_video_sets_stay_in_training(population):
    """H4's design point: only the positive class shifts."""
    train, val, test = population
    _kept_train, _v, _te, _stats = apply_holdout(
        train, val, test, HOLDOUTS["H4_video_deepfakes"], DATA_ROOT, verbose=False
    )
    groups = {node_group(node, DATA_ROOT) for node in _kept_train}
    assert "dfdc/real" in groups
    assert "ff++/real" in groups
    assert "dfdc/fake" not in groups
    assert "ff++/crop_img" not in groups


def test_a_holdout_matching_nothing_warns(population, capsys):
    """Silence here would look like "the holdout had no effect", a plausible result."""
    train, val, test = population
    spec = HoldoutSpec("H_absent", {"MMD_GAN_CelebA"}, "not in the fixture")
    apply_holdout(train, val, test, spec, DATA_ROOT, verbose=True)
    output = capsys.readouterr().out
    assert "WARNING: no test node matched" in output


def test_a_wrong_data_root_is_visible(population, capsys):
    """A bad root makes every group parse as the wrong component, matching nothing."""
    train, val, test = population
    apply_holdout(train, val, test, HOLDOUTS["H1_diffusion_unseen"],
                  "/completely/wrong/root", verbose=True)
    assert "WARNING: no test node matched" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Record partitions
# --------------------------------------------------------------------------- #

def make_records():
    return pd.DataFrame({
        "record_id": range(6),
        "domain": ["id", "id", "id", "ood", "ood", "ood"],
        "label": [0, 1, 0, 1, 1, 1],
        "prob": [0.2, 0.8, 0.3, 0.6, 0.4, 0.7],
    })


def test_partition_splits_on_domain():
    id_frame, ood_frame = partition_records(make_records())
    assert list(id_frame["record_id"]) == [0, 1, 2]
    assert list(ood_frame["record_id"]) == [3, 4, 5]


def test_partition_of_a_frame_with_no_domain_column():
    frame = make_records().drop(columns=["domain"])
    id_frame, ood_frame = partition_records(frame)
    assert len(id_frame) == 6
    assert len(ood_frame) == 0


def test_shifted_classification_restores_both_classes():
    """OOD fakes plus ID reals: accuracy is defined again without reusing seen fakes."""
    frame = shifted_classification_frame(make_records())
    assert set(frame["label"]) == {0, 1}
    # ID reals (records 0 and 2) plus all three OOD fakes.
    assert sorted(frame["record_id"]) == [0, 2, 3, 4, 5]


def test_shifted_classification_excludes_in_distribution_fakes():
    """Including them would dilute the shift with data the model trained on."""
    frame = shifted_classification_frame(make_records())
    seen_fakes = frame[(frame["domain"] == "id") & (frame["label"] == 1)]
    assert seen_fakes.empty


def test_shifted_classification_is_ordered_by_record_id():
    frame = shifted_classification_frame(make_records())
    assert list(frame["record_id"]) == sorted(frame["record_id"])


# --------------------------------------------------------------------------- #
# The paired control
# --------------------------------------------------------------------------- #

def test_every_holdout_has_a_distinct_control_id():
    ids = {control_id(holdout_id) for holdout_id in HOLDOUTS}
    assert len(ids) == len(HOLDOUTS)
    assert None not in ids


def test_no_holdout_has_a_control():
    assert control_id(None) is None
    assert control_id("none") is None


def test_summarize_lists_every_holdout(tmp_path):
    text = summarize_available(tmp_path / "holdouts.txt")
    for holdout_id in HOLDOUTS:
        assert holdout_id in text
    assert (tmp_path / "holdouts.txt").read_text().strip() == text.strip()
