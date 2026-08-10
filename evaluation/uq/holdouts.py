"""Held-out source groups: the primary distribution-shift axis.

A holdout removes one or more generator families from **training** and labels those
same families `domain='ood'` at **evaluation**. Test nodes are never dropped -- they are
labelled -- because dropping them would change which samples the ID number is computed
over, making held-out and control runs incomparable.

Three properties of this dataset make the protocol non-obvious, and getting any of them
wrong produces a number that looks fine:

**A holdout of generators is all-fake.** Every fake-only source is `Target=1`, so the
OOD partition has one class. Accuracy, ECE, Brier, and AUROC-of-error are all undefined
or meaningless there, and an AUROC computed on a single-class set is not a number worth
printing. So each holdout supports two *separate* evaluations: OOD **detection** (does
uncertainty separate held-out from in-distribution? single-class is fine, that is the
point) and **shifted classification** (held-out fakes plus ID reals, so both classes are
present and accuracy stays defined).

**Holding out fakes shifts the class prior.** The corrected dataset is 13.1% real
overall; remove 45k fakes from training and the training prior moves. Every
prior-sensitive metric moves with it. So a holdout run is only interpretable against a
**paired ID-only control** trained on the same reduced set -- otherwise "OOD
degradation" is indistinguishable from "trained on less data".

**Never hold out a real source.** The six real sources are 13.1% of the data and the
entire negative class. Removing one would not create a shift experiment, it would
create a broken classifier.

Group ids come from directory names, one of which (`taming_transformer:VQGAN`) contains
a colon -- so anything that reaches a filename or a cache key goes through
`sanitize_key_component`.
"""

import os

from evaluation.uq.records import parse_source_group

#: The six real corpora, per the corrected labelling. Holding any of these out is
#: refused: they are the whole negative class.
REAL_GROUPS = frozenset({
    "FFHQ", "wiki", "celebdf/real", "ff++/real", "dfd/real", "dfdc/real",
})

#: Every fake-only group present in the corrected manifests, with per-split row counts
#: measured from disk after the relabel. Recorded so a spec naming a group that does
#: not exist fails at plan time rather than silently holding out nothing.
FAKE_GROUP_COUNTS = {
    "AttGAN": (2398, 1008, 1178),
    "CommercialTools": (305, 132, 159),
    "DCFACe": (207954, 88871, 105978),
    "MMD_GAN_CelebA": (374, 180, 220),
    "MSG_STYLE_GAN": (386, 173, 202),
    "Palette": (2291, 1009, 1223),
    "ProGAN": (39254, 16752, 19979),
    "STARGAN": (2256, 976, 1046),
    "STGAN_CelebA": (375, 182, 193),
    "STYLEGAN": (3925, 1680, 2061),
    "StableDiffusion1.5": (7113, 3037, 3580),
    "StableDiffusion_Inpainting": (8234, 3528, 4188),
    "StyleGAN2_FFHQ": (46251, 20016, 23827),
    "celebdf/crop_img": (60920, 26022, 31052),
    "dfd/fake": (12405, 5184, 6323),
    "dfdc/fake": (14731, 6119, 7600),
    "ff++/crop_img": (41522, 17794, 20998),
    "latent_diffusion": (7846, 3403, 3847),
    "stylegan3": (10466, 4559, 5279),
    "taming_transformer:VQGAN": (19616, 8307, 10078),
}

SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


class HoldoutSpec:
    """One held-out family of source groups."""

    __slots__ = ("holdout_id", "groups", "rationale")

    def __init__(self, holdout_id, groups, rationale):
        self.holdout_id = holdout_id
        self.groups = frozenset(groups)
        self.rationale = rationale

    def counts(self):
        """Rows held out per split, from the measured table."""
        totals = [0, 0, 0]
        for group in sorted(self.groups):
            for index, value in enumerate(FAKE_GROUP_COUNTS[group]):
                totals[index] += value
        return {"train": totals[0], "val": totals[1], "test": totals[2]}

    def validate(self):
        """Refuse a spec that cannot produce a meaningful experiment."""
        problems = []
        if not self.groups:
            problems.append("holds out no groups")

        overlap = self.groups & REAL_GROUPS
        if overlap:
            problems.append(
                f"holds out real source(s) {sorted(overlap)}. Real sources are the "
                f"entire negative class ({len(REAL_GROUPS)} groups, 13.1% of the "
                f"data); removing one breaks the classifier rather than shifting it"
            )

        unknown = self.groups - set(FAKE_GROUP_COUNTS) - REAL_GROUPS
        if unknown:
            problems.append(
                f"names group(s) not present in the dataset: {sorted(unknown)}. Known "
                f"fake groups: {sorted(FAKE_GROUP_COUNTS)}"
            )

        if not problems:
            counts = self.counts()
            if counts["test"] == 0:
                problems.append("holds out no test rows, so there is no OOD set")

        if problems:
            raise ValueError(
                f"holdout {self.holdout_id!r} is invalid:\n  - " + "\n  - ".join(problems)
            )
        return self

    def cache_component(self):
        """The holdout's contribution to a graph cache key."""
        from test_helpers.cache_keys import sanitize_key_component

        return sanitize_key_component(self.holdout_id)

    def is_ood_group(self, group):
        return group in self.groups

    def describe(self):
        counts = self.counts()
        return (
            f"{self.holdout_id}: {len(self.groups)} group(s), "
            f"{counts['train']:,} train rows removed, {counts['test']:,} test rows "
            f"labelled ood -- {self.rationale}"
        )

    def __repr__(self):
        return f"HoldoutSpec({self.holdout_id!r}, {len(self.groups)} groups)"


#: The recommended protocols. H1 and H4 are the headline pair; the rest are secondary.
HOLDOUTS = {
    spec.holdout_id: spec
    for spec in (
        HoldoutSpec(
            "H1_diffusion_unseen",
            {"StableDiffusion1.5", "StableDiffusion_Inpainting", "latent_diffusion",
             "Palette", "taming_transformer:VQGAN"},
            "train on GANs, test an unseen generation paradigm -- the headline "
            "question, since a detector that only knows GAN artifacts is the failure "
            "mode practitioners actually hit",
        ),
        HoldoutSpec(
            "H4_video_deepfakes",
            {"celebdf/crop_img", "ff++/crop_img", "dfd/fake", "dfdc/fake"},
            "classic cross-dataset generalization. The real/ halves stay in training, "
            "so the negative distribution is untouched and only the positive class "
            "shifts",
        ),
        HoldoutSpec(
            "H2_stylegan_family",
            {"STYLEGAN", "StyleGAN2_FFHQ", "stylegan3", "MSG_STYLE_GAN"},
            "within-paradigm generalization across one architecture family",
        ),
        HoldoutSpec(
            "H3_celeba_gans",
            {"AttGAN", "STARGAN", "STGAN_CelebA", "MMD_GAN_CelebA"},
            "attribute-editing GANs, which manipulate rather than synthesize",
        ),
        HoldoutSpec(
            "H5_commercial",
            {"CommercialTools"},
            "159 OOD test rows: deliberately tiny, so it exercises the small-N and "
            "confidence-interval paths that a large holdout never reaches",
        ),
    )
}


def get_holdout(holdout_id):
    """Look up and validate a holdout by id. ``None``/``'none'`` means no holdout."""
    if holdout_id in (None, "", "none", "None"):
        return None
    if holdout_id not in HOLDOUTS:
        raise ValueError(
            f"unknown holdout {holdout_id!r}. Available: {sorted(HOLDOUTS)}, or 'none'"
        )
    return HOLDOUTS[holdout_id].validate()


def node_group(node, data_root=None):
    """The source group of a node, from its ``node_id`` path."""
    from evaluation.uq.records import relative_path

    node_id = getattr(node, "node_id", node)
    _top, group = parse_source_group(relative_path(node_id, data_root))
    return group


def apply_holdout(train_nodes, val_nodes, test_nodes, spec, data_root=None,
                  verbose=True):
    """Remove held-out groups from train/val; label them on test.

    Returns ``(train, val, test, stats)``. Test comes back the same length, with each
    node's ``attributes['domain']`` set to ``'id'`` or ``'ood'`` -- the records
    collector reads it from there.

    Val is filtered alongside train because val selects the checkpoint. Leaving
    held-out generators in val would pick the checkpoint that best fits data the
    experiment claims the model never saw, which is a leak that improves the OOD
    number for the wrong reason.
    """
    if spec is None:
        for node in test_nodes:
            node.attributes.setdefault("domain", "id")
        return list(train_nodes), list(val_nodes), list(test_nodes), {
            "holdout_id": None, "train_removed": 0, "val_removed": 0, "test_ood": 0,
        }

    spec.validate()

    def keep(nodes):
        kept, removed = [], 0
        for node in nodes:
            if spec.is_ood_group(node_group(node, data_root)):
                removed += 1
            else:
                kept.append(node)
        return kept, removed

    kept_train, train_removed = keep(train_nodes)
    kept_val, val_removed = keep(val_nodes)

    test_ood = 0
    for node in test_nodes:
        is_ood = spec.is_ood_group(node_group(node, data_root))
        node.attributes["domain"] = "ood" if is_ood else "id"
        test_ood += int(is_ood)

    stats = {
        "holdout_id": spec.holdout_id,
        "groups": sorted(spec.groups),
        "train_removed": train_removed,
        "val_removed": val_removed,
        "train_remaining": len(kept_train),
        "val_remaining": len(kept_val),
        "test_ood": test_ood,
        "test_id": len(test_nodes) - test_ood,
        "train_real_fraction": _real_fraction(kept_train),
        "train_real_fraction_before": _real_fraction(train_nodes),
    }

    if verbose:
        print(f"Holdout {spec.holdout_id}: removed {train_removed} train / "
              f"{val_removed} val nodes; labelled {test_ood} of {len(test_nodes)} "
              f"test nodes as ood")
        print(f"  train class prior (real fraction): "
              f"{stats['train_real_fraction_before']:.4f} -> "
              f"{stats['train_real_fraction']:.4f}")
        if test_ood == 0:
            print("  WARNING: no test node matched this holdout. Either the holdout "
                  "groups are absent from the cached node set, or data_root is wrong "
                  "so source groups are being parsed from the wrong path component.")
        if not kept_train:
            print("  WARNING: the holdout removed every training node.")

    return kept_train, kept_val, list(test_nodes), stats


def _real_fraction(nodes):
    """Fraction of nodes labelled real. The prior a holdout perturbs."""
    if not nodes:
        return 0.0
    real = sum(1 for node in nodes if int(getattr(node, "label", 1) or 0) == 0)
    return real / float(len(nodes))


def partition_records(frame):
    """``(id_frame, ood_frame)`` from a records table's ``domain`` column."""
    if "domain" not in frame.columns:
        return frame, frame.iloc[0:0]
    return frame[frame["domain"] == "id"], frame[frame["domain"] == "ood"]


def shifted_classification_frame(frame):
    """OOD fakes plus ID reals -- the partition where accuracy stays defined.

    An all-fake OOD set cannot support accuracy, ECE, or AUROC-of-error. Adding back
    the in-distribution *reals* restores both classes without reintroducing any fake
    the model was trained on, so the shifted number measures shift rather than being
    undefined.
    """
    if "domain" not in frame.columns or "label" not in frame.columns:
        return frame
    ood_fakes = frame[(frame["domain"] == "ood") & (frame["label"] == 1)]
    id_reals = frame[(frame["domain"] == "id") & (frame["label"] == 0)]
    import pandas as pd

    combined = pd.concat([ood_fakes, id_reals], ignore_index=True)
    return combined.sort_values("record_id").reset_index(drop=True)


def control_id(holdout_id):
    """The paired ID-only control's id for a holdout.

    The report refuses to print an OOD delta without this run: without it, degradation
    from shift is confounded with degradation from a smaller training set.
    """
    if holdout_id in (None, "none"):
        return None
    return f"{holdout_id}__control"


def summarize_available(path=None):
    """Human-readable table of every holdout. Used by the CLI and the docs."""
    lines = ["Available holdouts:"]
    for holdout_id in sorted(HOLDOUTS):
        lines.append("  " + HOLDOUTS[holdout_id].describe())
    text = "\n".join(lines)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(text + "\n")
    return text
