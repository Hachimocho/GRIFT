"""Per-sample record collection and persistence.

The seam test at the bottom is the important one: it drives the real
``evaluate_model`` with a fake duck-typed model and asserts that attaching a
collector leaves every pre-existing metric bit-identical. That is what makes the
benchmark safe to add to a function three live call sites depend on.
"""

import gzip
import json

import numpy as np
import pytest

from evaluation.uq.records import (
    ANNOTATION_ATTRIBUTES, PredictionRecordCollector, RecordCollectionError,
    default_meta_path, parse_source_group, read_records, relative_path, save_records,
    sha256_of_file, write_records,
)


# --------------------------------------------------------------------------- #
# Source-group parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "rel_path,expected_top,expected_group",
    [
        ("casia-webface/007489/00388746.jpg", "casia-webface", "casia-webface"),
        ("CelebA/167828.jpg", "CelebA", "CelebA"),
        ("StableDiffusion1.5/000/1.png", "StableDiffusion1.5", "StableDiffusion1.5"),
        # The mixed video sets carry real/fake in their second component.
        ("dfdc/real/0_6063_61.png", "dfdc", "dfdc/real"),
        ("celebdf/crop_img/id0_0000.png", "celebdf", "celebdf/crop_img"),
        ("ff++/real/x.png", "ff++", "ff++/real"),
        ("dfd/fake/y.png", "dfd", "dfd/fake"),
        # A colon in a real source folder name.
        ("taming_transformer:VQGAN/a.png", "taming_transformer:VQGAN", "taming_transformer:VQGAN"),
        ("", "unknown", "unknown"),
    ],
)
def test_parse_source_group(rel_path, expected_top, expected_group):
    top, group = parse_source_group(rel_path)
    assert top == expected_top
    assert group == expected_group


def test_parse_source_group_handles_leading_slash_and_backslashes():
    assert parse_source_group("/CelebA/1.png") == ("CelebA", "CelebA")
    assert parse_source_group("dfdc\\real\\1.png") == ("dfdc", "dfdc/real")


def test_relative_path_strips_the_data_root():
    assert relative_path("/data/ai-face/CelebA/1.png", "/data/ai-face") == "CelebA/1.png"
    assert relative_path("/data/ai-face/CelebA/1.png", "/data/ai-face/") == "CelebA/1.png"
    assert relative_path("CelebA/1.png", None) == "CelebA/1.png"


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def make_bundle(probabilities, uncertainty=None, features=None, logits=None):
    import torch
    from models.uncertainty import PredictionBundle

    probs = torch.tensor(probabilities, dtype=torch.float32).reshape(-1, 1)
    return PredictionBundle(
        logits=torch.logit(probs.clamp(1e-6, 1 - 1e-6)) if logits is None
        else torch.tensor(logits, dtype=torch.float32).reshape(-1, 1),
        probabilities=probs,
        features=None if features is None else torch.tensor(features, dtype=torch.float32),
        uncertainty={
            name: torch.tensor(value, dtype=torch.float32).reshape(-1, 1)
            for name, value in (uncertainty or {}).items()
        },
    ).with_predictions()


def test_collect_one_batch(attr_nodes):
    import torch

    collector = PredictionRecordCollector(split="test")
    nodes = attr_nodes[:4]
    bundle = make_bundle(
        [0.9, 0.2, 0.6, 0.4], uncertainty={"sngp_variance": [0.1, 0.4, 0.2, 0.3]}
    )
    labels = torch.tensor([[1.0], [0.0], [1.0], [1.0]])

    collector.add_batch(nodes, labels, bundle)
    frame = collector.to_frame()

    assert len(frame) == 4
    assert list(frame["record_id"]) == [0, 1, 2, 3]
    assert list(frame["prob"]) == pytest.approx([0.9, 0.2, 0.6, 0.4])
    assert list(frame["pred"]) == [1, 0, 1, 0]
    assert list(frame["label"]) == [1, 0, 1, 1]
    assert list(frame["correct"]) == [1, 1, 1, 0]
    assert "u_sngp_variance" in frame.columns
    assert list(frame["u_sngp_variance"]) == pytest.approx([0.1, 0.4, 0.2, 0.3])


def test_record_ids_are_sequential_across_batches(attr_nodes):
    import torch

    collector = PredictionRecordCollector()
    for start in (0, 2, 4):
        nodes = attr_nodes[start:start + 2]
        collector.add_batch(
            nodes, torch.tensor([[1.0], [0.0]]), make_bundle([0.8, 0.3]), batch_index=start
        )
    assert list(collector.to_frame()["record_id"]) == [0, 1, 2, 3, 4, 5]


def test_demographics_are_captured_from_numpy_ints(attr_nodes):
    """The dataset stores these as np.int64.

    An `isinstance(value, (int, float))` guard would drop every one of them -- the
    same bug that made graph-distance ignore demographics entirely.
    """
    import torch

    collector = PredictionRecordCollector()
    collector.add_batch(attr_nodes[:2], torch.tensor([[1.0], [0.0]]), make_bundle([0.7, 0.3]))
    frame = collector.to_frame()
    assert frame["gt_gender"].notna().all()
    assert frame["gt_race"].notna().all()
    assert frame["gt_age"].notna().all()


def test_annotation_uncertainty_is_namespaced_separately(attr_nodes):
    """AI-Face's label-uncertainty columns must never look like UQ scores."""
    import torch

    for node in attr_nodes[:2]:
        node.attributes["Uncertainty Score Gender"] = 0.21
        node.attributes["Uncertainty Score Race"] = 0.26

    collector = PredictionRecordCollector()
    collector.add_batch(attr_nodes[:2], torch.tensor([[1.0], [0.0]]), make_bundle([0.7, 0.3]))
    frame = collector.to_frame()

    assert frame["anno_unc_gender"].tolist() == pytest.approx([0.21, 0.21])
    for column in ANNOTATION_ATTRIBUTES.values():
        assert column.startswith("anno_")
        assert not column.startswith("u_"), "annotation columns must not use the u_ prefix"


def test_graph_degree_is_recorded(ring_graph):
    import torch

    _, nodes, _ = ring_graph
    collector = PredictionRecordCollector()
    collector.add_batch(nodes[:3], torch.tensor([[1.0], [0.0], [1.0]]), make_bundle([0.7, 0.3, 0.6]))
    # Ring topology: every node has exactly two neighbors.
    assert list(collector.to_frame()["graph_degree"]) == [2, 2, 2]


def test_features_absent_is_recorded_as_data_not_a_crash(attr_nodes):
    """squeezenetdf exposes no penultimate features; that must be a column value."""
    import torch

    collector = PredictionRecordCollector()
    collector.add_batch(attr_nodes[:2], torch.tensor([[1.0], [0.0]]), make_bundle([0.7, 0.3]))
    frame = collector.to_frame()
    assert list(frame["features_available"]) == [False, False]
    assert frame["feature_norm"].isna().all()


def test_features_present_records_their_norm(attr_nodes):
    import torch

    collector = PredictionRecordCollector()
    collector.add_batch(
        attr_nodes[:2], torch.tensor([[1.0], [0.0]]),
        make_bundle([0.7, 0.3], features=[[3.0, 4.0], [0.0, 5.0]]),
    )
    frame = collector.to_frame()
    assert list(frame["features_available"]) == [True, True]
    assert list(frame["feature_norm"]) == pytest.approx([5.0, 5.0])


def test_domain_labelling_from_source(attr_nodes):
    import torch

    for index, node in enumerate(attr_nodes[:4]):
        node.node_id = f"/root/{'ProGAN' if index < 2 else 'CelebA'}/{index}.png"

    collector = PredictionRecordCollector(
        data_root="/root", domain_of_source={"ProGAN": "ood"}, default_domain="id"
    )
    collector.add_batch(
        attr_nodes[:4], torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        make_bundle([0.9, 0.8, 0.2, 0.1]),
    )
    assert list(collector.to_frame()["domain"]) == ["ood", "ood", "id", "id"]


def test_mismatched_tensor_length_raises_record_collection_error(attr_nodes):
    import torch

    collector = PredictionRecordCollector()
    with pytest.raises(RecordCollectionError):
        collector.add_batch(
            attr_nodes[:3], torch.tensor([[1.0], [0.0], [1.0]]), make_bundle([0.5, 0.5])
        )


def test_empty_batch_is_a_noop(attr_nodes):
    import torch

    collector = PredictionRecordCollector()
    collector.add_batch([], torch.zeros(0, 1), make_bundle([]))
    assert collector.to_frame().empty


def test_coverage_and_summary(attr_nodes):
    import torch

    collector = PredictionRecordCollector()
    collector.note_requested(8)
    collector.add_batch(attr_nodes[:4], torch.tensor([[1.0]] * 4), make_bundle([0.6] * 4))
    collector.note_batch_failure(ValueError("boom"))

    summary = collector.summary()
    assert summary["n_rows"] == 4
    assert summary["n_requested"] == 8
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["n_batches_failed"] == 1
    assert "boom" in summary["first_error"]
    json.dumps(summary)  # must be safe to embed in the printed metrics


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

@pytest.fixture
def populated_collector(attr_nodes):
    import torch

    collector = PredictionRecordCollector(split="test")
    collector.note_requested(6)
    collector.add_batch(
        attr_nodes[:6], torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]),
        make_bundle(
            [0.91, 0.12, 0.63, 0.44, 0.75, 0.28],
            uncertainty={
                "sngp_variance": [0.11, 0.42, 0.23, 0.34, 0.15, 0.26],
                "hybrid_distance": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            },
        ),
    )
    return collector


def test_float_roundtrip_is_exact(populated_collector, tmp_path):
    """%.9g round-trips float32 exactly, so scores survive the CSV."""
    path = tmp_path / "records.csv.gz"
    frame = populated_collector.to_frame()
    write_records(frame, path)
    restored = read_records(path, verify=False)
    for column in ("prob", "u_sngp_variance", "u_hybrid_distance"):
        assert np.allclose(
            restored[column].to_numpy(), frame[column].to_numpy(), rtol=0, atol=0
        ), f"{column} did not round-trip exactly"


def test_written_bytes_are_stable(populated_collector, tmp_path):
    """Two writes of the same table must be byte-identical.

    gzip embeds an mtime in its header by default, which would make every write
    differ and defeat any content-addressed provenance.
    """
    frame = populated_collector.to_frame()
    first, second = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    digest_a = write_records(frame, first)
    digest_b = write_records(frame, second)
    assert first.read_bytes() == second.read_bytes()
    assert digest_a == digest_b


def test_column_order_is_fixed(populated_collector, tmp_path):
    path = tmp_path / "records.csv.gz"
    write_records(populated_collector.to_frame(), path)
    with gzip.open(path, "rt") as handle:
        header = handle.readline().strip().split(",")
    assert header[:6] == [
        "record_id", "rel_path", "node_id", "source_top", "source_group", "split",
    ]
    assert header == populated_collector.columns()


def test_uncertainty_columns_are_sorted_not_insertion_ordered(populated_collector):
    columns = populated_collector.columns()
    uncertainty = [column for column in columns if column.startswith("u_")]
    assert uncertainty == sorted(uncertainty), (
        "a newly added uncertainty key must not reorder the existing ones"
    )


def test_save_records_writes_a_verifiable_manifest(populated_collector, tmp_path):
    path = tmp_path / "records.csv.gz"
    frame, manifest = save_records(populated_collector, path, extra_manifest={"seed": 42})

    meta_path = default_meta_path(str(path))
    assert manifest["sha256_records"] == sha256_of_file(str(path))
    assert manifest["seed"] == 42
    assert manifest["coverage"] == pytest.approx(1.0)
    assert "numpy" in manifest["versions"]
    assert json.loads(open(meta_path).read())["schema_version"] == 1


def test_read_records_detects_tampering(populated_collector, tmp_path):
    """A modified table must not be silently scored."""
    path = tmp_path / "records.csv.gz"
    save_records(populated_collector, path)

    frame = read_records(path)  # clean read verifies
    assert len(frame) == 6

    with gzip.open(path, "wt") as handle:
        handle.write("record_id,prob\n0,0.5\n")
    with pytest.raises(ValueError, match="does not match its manifest"):
        read_records(path)


def test_default_meta_path():
    assert default_meta_path("a/b.csv.gz") == "a/b.meta.json"
    assert default_meta_path("a/b.parquet") == "a/b.meta.json"


def test_manifest_records_cv2_version(populated_collector, tmp_path):
    """Corruption bytes depend on the OpenCV JPEG encoder, so pin its version."""
    pytest.importorskip("cv2")
    _, manifest = save_records(populated_collector, tmp_path / "r.csv.gz")
    assert "cv2" in manifest["versions"]


# --------------------------------------------------------------------------- #
# The evaluate_model seam
# --------------------------------------------------------------------------- #

class FakeModel:
    """Duck-types the interface evaluate_model actually requires.

    Nothing more than `.transform`, `.eval`/`.train`, `.model`, `__call__`, and the
    optional uncertainty trio -- which is exactly the informal contract the real
    call sites rely on.
    """

    def __init__(self, uncertainty=True):
        import torch
        import torch.nn as nn

        self.model = nn.Identity()
        self.current_mode = "eval"
        self._uncertainty = uncertainty
        self.mc_dropout_samples = 0

    def transform(self, image):
        import torch
        return torch.zeros(3, 8, 8)

    def eval(self):
        self.current_mode = "eval"

    def train(self):
        self.current_mode = "train"

    def __call__(self, batch):
        import torch
        return torch.linspace(-2.0, 2.0, batch.shape[0]).reshape(-1, 1)

    def forward_with_uncertainty(self, batch, nodes=None, **kwargs):
        import torch
        from models.uncertainty import PredictionBundle

        if not self._uncertainty:
            raise AttributeError("uncertainty disabled")
        logits = self(batch)
        probabilities = torch.sigmoid(logits)
        return PredictionBundle(
            logits=logits, probabilities=probabilities,
            uncertainty={"fake_variance": probabilities * 0.1},
        ).with_predictions()

    def compute_loss(self, bundle_or_logits, labels, base_criterion=None):
        import torch
        from models.uncertainty import PredictionBundle

        logits = (
            bundle_or_logits.logits
            if isinstance(bundle_or_logits, PredictionBundle) else bundle_or_logits
        )
        criterion = base_criterion or torch.nn.BCEWithLogitsLoss()
        return criterion(logits, labels)

    def summarize_uncertainty(self, bundle):
        return {
            name: float(value.mean().item()) for name, value in bundle.uncertainty.items()
        }


def _fake_nodes(count, tmp_path):
    from tests.helpers.images import make_image_nodes
    return make_image_nodes(tmp_path / "imgs", count=count, size=8)


def test_collector_does_not_change_the_existing_metrics(tmp_path):
    """The load-bearing seam test.

    Attaching a collector must leave every metric the three existing call sites read
    bit-identical -- otherwise adding the benchmark silently perturbs training and
    model selection.
    """
    import torch
    import test_hierarchical

    nodes = _fake_nodes(8, tmp_path)
    criterion = torch.nn.BCEWithLogitsLoss()

    without = test_hierarchical.evaluate_model(
        FakeModel(), nodes, criterion, batch_size=4, device="cpu", num_workers=1,
    )
    collector = PredictionRecordCollector(split="test")
    with_records = test_hierarchical.evaluate_model(
        FakeModel(), nodes, criterion, batch_size=4, device="cpu", num_workers=1,
        record_collector=collector,
    )

    shared = set(without) & set(with_records)
    for name in sorted(shared):
        assert without[name] == with_records[name], f"metric {name!r} changed"
    assert set(with_records) - set(without) == {"records"}
    assert len(collector.rows) == len(nodes)


def test_collector_rows_are_in_deterministic_order(tmp_path):
    """Requires evaluate_model to collect thread results in submission order."""
    import torch
    import test_hierarchical

    nodes = _fake_nodes(12, tmp_path)
    criterion = torch.nn.BCEWithLogitsLoss()

    def collect(num_workers):
        collector = PredictionRecordCollector()
        test_hierarchical.evaluate_model(
            FakeModel(), nodes, criterion, batch_size=4, device="cpu",
            num_workers=num_workers, record_collector=collector,
        )
        return list(collector.to_frame()["node_id"])

    baseline = collect(1)
    for workers in (2, 4, 8):
        assert collect(workers) == baseline, (
            f"record order changed with num_workers={workers}"
        )


def test_evaluate_model_refuses_a_model_whose_eval_does_not_take_effect(tmp_path):
    """A broken eval() must fail loudly, not randomize evaluation silently.

    transform() dispatches on current_mode from worker threads, so a model that
    stays in train mode would apply stochastic augmentation during evaluation --
    producing quietly noisier, irreproducible metrics rather than an error.
    """
    import torch
    import test_hierarchical

    class StuckInTrainMode(FakeModel):
        def eval(self):
            pass  # no-op: current_mode stays "train"

    model = StuckInTrainMode()
    model.current_mode = "train"
    with pytest.raises(RuntimeError, match="current_mode is still"):
        test_hierarchical.evaluate_model(
            model, _fake_nodes(4, tmp_path), torch.nn.BCEWithLogitsLoss(),
            batch_size=2, device="cpu", num_workers=1,
        )


def test_evaluate_model_puts_the_model_in_eval_mode(tmp_path):
    """The normal path: a model handed over in train mode is switched, not rejected."""
    import torch
    import test_hierarchical

    model = FakeModel()
    model.train()
    assert model.current_mode == "train"
    test_hierarchical.evaluate_model(
        model, _fake_nodes(4, tmp_path), torch.nn.BCEWithLogitsLoss(),
        batch_size=2, device="cpu", num_workers=1,
    )
    assert model.current_mode == "eval"


def test_evaluate_model_raises_instead_of_reporting_zero_accuracy(tmp_path):
    """A total inference failure must not look like a badly-performing model.

    This is how the evidential/MC-dropout crash hid: every batch raised, the handler
    swallowed it, and the run printed "Accuracy=0.00%".
    """
    import torch
    import test_hierarchical

    class AlwaysFailing(FakeModel):
        def forward_with_uncertainty(self, batch, nodes=None, **kwargs):
            raise ValueError("simulated inference failure")

        def __call__(self, batch):
            raise ValueError("simulated inference failure")

    from test_helpers.determinism import configure_determinism
    configure_determinism(seed=0, mode="fast", allow_multi_gpu=True)  # so it is not re-raised early

    with pytest.raises(RuntimeError, match="processed 0 of"):
        test_hierarchical.evaluate_model(
            AlwaysFailing(), _fake_nodes(4, tmp_path), torch.nn.BCEWithLogitsLoss(),
            batch_size=2, device="cpu", num_workers=1,
        )


def test_metrics_report_coverage(tmp_path):
    import torch
    import test_hierarchical

    metrics = test_hierarchical.evaluate_model(
        FakeModel(), _fake_nodes(6, tmp_path), torch.nn.BCEWithLogitsLoss(),
        batch_size=3, device="cpu", num_workers=1,
    )
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["batches_failed"] == 0


class PlainModel:
    """A model with no uncertainty support at all.

    Deliberately a separate class rather than a FakeModel with the method removed,
    so that `hasattr(model, 'forward_with_uncertainty')` is genuinely False and the
    legacy branch of evaluate_model is the one exercised.
    """

    def __init__(self):
        import torch.nn as nn
        self.model = nn.Identity()
        self.current_mode = "eval"

    def transform(self, image):
        import torch
        return torch.zeros(3, 8, 8)

    def eval(self):
        self.current_mode = "eval"

    def __call__(self, batch):
        import torch
        return torch.linspace(-2.0, 2.0, batch.shape[0]).reshape(-1, 1)


def test_collector_works_on_the_legacy_no_uncertainty_path(tmp_path):
    """Records must still be produced for a model without forward_with_uncertainty."""
    import torch
    import test_hierarchical

    assert not hasattr(PlainModel(), "forward_with_uncertainty")

    collector = PredictionRecordCollector()
    metrics = test_hierarchical.evaluate_model(
        PlainModel(), _fake_nodes(4, tmp_path), torch.nn.BCEWithLogitsLoss(),
        batch_size=2, device="cpu", num_workers=1, record_collector=collector,
    )
    assert metrics["coverage"] == pytest.approx(1.0)
    assert len(collector.rows) == 4
    frame = collector.to_frame()
    # No uncertainty columns, but probabilities and labels are still captured -- which
    # is enough for the baseline max-prob and entropy methods.
    assert not [column for column in frame.columns if column.startswith("u_")]
    assert frame["prob"].notna().all()
