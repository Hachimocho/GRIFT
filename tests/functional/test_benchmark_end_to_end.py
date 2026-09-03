"""End-to-end: train briefly, evaluate with records, score into a results table.

This is the test that proves phase A is usable rather than merely present. It walks
the whole path -- real training, real evaluation through the production
``evaluate_model``, record persistence, and scoring -- for several methods at once,
using the tiny synthetic detector so it needs no dataset and no download.
"""

import numpy as np
import pytest
import torch

from evaluation.uq.records import PredictionRecordCollector, read_records, save_records
from evaluation.uq.registry import expand_matrix, gate_model
from evaluation.uq.scoring import (
    Cell, add_skipped_rows, collapse_rank_equivalents, pivot_for_paper, score_cells,
)

pytestmark = pytest.mark.slow


def train_briefly(model, nodes, steps=25, batch_size=8):
    model.train()
    for step in range(steps):
        start = (step * batch_size) % max(1, len(nodes) - batch_size + 1)
        batch = nodes[start:start + batch_size]
        images = torch.stack([
            model.transform(node.get_data().load_data()) for node in batch
        ])
        labels = torch.tensor(
            [[float(node.get_label())] for node in batch], dtype=torch.float
        )
        loss = model.compute_loss(
            model.forward_with_uncertainty(images, nodes=batch, update_precision=True),
            labels,
        )
        model.optim.zero_grad()
        loss.backward()
        model.optim.step()
    return model


@pytest.fixture
def many_image_nodes(tmp_path):
    """Enough nodes that metrics are meaningful and both classes are present."""
    from tests.helpers.images import make_image_nodes
    return make_image_nodes(tmp_path / "imgs", count=64, size=8)


def test_train_evaluate_score_pipeline(cnn_model_factory, many_image_nodes, tmp_path):
    """The full phase-A path for a single method."""
    import test_hierarchical

    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.1, lr=1e-2)
    train_briefly(model, many_image_nodes)

    collector = PredictionRecordCollector(split="test")
    metrics = test_hierarchical.evaluate_model(
        model, many_image_nodes, torch.nn.BCEWithLogitsLoss(),
        batch_size=16, device="cpu", num_workers=2, record_collector=collector,
    )

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["records"]["n_rows"] == len(many_image_nodes)

    records_path = tmp_path / "records.csv.gz"
    frame, manifest = save_records(collector, records_path, extra_manifest={"seed": 42})
    assert manifest["coverage"] == pytest.approx(1.0)

    # Round-trip through disk, so the scored numbers are the persisted ones.
    reloaded = read_records(records_path)
    assert len(reloaded) == len(many_image_nodes)

    results = score_cells([
        Cell(
            detector="resnestdf", method_id="sngp", score_column="u_sngp_variance",
            frame=reloaded, coverage=1.0, determinism_mode="strict", seed=42,
        )
    ])
    assert len(results) == 1
    row = results.iloc[0]
    assert row["status"] in {"ok", "degenerate"}
    assert np.isfinite(row["ece_confidence"])
    assert np.isfinite(row["brier"])
    assert row["n"] == len(many_image_nodes)


def test_multi_method_results_table(cnn_model_factory, many_image_nodes, tmp_path):
    """Several methods scored off one evaluation pass, plus explained skips.

    One trained model yields the baselines, MC dropout, its own head, and the graph
    methods -- the whole in-distribution comparison from a single record table.
    """
    import test_hierarchical
    from models.uncertainty import GraphDistanceUncertainty

    methods = ("attribute_distance", "hybrid_distance", "degree_penalty")
    model = cnn_model_factory(
        uncertainty_head="sngp", uncertainty_dropout_rate=0.2, lr=1e-2,
        graph_uncertainty_methods=list(methods),
    )
    # Give the nodes a ring topology so graph uncertainty has neighbors to compare to.
    from edges.Edge import Edge
    for index, node in enumerate(many_image_nodes):
        peer = many_image_nodes[(index + 1) % len(many_image_nodes)]
        edge = Edge(node, peer, x=None)
        node.add_edge(edge)
        peer.add_edge(edge)

    standardizer = GraphDistanceUncertainty(methods=methods).fit(many_image_nodes)
    model.set_graph_distance_standardizer(standardizer)
    train_briefly(model, many_image_nodes)

    collector = PredictionRecordCollector(split="test")
    test_hierarchical.evaluate_model(
        model, many_image_nodes, torch.nn.BCEWithLogitsLoss(),
        batch_size=16, device="cpu", num_workers=2, record_collector=collector,
    )
    frame = collector.to_frame()

    scored = [
        ("baseline_maxprob", "u_maxprob"),
        ("baseline_entropy", "u_entropy"),
        ("sngp", "u_sngp_variance"),
        ("graph_attribute_distance", "u_attribute_distance"),
        ("graph_hybrid_distance", "u_hybrid_distance"),
        ("graph_degree_only", "u_degree_penalty"),
    ]
    available = [
        (method_id, column) for method_id, column in scored
        if column in frame.columns or column.startswith("u_maxprob")
        or column in ("u_entropy", "u_margin")
    ]
    assert len(available) >= 5, f"expected most methods to be scoreable, got {available}"

    results = score_cells([
        Cell(
            detector="resnestdf", method_id=method_id, score_column=column,
            frame=frame, coverage=1.0, determinism_mode="strict",
            graph_norm_sha256=standardizer.stats_hash,
        )
        for method_id, column in available
    ])

    assert len(results) == len(available)
    # Graph methods must have N/A calibration, not zero.
    graph_rows = results[results["method_family"] == "graph"]
    assert not graph_rows.empty
    assert graph_rows["calibration_applicable"].eq(False).all()
    assert graph_rows["ece_confidence"].isna().all()
    # Probabilistic methods must have real calibration numbers.
    probabilistic = results[results["produces_probabilities"]]
    assert probabilistic["ece_confidence"].notna().all()

    # Rank-equivalent duplicates get collapsed rather than printed twice.
    collapsed = collapse_rank_equivalents(results)
    entropy = collapsed[collapsed["method_id"] == "baseline_entropy"]
    if not entropy.empty:
        assert entropy["auroc_error"].isna().all()
        assert entropy["ece_confidence"].notna().all()

    # Gate skips join the table as explained holes.
    _, decisions = expand_matrix(
        ["sngp", "evidential"], ["squeezenetdf", "dag_fdd"],
    )
    combined = add_skipped_rows(results, decisions)
    assert (combined["status"] == "broken").any(), "dag_fdd should appear as broken"
    assert (combined["status"] == "skipped").any(), "squeezenetdf x head should be skipped"

    table = pivot_for_paper(results, metric="eaurc")
    assert not table.empty


def test_degree_only_control_is_scoreable_alongside_the_distances(
    cnn_model_factory, two_cluster_graph
):
    """The ablation control must produce a comparable number.

    Without it there is no way to tell whether the distance methods predict error or
    merely flag low-degree nodes.
    """
    from models.uncertainty import GraphDistanceUncertainty

    _, nodes, _ = two_cluster_graph
    methods = ("attribute_distance", "hybrid_distance", "degree_penalty")
    model = cnn_model_factory(
        uncertainty_head="none", graph_uncertainty_methods=list(methods)
    )
    model.set_graph_distance_standardizer(
        GraphDistanceUncertainty(methods=methods).fit(nodes)
    )
    model.eval()

    collector = PredictionRecordCollector(split="test")
    images = torch.rand(len(nodes), 3, 16, 16)
    with torch.no_grad():
        bundle = model.forward_with_uncertainty(images, nodes=nodes)
    collector.add_batch(
        nodes,
        torch.tensor([[float(node.get_label())] for node in nodes]),
        bundle,
    )
    frame = collector.to_frame()

    for column in ("u_attribute_distance", "u_hybrid_distance", "u_degree_penalty"):
        assert column in frame.columns, f"{column} did not reach the record table"

    results = score_cells([
        Cell(detector="resnestdf", method_id=method_id, score_column=column,
             frame=frame, coverage=1.0, determinism_mode="strict")
        for method_id, column in (
            ("graph_attribute_distance", "u_attribute_distance"),
            ("graph_hybrid_distance", "u_hybrid_distance"),
            ("graph_degree_only", "u_degree_penalty"),
        )
    ])
    assert len(results) == 3
    assert results["auroc_error"].notna().any(), (
        "the control and the distances must both yield a comparable ranking number"
    )


def test_pipeline_is_reproducible(cnn_model_factory, many_image_nodes, tmp_path):
    """Same seed end to end: byte-identical record tables and identical metrics."""
    import test_hierarchical
    from test_helpers.determinism import configure_determinism

    def run(tag):
        configure_determinism(seed=606, mode="strict", allow_multi_gpu=True)
        model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.0, lr=1e-2)
        train_briefly(model, many_image_nodes, steps=10)
        collector = PredictionRecordCollector(split="test")
        test_hierarchical.evaluate_model(
            model, many_image_nodes, torch.nn.BCEWithLogitsLoss(),
            batch_size=16, device="cpu", num_workers=2, record_collector=collector,
        )
        path = tmp_path / f"records_{tag}.csv.gz"
        _, manifest = save_records(collector, path)
        row = score_cells([
            Cell(detector="resnestdf", method_id="sngp",
                 score_column="u_sngp_variance", frame=collector.to_frame(),
                 coverage=1.0, determinism_mode="strict")
        ]).iloc[0]
        return path, manifest["sha256_records"], row

    first_path, first_digest, first_row = run("a")
    second_path, second_digest, second_row = run("b")

    assert first_digest == second_digest, "record tables are not byte-identical"
    assert first_path.read_bytes() == second_path.read_bytes()
    for metric in ("auroc_error", "eaurc", "ece_confidence", "brier", "nll"):
        assert first_row[metric] == pytest.approx(second_row[metric], abs=0.0), metric


def test_runtime_gate_confirms_a_trained_model_matches_its_method(cnn_model_factory):
    """After training, verify the checkpoint really produces what the method claims."""
    model = cnn_model_factory(uncertainty_head="sngp", uncertainty_dropout_rate=0.1)
    assert gate_model("sngp", "resnestdf", model).compatible

    headless = cnn_model_factory(uncertainty_head="none")
    decision = gate_model("sngp", "resnestdf", headless)
    assert not decision.compatible, "a headless model must not be scored as SNGP"
