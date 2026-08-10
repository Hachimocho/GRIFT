"""Per-sample prediction records: the benchmark's input.

Everything else here depends on this. Previously the evaluation path thresholded
probabilities and discarded the continuous scores
(``predictions = probabilities.cpu().numpy() > 0.5``), built ``all_predictions`` and
``all_labels`` but never returned them, and reduced each per-sample uncertainty
tensor to a single batch mean one line after it was computed. So calibration,
selective prediction, and any correlation between uncertainty and correctness were
all unobtainable -- not hard, unobtainable.

Design notes:

**A collector injected into the existing ``evaluate_model``**, rather than a second
evaluation entrypoint or a bigger return dict. The metrics dict is ``json.dumps``'d
to stdout and scraped by the GPU queue manager, so putting 400k rows in it would
bloat every log and break the parser. And the loader (thread pool, transform
dispatch, tuple-label handling, bundle/``hasattr`` fallbacks) is subtle enough that
a copy would drift -- one inference path guarantees the benchmark measures what
training measures.

**``.csv.gz``, not Parquet.** No Parquet engine is installed in the training
environment (``pyarrow`` and ``fastparquet`` are both absent), and ``.csv.gz`` matches
the repo's existing edge-cache convention. ``write_records`` will prefer Parquet if an
engine appears, but the determinism test pins the csv.gz path.

**Byte-stable output.** ``float_format='%.17g'`` round-trips float64 exactly, columns
are in fixed order, rows are sorted by ``record_id``, and gzip is written with
``mtime=0`` so the header carries no timestamp. Two runs that computed the same
predictions therefore produce identical files.
"""

import gzip
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SCHEMA_VERSION = 1

#: Fixed column order. Anything not listed is appended in sorted order, so a new
#: uncertainty key does not silently reorder existing columns.
IDENTITY_COLUMNS = (
    "record_id", "rel_path", "node_id", "source_top", "source_group",
    "split", "domain", "corruption", "severity", "graph_degree",
)
OUTCOME_COLUMNS = ("label", "pred", "correct", "logit", "prob", "loss_sample")
FEATURE_COLUMNS = ("features_available", "feature_norm")
DEMOGRAPHIC_COLUMNS = ("gt_gender", "gt_age", "gt_race", "intersection")

#: Prefix for AI-Face's per-sample *annotation* uncertainty over demographic labels.
#: Namespaced so it can never be mistaken for predictive uncertainty; the registry
#: refuses to score any column with this prefix.
ANNOTATION_PREFIX = "anno_"
UNCERTAINTY_PREFIX = "u_"

#: Node attribute names carrying AI-Face annotation uncertainty.
ANNOTATION_ATTRIBUTES = {
    "Uncertainty Score Gender": "anno_unc_gender",
    "Uncertainty Score Age": "anno_unc_age",
    "Uncertainty Score Race": "anno_unc_race",
}
DEMOGRAPHIC_ATTRIBUTES = {
    "Ground Truth Gender": "gt_gender",
    "Ground Truth Age": "gt_age",
    "Ground Truth Race": "gt_race",
    "Intersection": "intersection",
}

#: Second-level directory carries real/fake for these mixed video sources, so the
#: group id must include it or reals and fakes from the same set are conflated.
MIXED_SOURCES = frozenset({"celebdf", "ff++", "dfd", "dfdc"})


class RecordCollectionError(RuntimeError):
    """Raised when record collection itself fails.

    Deliberately distinct so ``evaluate_model`` can re-raise it rather than let it be
    absorbed by the per-batch ``except Exception: continue`` and misreported as an
    inference failure.
    """


def parse_source_group(rel_path):
    """``(source_top, source_group)`` from a dataset-relative image path.

    ``source_group`` includes the second path component for the mixed video sets, so
    ``celebdf/real`` and ``celebdf/crop_img`` are distinguishable. Note this is for
    *grouping* only -- labels always come from ``Target``, because the ``real/``
    folders are 9-18% fake.
    """
    parts = [part for part in str(rel_path).replace("\\", "/").split("/") if part]
    if not parts:
        return "unknown", "unknown"
    top = parts[0]
    if top in MIXED_SOURCES and len(parts) > 1:
        return top, f"{top}/{parts[1]}"
    return top, top


def relative_path(node_id, data_root=None):
    """Strip ``data_root`` from an absolute node id, leaving a portable path."""
    text = str(node_id).replace("\\", "/")
    if data_root:
        root = str(data_root).replace("\\", "/").rstrip("/")
        if text.startswith(root):
            text = text[len(root):]
    return text.lstrip("/")


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1)


def _scalar_column(value, expected):
    """Flatten a per-sample tensor to a 1-D array of length ``expected``."""
    array = _to_numpy(value)
    if array is None:
        return None
    if array.ndim == 2:
        array = array[:, 0] if array.shape[1] == 1 else array.mean(axis=1)
    if array.shape[0] != expected:
        raise RecordCollectionError(
            f"per-sample tensor has {array.shape[0]} rows, expected {expected}"
        )
    return array


def _numeric(value, default=np.nan):
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class PredictionRecordCollector:
    """Accumulates one row per evaluated sample.

    Populated from inside ``evaluate_model`` at the point where the prediction
    bundle, the loaded nodes, and the labels are all simultaneously in scope.
    """

    split: str = "test"
    data_root: Optional[str] = None
    domain_of_source: Dict[str, str] = field(default_factory=dict)
    corruption: str = "none"
    severity: int = 0
    default_domain: str = "id"

    rows: List[dict] = field(default_factory=list)
    uncertainty_keys: set = field(default_factory=set)
    n_requested: int = 0
    n_batches_failed: int = 0
    first_error: Optional[str] = None

    def note_requested(self, count):
        self.n_requested = int(count)

    def note_batch_failure(self, exc):
        self.n_batches_failed += 1
        if self.first_error is None:
            self.first_error = f"{type(exc).__name__}: {exc}"

    def add_batch(self, nodes, labels, bundle, batch_index=0):
        """Append one row per sample in this batch."""
        try:
            self._add_batch(nodes, labels, bundle, batch_index)
        except RecordCollectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as our own type
            raise RecordCollectionError(
                f"failed collecting records for batch {batch_index}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _add_batch(self, nodes, labels, bundle, batch_index):
        count = len(nodes)
        if count == 0:
            return

        probabilities = _scalar_column(bundle.probabilities, count)
        logits = _scalar_column(bundle.logits, count)
        predictions = _scalar_column(bundle.predictions, count)
        label_values = _scalar_column(labels, count)
        if probabilities is None or label_values is None:
            raise RecordCollectionError("bundle is missing probabilities or labels")
        if predictions is None:
            predictions = (probabilities > 0.5).astype(np.float32)

        uncertainty = {}
        for name, tensor in (bundle.uncertainty or {}).items():
            column = f"{UNCERTAINTY_PREFIX}{name}"
            uncertainty[column] = _scalar_column(tensor, count)
            self.uncertainty_keys.add(column)

        features = _to_numpy(bundle.features)
        feature_norms = None
        if features is not None and features.ndim == 2:
            feature_norms = np.linalg.norm(features, axis=1)

        for index, node in enumerate(nodes):
            node_id = getattr(node, "node_id", f"batch{batch_index}_{index}")
            rel = relative_path(node_id, self.data_root)
            source_top, source_group = parse_source_group(rel)
            attributes = getattr(node, "attributes", {}) or {}

            row = {
                "record_id": len(self.rows),
                "rel_path": rel,
                "node_id": str(node_id),
                "source_top": source_top,
                "source_group": source_group,
                "split": self.split,
                # The node's own attribute wins: `holdouts.apply_holdout` sets it on
                # exactly the nodes it filtered, so it is the same code's own record of
                # what it did. `domain_of_source` remains for callers that label by
                # group without having run a holdout -- but if both are present and
                # disagree, trusting the group map would silently mislabel whichever
                # nodes the holdout actually moved.
                "domain": attributes.get("domain")
                or self.domain_of_source.get(source_group)
                or self.domain_of_source.get(source_top)
                or self.default_domain,
                "corruption": self.corruption,
                "severity": int(self.severity),
                "graph_degree": len(node.get_adjacent_nodes())
                if hasattr(node, "get_adjacent_nodes") else -1,
                "label": int(label_values[index]),
                "pred": int(predictions[index]),
                "prob": float(probabilities[index]),
                "logit": float(logits[index]) if logits is not None else np.nan,
                "loss_sample": np.nan,
                "features_available": features is not None,
                "feature_norm": float(feature_norms[index]) if feature_norms is not None else np.nan,
            }
            row["correct"] = int(row["pred"] == row["label"])

            for column, values in uncertainty.items():
                row[column] = float(values[index]) if values is not None else np.nan

            # Demographics and annotation uncertainty. `_numeric` accepts numpy
            # scalars -- an isinstance(value, (int, float)) test would silently drop
            # every np.int64 the dataset produces.
            for attribute, column in DEMOGRAPHIC_ATTRIBUTES.items():
                row[column] = _numeric(attributes.get(attribute))
            for attribute, column in ANNOTATION_ATTRIBUTES.items():
                row[column] = _numeric(attributes.get(attribute))

            self.rows.append(row)

    # -- output ------------------------------------------------------------- #

    def columns(self):
        """Column order: fixed groups first, then uncertainty and annotation."""
        ordered = list(IDENTITY_COLUMNS) + list(OUTCOME_COLUMNS)
        ordered += sorted(self.uncertainty_keys)
        ordered += list(DEMOGRAPHIC_COLUMNS)
        ordered += sorted(ANNOTATION_ATTRIBUTES.values())
        ordered += list(FEATURE_COLUMNS)
        seen, unique = set(), []
        for column in ordered:
            if column not in seen:
                seen.add(column)
                unique.append(column)
        return unique

    def to_frame(self):
        import pandas as pd

        frame = pd.DataFrame(self.rows)
        if frame.empty:
            return pd.DataFrame(columns=self.columns())
        for column in self.columns():
            if column not in frame.columns:
                frame[column] = np.nan
        return frame.sort_values("record_id").reset_index(drop=True)[self.columns()]

    def coverage(self):
        """Fraction of requested samples that produced a record.

        The report refuses cells below 99%: `evaluate_model`'s per-batch
        `except Exception: continue` can otherwise compute a headline number on a
        small fraction of the data with no visible sign.
        """
        if not self.n_requested:
            return 1.0 if self.rows else 0.0
        return len(self.rows) / float(self.n_requested)

    def summary(self):
        """Small, JSON-safe dict, safe to embed in the metrics printed to stdout."""
        return {
            "n_rows": len(self.rows),
            "n_requested": self.n_requested,
            "n_batches_failed": self.n_batches_failed,
            "coverage": round(self.coverage(), 6),
            "first_error": self.first_error,
            "uncertainty_columns": sorted(self.uncertainty_keys),
        }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def write_records(frame, path, engine="auto"):
    """Write a record table deterministically. Returns its sha256.

    Written to a temporary file and renamed, so a crashed run cannot leave a
    half-written table that later looks complete.
    """
    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    if engine == "auto":
        engine = "parquet" if (path.endswith(".parquet") and _parquet_available()) else "csv.gz"

    temporary = f"{path}.tmp"
    if engine == "parquet":
        frame.to_parquet(temporary, index=False)
    else:
        # %.17g, not %.9g: 9 significant digits round-trip float32 exactly, but the
        # DataFrame holds Python floats (float64), so 9 digits would quietly lose the
        # low bits of every value. 17 digits round-trips float64 exactly, and the
        # extra width costs almost nothing after gzip.
        payload = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
        buffer = io.BytesIO()
        # mtime=0: the gzip header otherwise embeds the current time, so two
        # identical tables would produce different bytes.
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as handle:
            handle.write(payload.encode("utf-8"))
        with open(temporary, "wb") as handle:
            handle.write(buffer.getvalue())

    os.replace(temporary, path)
    return sha256_of_file(path)


def _parquet_available():
    for module in ("pyarrow", "fastparquet"):
        try:
            __import__(module)
            return True
        except ImportError:
            continue
    return False


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path, meta_path=None, verify=True):
    """Read a record table, verifying it against its sidecar by default."""
    import pandas as pd

    path = str(path)
    if verify:
        meta_path = meta_path or default_meta_path(path)
        if os.path.exists(meta_path):
            with open(meta_path) as handle:
                meta = json.load(handle)
            expected = meta.get("sha256_records")
            if expected:
                actual = sha256_of_file(path)
                if actual != expected:
                    raise ValueError(
                        f"record table {path} does not match its manifest "
                        f"(sha256 {actual} != {expected}); it was modified or "
                        f"truncated after being written"
                    )

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # float_precision="round_trip" is required for exactness: pandas' default CSV
    # float parser ("high") is fast but not round-trip correct, so scores would come
    # back perturbed in their low bits even though the file holds enough digits.
    return pd.read_csv(
        path,
        compression="gzip" if path.endswith(".gz") else None,
        float_precision="round_trip",
    )


def default_meta_path(records_path):
    text = str(records_path)
    for suffix in (".csv.gz", ".parquet", ".csv"):
        if text.endswith(suffix):
            return text[: -len(suffix)] + ".meta.json"
    return text + ".meta.json"


def write_manifest(path, records_path, records_sha256, collector=None, extra=None):
    """Write the provenance sidecar.

    Non-negotiable for a benchmark artifact: without the seed, config, code version,
    and library versions, a result cannot be traced to the conditions that produced
    it, and two runs that disagree cannot be diagnosed.
    """
    import numpy as np_module

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "records_path": os.path.basename(str(records_path)),
        "sha256_records": records_sha256,
        "versions": {"numpy": np_module.__version__},
    }

    try:
        import cv2
        # Recorded because image corruption bytes depend on the OpenCV JPEG encoder.
        manifest["versions"]["cv2"] = cv2.__version__
    except ImportError:
        pass
    try:
        import torch
        manifest["versions"]["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import pandas
        manifest["versions"]["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        from test_helpers.determinism import is_configured, run_fingerprint
        if is_configured():
            manifest["determinism"] = run_fingerprint()
    except ImportError:
        pass

    if collector is not None:
        manifest.update(collector.summary())
        manifest["split"] = collector.split
        manifest["corruption"] = {
            "id": collector.corruption, "severity": collector.severity,
        }
    if extra:
        manifest.update(extra)

    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)
    return manifest


def save_records(collector, records_path, extra_manifest=None):
    """Write both the table and its manifest. Returns ``(frame, manifest)``."""
    frame = collector.to_frame()
    digest = write_records(frame, records_path)
    manifest = write_manifest(
        default_meta_path(records_path), records_path, digest,
        collector=collector, extra=extra_manifest,
    )
    return frame, manifest
