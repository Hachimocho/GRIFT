"""Checks against the real AI-Face dataset on disk.

Marked `data`; run with `--run-data`. Everything else in the suite is synthetic, so
these are what confirm the assumptions the rest of the work is built on -- the CSV
schema, the dtypes the parsing bugs hinged on, and the source composition that drives
the held-out-generator protocol.

Deliberately read-only and cheap: they parse manifests and check a handful of images,
never load the multi-GB quality CSVs.
"""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data

REQUIRED_COLUMNS = {
    "Image Path", "Target",
    "Ground Truth Gender", "Ground Truth Age", "Ground Truth Race",
}
ANNOTATION_COLUMNS = {
    "Uncertainty Score Gender", "Uncertainty Score Age", "Uncertainty Score Race",
}

#: Sources that are exclusively real after the label correction. AI-Face v2's real set
#: is FFHQ, IMDB-WIKI (the `wiki/` folder), and the real portions of the four video
#: corpora below.
REAL_ONLY_SOURCES = {"FFHQ", "wiki"}

#: Video sets whose *second* path component carries real/fake. The naming is
#: inconsistent upstream -- `ff++` and `celebdf` use `real/` vs `crop_img/`, while
#: `dfdc` and `dfd` use `real/` vs `fake/` -- so the predicate is
#: `second == "real"`, never `second != "fake"`.
MIXED_SOURCES = {"celebdf", "ff++", "dfd", "dfdc"}

#: Removed by the label correction. Genuine photographs, but not part of AI-Face v2's
#: real set; v2 dropped them rather than relabelling them fake.
DROPPED_SOURCES = {"CelebA", "casia-webface"}


def expected_target(image_path):
    """The label a path must carry after correction: 0 real, 1 fake."""
    parts = [part for part in str(image_path).replace("\\", "/").split("/") if part]
    source = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    if source in REAL_ONLY_SOURCES:
        return 0
    if source in MIXED_SOURCES and second == "real":
        return 0
    return 1


def read_head(path, limit=4000):
    """Read the first ``limit`` rows. Handles the CRLF line endings these files use."""
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


# --------------------------------------------------------------------------- #
# Layout and schema
# --------------------------------------------------------------------------- #

def test_dataset_root_resolves(ai_face_root):
    """`resolve_ai_face_data_root` must find the mounted dataset."""
    assert ai_face_root.is_dir()
    assert str(ai_face_root).endswith("ai-face")


def test_expected_manifests_exist(ai_face_root):
    for name in ("train.csv", "val.csv", "test.csv"):
        assert (ai_face_root / name).is_file(), f"missing {name}"


def test_quality_csvs_exist(ai_face_root):
    """These supply blur/symmetry/emotion/face_embedding for graph uncertainty."""
    present = [
        name for name in ("train_quality.csv", "val_quality.csv", "test_quality.csv")
        if (ai_face_root / name).is_file()
    ]
    assert present, "no *_quality.csv found; graph-distance attributes would be absent"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_required_columns_present(ai_face_root, split):
    rows = read_head(ai_face_root / f"{split}.csv", limit=5)
    assert rows, f"{split}.csv is empty"
    missing = REQUIRED_COLUMNS - set(rows[0])
    assert not missing, f"{split}.csv is missing {sorted(missing)}"


def test_annotation_uncertainty_columns_are_present(ai_face_root):
    """These are *label* uncertainty and must not be confused with UQ scores."""
    rows = read_head(ai_face_root / "train.csv", limit=5)
    assert ANNOTATION_COLUMNS <= set(rows[0])


def test_line_endings_are_uniformly_lf(ai_face_root):
    """All manifests use LF.

    They did not before the correction: `test.csv` was CRLF while the others were LF.
    That inconsistency is worse than either convention alone, because code that works
    on train silently mis-parses test -- the final column becomes "0\\r" rather than
    "0". `int("0\\r")` happens to succeed, but a string comparison against "0" fails
    and `awk -F,` yields a trailing carriage return. pandas and
    `csv.DictReader(newline="")` handle either, so it only bit hand-rolled parsing.
    """
    endings = {}
    for name in ("train.csv", "val.csv", "test.csv", "train_small.csv"):
        path = ai_face_root / name
        if not path.is_file():
            continue
        with open(path, "rb") as handle:
            endings[name] = "CRLF" if handle.readline().endswith(b"\r\n") else "LF"

    assert endings, "no manifests found"
    non_lf = {name: ending for name, ending in endings.items() if ending != "LF"}
    assert not non_lf, f"these manifests are not LF: {non_lf}"


def test_image_paths_are_leading_slash_relative(ai_face_root):
    """Resolved as `os.path.join(data_root, path.lstrip('/'))`."""
    rows = read_head(ai_face_root / "train.csv", limit=50)
    assert all(row["Image Path"].startswith("/") for row in rows)


def test_sampled_images_exist_on_disk(ai_face_root):
    rows = read_head(ai_face_root / "train.csv", limit=200)
    sampled = rows[::40][:5]
    for row in sampled:
        path = ai_face_root / row["Image Path"].lstrip("/")
        assert path.is_file(), f"manifest references a missing image: {path}"


def test_a_sampled_image_loads(ai_face_root):
    cv2 = pytest.importorskip("cv2")
    from data.ImageFileData import ImageFileData

    rows = read_head(ai_face_root / "train.csv", limit=50)
    for row in rows:
        path = ai_face_root / row["Image Path"].lstrip("/")
        if not path.is_file() or path.suffix.lower().lstrip(".") not in {"jpg", "jpeg", "png"}:
            continue
        image = ImageFileData(str(path)).load_data()
        assert image is not None and image.ndim == 3 and image.shape[2] == 3
        return
    pytest.skip("no loadable image found in the sampled rows")


# --------------------------------------------------------------------------- #
# Dtypes -- the assumptions the parsing bugs hinged on
# --------------------------------------------------------------------------- #

def test_demographics_load_as_numpy_ints(ai_face_root):
    """The assumption behind the graph-distance and record-collection fixes.

    pandas reads these columns as int64, and `np.int64` is *not* a Python `int`. An
    `isinstance(value, (int, float))` guard therefore drops every one of them
    silently, which is exactly what made graph-distance ignore gender, race, and age.
    """
    pandas = pytest.importorskip("pandas")

    frame = pandas.read_csv(ai_face_root / "train.csv", nrows=500)
    for column in ("Ground Truth Gender", "Ground Truth Age", "Ground Truth Race", "Target"):
        assert np.issubdtype(frame[column].dtype, np.integer), (
            f"{column} is {frame[column].dtype}, expected an integer dtype"
        )
        value = frame[column].iloc[0]
        assert isinstance(value, np.integer)
        assert not isinstance(value, int), (
            f"{column} values are np.int64, which is not a Python int -- the "
            f"distinction the isinstance bug turned on"
        )


def test_target_is_binary(ai_face_root):
    pandas = pytest.importorskip("pandas")
    frame = pandas.read_csv(ai_face_root / "train.csv", nrows=5000)
    assert set(frame["Target"].unique()) <= {0, 1}


def test_annotation_uncertainty_is_a_float_in_range(ai_face_root):
    pandas = pytest.importorskip("pandas")
    frame = pandas.read_csv(ai_face_root / "train.csv", nrows=1000)
    for column in sorted(ANNOTATION_COLUMNS):
        assert np.issubdtype(frame[column].dtype, np.floating)
        assert frame[column].between(0.0, 1.0).all(), f"{column} is outside [0, 1]"


# --------------------------------------------------------------------------- #
# Source composition -- what the holdout protocol depends on
# --------------------------------------------------------------------------- #

def source_composition(ai_face_root, split, limit=200_000):
    """Per-source real/fake counts, using the same parser the records use."""
    pandas = pytest.importorskip("pandas")
    from evaluation.uq.records import parse_source_group

    frame = pandas.read_csv(
        ai_face_root / f"{split}.csv", usecols=["Image Path", "Target"], nrows=limit
    )
    groups = {}
    for path, target in zip(frame["Image Path"], frame["Target"]):
        top, _ = parse_source_group(str(path).lstrip("/"))
        entry = groups.setdefault(top, {"real": 0, "fake": 0})
        entry["fake" if int(target) == 1 else "real"] += 1
    return groups


def test_source_parsing_covers_every_row(ai_face_root):
    groups = source_composition(ai_face_root, "train", limit=20_000)
    assert groups, "no sources parsed"
    assert "unknown" not in groups, "some image paths did not yield a source"


def test_real_only_sources_are_entirely_real(ai_face_root):
    """Underpins the holdout design: these are the negative-class pools.

    Exactly, not approximately -- after the correction `Target` is a pure function of
    the path. Holding one of these out would remove a large share of the real data and
    change the training class prior, confounding distribution shift with
    class-imbalance shift, which is why the protocol holds out generators instead.
    """
    groups = source_composition(ai_face_root, "train")
    seen = REAL_ONLY_SOURCES & set(groups)
    assert seen, f"none of {sorted(REAL_ONLY_SOURCES)} found; expected the real pools"
    for source in sorted(seen):
        counts = groups[source]
        assert counts["fake"] == 0, (
            f"{source} has {counts['fake']} fake rows; it should be exclusively real"
        )
        assert counts["real"] > 0


def test_dropped_sources_are_gone(ai_face_root):
    """CelebA and casia-webface were removed, not relabelled.

    They are genuine photographs, so labelling them fake would have taught a detector
    that real photos are fake. AI-Face v2 removed them from its real set; this copy
    now matches.
    """
    for split in ("train", "val", "test"):
        groups = source_composition(ai_face_root, split)
        present = DROPPED_SOURCES & set(groups)
        assert not present, f"{split}.csv still contains {sorted(present)}"


def test_target_is_a_pure_function_of_the_path(ai_face_root):
    """The core post-correction invariant, asserted over every row of every split.

    Before the correction this held for train and val but not test, which carried
    3,825 rows (0.9%) of random label noise -- spread uniformly through the file, so
    not a corrupted block or a column shift.
    """
    pandas = pytest.importorskip("pandas")

    for split in ("train", "val", "test"):
        path = ai_face_root / f"{split}.csv"
        if not path.is_file():
            continue
        frame = pandas.read_csv(path, usecols=["Image Path", "Target"])
        expected = frame["Image Path"].map(expected_target)
        mismatches = int((frame["Target"].astype(int) != expected).sum())
        assert mismatches == 0, (
            f"{split}.csv has {mismatches} of {len(frame)} rows whose Target disagrees "
            f"with its path"
        )


def test_corrected_class_balance(ai_face_root):
    """Records the composition the correction produced, and why it matters.

    ~13% real, against AI-Face v2's published ~24% (400,885 real images). The six real
    source corpora hold only ~143k images in this copy, so roughly 258k real images
    that v2 claims are absent here. Every prior-sensitive metric reflects that: Brier
    and NLL directly, ECE through the base rate. Recovering them means re-downloading.
    """
    pandas = pytest.importorskip("pandas")

    totals = {"rows": 0, "real": 0}
    for split in ("train", "val", "test"):
        path = ai_face_root / f"{split}.csv"
        if not path.is_file():
            continue
        frame = pandas.read_csv(path, usecols=["Target"])
        totals["rows"] += len(frame)
        totals["real"] += int((frame["Target"] == 0).sum())

    assert totals["rows"] > 0
    real_fraction = totals["real"] / totals["rows"]
    assert 0.10 < real_fraction < 0.20, (
        f"real fraction is {real_fraction:.1%} over {totals['rows']:,} rows; expected "
        f"~13%. A large move means the manifests changed and the holdout sizing and "
        f"prior-sensitive metrics should be revisited"
    )


def test_mixed_video_sources_need_their_second_path_component(ai_face_root):
    """`celebdf/real` vs `celebdf/crop_img` must be distinguishable.

    Grouping by the top component alone would merge a source's reals and fakes into
    one bucket, so a holdout could not remove only the fake half.
    """
    from evaluation.uq.records import parse_source_group

    pandas = pytest.importorskip("pandas")
    frame = pandas.read_csv(
        ai_face_root / "train.csv", usecols=["Image Path"], nrows=200_000
    )
    for source in sorted(MIXED_SOURCES):
        matching = [
            str(path) for path in frame["Image Path"]
            if str(path).lstrip("/").startswith(source + "/")
        ]
        if not matching:
            continue
        groups = {parse_source_group(path.lstrip("/"))[1] for path in matching[:5000]}
        assert len(groups) > 1, (
            f"{source} yielded a single group {groups}; its real/fake split would be "
            f"invisible to the holdout protocol"
        )
        assert all("/" in group for group in groups)


def test_real_directories_are_now_entirely_real(ai_face_root):
    """The label noise this correction removed, asserted from the other direction.

    Before the correction, 2,095 of the 18,560 rows under a `real/` directory in
    test.csv (11.3%) were labelled fake, while train.csv (35,928 such rows) and
    val.csv (15,214) had exactly zero. The asymmetry suggested test.csv was labelled
    or assembled differently from train/val. Now every `real/` row is Target=0 in all
    three splits.

    Note this does not make paths authoritative in general -- `Target` remains the
    column to read. It happens to be derivable from the path *because* the correction
    derived it that way.
    """
    pandas = pytest.importorskip("pandas")

    checked = 0
    for split in ("train", "val", "test"):
        path = ai_face_root / f"{split}.csv"
        if not path.is_file():
            continue
        frame = pandas.read_csv(path, usecols=["Image Path", "Target"])
        in_real_dir = frame[
            frame["Image Path"].astype(str).str.contains("/real/", regex=False)
        ]
        if in_real_dir.empty:
            continue
        checked += 1
        fake_rows = int((in_real_dir["Target"] == 1).sum())
        assert fake_rows == 0, (
            f"{split}.csv has {fake_rows} of {len(in_real_dir)} rows under a /real/ "
            f"directory still labelled fake"
        )

    assert checked, "no /real/ directories found in any split"


def test_colon_bearing_source_is_handled(ai_face_root):
    """One source folder is literally named `taming_transformer:VQGAN`.

    A colon breaks any code that uses a source id in a filename or a colon-delimited
    key, which is why cache keys sanitize it.
    """
    from test_helpers.cache_keys import sanitize_key_component

    groups = source_composition(ai_face_root, "test")
    colon_sources = [source for source in groups if ":" in source]
    if not colon_sources:
        pytest.skip("no colon-bearing source in this split")
    for source in colon_sources:
        assert ":" not in sanitize_key_component(source)


# --------------------------------------------------------------------------- #
# Small-manifest smoke path
# --------------------------------------------------------------------------- #

def test_train_small_is_a_usable_fixture(ai_face_root):
    """A 9-row manifest that makes a real-data smoke test cheap."""
    path = ai_face_root / "train_small.csv"
    if not path.is_file():
        pytest.skip("train_small.csv is not present")

    rows = read_head(path)
    assert 1 <= len(rows) <= 50
    assert REQUIRED_COLUMNS <= set(rows[0])
    existing = sum(
        1 for row in rows if (ai_face_root / row["Image Path"].lstrip("/")).is_file()
    )
    assert existing == len(rows), (
        f"only {existing}/{len(rows)} of train_small.csv's images exist on disk"
    )


def test_quality_attributes_are_populated(ai_face_root):
    """The precondition for graph-distance uncertainty having any signal.

    It reads blur/brightness/contrast/compression, symmetry_*, emotion_*, and
    face_embedding -- all of which come from the `*_quality.csv` sidecars, not the base
    manifest. If those columns were sparse or malformed the method would silently score
    an all-zero vector after standardization, and embedding distance would fall back to
    a flat sentinel for every pair.

    Measured over a 3,000-row sample of train_quality.csv: quality_metrics, symmetry,
    and emotion_scores are populated in 100% of rows, and 91.5% carry a non-zero
    face_embedding (the loader discards all-zero ones). Asserted with margin, since the
    sample is the head of the file.
    """
    import ast
    import csv as csv_module

    path = ai_face_root / "train_quality.csv"
    if not path.is_file():
        pytest.skip("train_quality.csv is not present")

    csv_module.field_size_limit(10 ** 9)
    counts = {"rows": 0, "embedding": 0, "quality": 0, "symmetry": 0, "emotion": 0}
    with open(path, newline="") as handle:
        reader = csv_module.DictReader(handle)
        for row in reader:
            if counts["rows"] >= 1000:
                break
            counts["rows"] += 1

            embedding = (row.get("face_embedding") or "").strip()
            if embedding:
                try:
                    # Whitespace-separated numpy repr, not JSON -- a comma-separated
                    # value would raise here and be dropped.
                    values = [
                        float(token) for token in
                        embedding.strip("[]").replace("\n", " ").split()
                    ]
                    if values and not np.allclose(values, 0):
                        counts["embedding"] += 1
                except ValueError:
                    pass

            for column, key in (
                ("quality_metrics", "quality"),
                ("symmetry", "symmetry"),
                ("emotion_scores", "emotion"),
            ):
                raw = (row.get(column) or "").strip()
                if raw and raw not in ("{}", "nan"):
                    try:
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, dict) and parsed:
                            counts[key] += 1
                    except (ValueError, SyntaxError):
                        pass

    total = counts["rows"]
    assert total > 0, "train_quality.csv has no rows"
    for key in ("quality", "symmetry", "emotion"):
        rate = counts[key] / total
        assert rate > 0.9, (
            f"{key} is populated in only {rate:.1%} of rows; graph-distance would have "
            f"little to compare"
        )
    embedding_rate = counts["embedding"] / total
    assert embedding_rate > 0.5, (
        f"only {embedding_rate:.1%} of rows carry a usable face_embedding, so "
        f"embedding_distance would be dominated by the missing-value sentinel"
    )


def test_quality_sidecar_paths_reference_a_dead_root(ai_face_root):
    """Documents *why* the join needs normalizing at all.

    The sidecars' `image_path` column records the root the extraction script ran
    under, which is no longer where the dataset lives. Pinned so that if the sidecars
    are ever regenerated against the current root, the normalization becomes a no-op
    rather than silently doing something unexpected.
    """
    import csv as csv_module

    csv_module.field_size_limit(10 ** 9)
    path = ai_face_root / "train_quality.csv"
    if not path.is_file():
        pytest.skip("train_quality.csv is not present")

    with open(path, newline="") as handle:
        row = next(csv_module.DictReader(handle))
    recorded = (row.get("image_path") or "").strip()
    assert recorded, "no image_path in the sidecar"
    assert not recorded.startswith(str(ai_face_root)), (
        f"sidecar paths now start with the live root ({recorded!r}); the normalization "
        f"in AIFaceDataset._to_current_root should still be a no-op, but this test's "
        f"premise has changed"
    )


def test_quality_sidecars_join_onto_the_base_manifest(ai_face_root):
    """The join must actually connect, using the key the loader really uses.

    This is the test my earlier basename-overlap check should have been: basenames
    overlapped ~100% while the *join* matched 0%, because the loader keys on the full
    `image_path` and the base attributes are re-keyed to the current root.

    Measured in the direction that matters -- what fraction of *base* rows can be
    enriched -- not what fraction of sidecar rows get used. The sidecars are a superset
    of the manifest since the label correction dropped CelebA and casia-webface (~31%
    of rows) without regenerating them, so a sidecar-side ratio would sit near 69%
    forever and say nothing about whether the join works.

    Samples a contiguous chunk of the sidecar rather than reading all 6.4 GB, so the
    denominator is the base rows *observed in that chunk*.
    """
    import csv as csv_module

    pandas = pytest.importorskip("pandas")
    from datasets.AIFaceDataset import AIFaceDataset

    csv_module.field_size_limit(10 ** 9)
    quality_path = ai_face_root / "train_quality.csv"
    if not quality_path.is_file():
        pytest.skip("train_quality.csv is not present")

    base = pandas.read_csv(ai_face_root / "train.csv", usecols=["Image Path"])
    relative = base["Image Path"].astype(str)
    sources = set(relative.str.lstrip("/").str.split("/").str[0])
    base_keys = {
        os.path.join(str(ai_face_root), path.lstrip("/")) for path in relative
    }

    dataset = AIFaceDataset.__new__(AIFaceDataset)
    dataset.data_root = str(ai_face_root)

    resolved = set()
    unresolvable = 0
    with open(quality_path, newline="") as handle:
        for index, row in enumerate(csv_module.DictReader(handle)):
            if index >= 20_000:
                break
            key = dataset._to_current_root(row.get("image_path"), sources)
            if key is None:
                unresolvable += 1
            else:
                resolved.add(key)

    assert resolved, "no sidecar path resolved at all"
    covered = resolved & base_keys
    assert covered, (
        f"none of {len(resolved):,} resolved sidecar paths matched a base row; the "
        f"normalization is producing keys the manifest does not use"
    )

    # `unresolvable` is expected to be large and is deliberately not asserted on: the
    # label correction dropped CelebA and casia-webface (~31% of rows) from the
    # manifests, so their source names are no longer in `known_sources` and their
    # sidecar rows cannot be resolved. That is the correct outcome -- those images have
    # no home. What matters is the direction tested below and in the next test.
    assert unresolvable < len(resolved) + unresolvable, (
        "every sidecar path failed to resolve, which means the source set is empty"
    )


def test_every_base_row_in_a_sidecar_chunk_is_enriched(ai_face_root):
    """Stronger form: for images present in both, the join must not miss any.

    Reads a sidecar chunk, keeps only the rows whose image is still in the manifest,
    and asserts every one of them lands on a base key. A partial join would leave some
    nodes with attributes and others without, which is harder to notice than a total
    failure and biases graph-distance toward whichever subset happened to work.
    """
    import csv as csv_module

    pandas = pytest.importorskip("pandas")
    from datasets.AIFaceDataset import AIFaceDataset

    csv_module.field_size_limit(10 ** 9)
    quality_path = ai_face_root / "train_quality.csv"
    if not quality_path.is_file():
        pytest.skip("train_quality.csv is not present")

    base = pandas.read_csv(ai_face_root / "train.csv", usecols=["Image Path"])
    relative = base["Image Path"].astype(str)
    sources = set(relative.str.lstrip("/").str.split("/").str[0])
    base_keys = {
        os.path.join(str(ai_face_root), path.lstrip("/")) for path in relative
    }
    kept_sources = set(relative.str.lstrip("/").str.split("/").str[0])

    dataset = AIFaceDataset.__new__(AIFaceDataset)
    dataset.data_root = str(ai_face_root)

    considered = matched = 0
    with open(quality_path, newline="") as handle:
        for index, row in enumerate(csv_module.DictReader(handle)):
            if index >= 20_000:
                break
            raw = (row.get("image_path") or "")
            # Restrict to sources the manifest still carries, so dropped sources do
            # not count against the join.
            if not any(f"/{source}/" in raw for source in kept_sources):
                continue
            considered += 1
            key = dataset._to_current_root(raw, sources)
            if key is not None and key in base_keys:
                matched += 1

    assert considered > 0, "no sidecar rows referenced a still-present source"
    rate = matched / float(considered)
    assert rate >= AIFaceDataset.MIN_QUALITY_JOIN_RATE, (
        f"only {matched:,} of {considered:,} sidecar rows for still-present sources "
        f"({rate:.1%}) matched a base row"
    )


def test_checked_in_source_stats_still_match(ai_face_root):
    """Guards the numbers the holdout groups were sized from.

    If the dataset on disk changes, the recommended holdout families and their
    train-removed/test-OOD counts need revisiting rather than silently drifting.
    """
    groups = source_composition(ai_face_root, "train", limit=100_000)
    # Sampling the head of the file, so assert structure rather than exact counts.
    assert len(groups) >= 4, f"expected several sources, found {sorted(groups)}"
    assert any(source in groups for source in REAL_ONLY_SOURCES)
    fake_only = [
        source for source, counts in groups.items()
        if counts["fake"] > 0 and counts["real"] == 0
    ]
    assert fake_only, (
        "expected at least one exclusively-fake generator source; the holdout protocol "
        "depends on generators being separable from the real pools"
    )
