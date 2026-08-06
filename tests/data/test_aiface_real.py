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

#: Sources measured to be exclusively real in the train split.
REAL_ONLY_SOURCES = {"casia-webface", "CelebA", "FFHQ", "wiki"}
#: Video sets whose second path component carries real/fake.
MIXED_SOURCES = {"celebdf", "ff++", "dfd", "dfdc"}


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


def test_line_endings_are_mixed_across_splits(ai_face_root):
    """The manifests do NOT agree on line endings.

    Measured: `test.csv` is CRLF, while `train.csv`, `val.csv`, and `train_small.csv`
    are LF. Inconsistency is worse than either convention on its own -- code that
    works on train silently mis-parses test, where the final column becomes "0\\r"
    rather than "0". `int("0\\r")` happens to succeed, but a string comparison against
    "0" fails and `awk -F,` yields a trailing carriage return.

    pandas and `csv.DictReader` (with `newline=""`) both handle either, which is why
    this only bites hand-rolled parsing. Pinned so the asymmetry is documented rather
    than rediscovered.
    """
    endings = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        path = ai_face_root / name
        if not path.is_file():
            continue
        with open(path, "rb") as handle:
            endings[name] = "CRLF" if handle.readline().endswith(b"\r\n") else "LF"

    assert endings, "no manifests found"
    assert len(set(endings.values())) > 1, (
        f"line endings are now uniform ({endings}); if the dataset was normalized, "
        f"this test can be relaxed -- but any hand-rolled parser should still handle both"
    )


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


def test_real_only_sources_are_predominantly_real(ai_face_root):
    """Underpins the holdout design: these are the negative-class pools.

    Holding one of them out would remove a large share of the real data and change the
    training class prior, confounding distribution shift with class-imbalance shift --
    which is why the protocol holds out generators, not real sources.
    """
    groups = source_composition(ai_face_root, "train")
    seen = REAL_ONLY_SOURCES & set(groups)
    assert seen, f"none of {sorted(REAL_ONLY_SOURCES)} found; expected the real pools"
    for source in sorted(seen):
        counts = groups[source]
        total = counts["real"] + counts["fake"]
        assert counts["real"] / total > 0.95, (
            f"{source} is {counts['fake']}/{total} fake, not a real-only pool"
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


def test_real_directory_label_noise_is_test_split_only(ai_face_root):
    """`Target` is authoritative; a `real/` directory is not.

    Measured over the full manifests: 11.3% of the 18,560 rows under a `real/`
    directory in **test.csv** are labeled fake (2,095 rows), while train.csv (35,928
    rows) and val.csv (15,214 rows) have exactly zero.

    That asymmetry matters beyond parsing hygiene. A path-based label shortcut would
    appear to work perfectly in training and then be wrong for 2,095 test rows -- and
    it suggests the test split was labeled or assembled differently from train/val,
    which is worth knowing before reading any cross-split result.
    """
    pandas = pytest.importorskip("pandas")

    fractions = {}
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
        fractions[split] = (float(in_real_dir["Target"].mean()), len(in_real_dir))

    assert fractions, "no /real/ directories found in any split"
    assert "test" in fractions, "expected /real/ rows in the test split"

    test_fraction, test_rows = fractions["test"]
    assert test_fraction > 0.05, (
        f"expected substantial label noise under test.csv's /real/ directories, "
        f"found {test_fraction:.1%} of {test_rows} rows"
    )
    for split in ("train", "val"):
        if split in fractions:
            split_fraction, _ = fractions[split]
            assert split_fraction == 0.0, (
                f"{split}.csv now has {split_fraction:.1%} fake rows under /real/; "
                f"previously zero. The train/test labeling asymmetry has changed and "
                f"any conclusion drawn from it should be revisited"
            )


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
