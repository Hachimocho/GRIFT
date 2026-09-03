#!/usr/bin/env python3
"""Correct the AI-Face split manifests: relabel, drop non-real sources, normalize EOL.

The labelling rule (AI-Face v2): an image is **real** iff it comes from one of the six
real source corpora --

    FFHQ, IMDB-WIKI (the `wiki/` folder), and the *real* portions of
    FF++, DFDC, DFD, and Celeb-DF-v2

-- and **fake** otherwise. Rows from `CelebA/` and `casia-webface/` are **dropped**,
matching what AI-Face v2 did when it removed them as real sources (it removed them
rather than relabelling them, and relabelling ~514k genuine photographs as fake would
teach a detector that real photos are fake).

Two things this fixes that are worth naming:

* **`test.csv` carries ~0.9% random label noise.** In train and val, `Target` is
  already a perfect function of the path. In test, 3,825 of 422,352 rows disagree with
  what the path implies, spread uniformly through the file.
* **The real/fake encoding is inconsistent.** `ff++` and `celebdf` use `real/` vs
  `crop_img/`; `dfdc` and `dfd` use `real/` vs `fake/`. So the predicate must be
  `second component == "real"`, never `!= "fake"`.

A caveat to carry forward: the six real sources hold only 142,951 images in this copy,
against AI-Face v2's published 400,885. The corrected dataset is therefore ~13% real
rather than ~24%, and every prior-sensitive metric (Brier, NLL, ECE via the base rate)
will reflect that. Recovering the missing real data means re-downloading.

Only the four base manifests are touched. The `*_quality.csv` sidecars have no
`Target` column, and they must never be processed line-wise: the `face_embedding`
field is a multi-line numpy repr, so `wc -l` over-counts by ~109x and sed/awk would
split rows.

Usage:
    python development_tools/correct_aiface_labels.py --dry-run      # report only
    python development_tools/correct_aiface_labels.py --apply        # rewrite in place
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

#: Sources whose every image is real.
REAL_ONLY_SOURCES = frozenset({"FFHQ", "wiki"})

#: Video sources holding both classes, split by their second path component.
#: The component naming is deliberately inconsistent upstream, hence REAL_SUBDIR.
MIXED_SOURCES = frozenset({"ff++", "dfdc", "dfd", "celebdf"})

#: The one second-level component that means "real" in every mixed source. Its
#: counterpart is `crop_img/` for ff++/celebdf and `fake/` for dfdc/dfd, which is
#: exactly why we test for `real` rather than against `fake`.
REAL_SUBDIR = "real"

#: Sources removed entirely. Real photographs, but not part of AI-Face v2's real set.
DROP_SOURCES = frozenset({"CelebA", "casia-webface"})

MANIFESTS = ("train.csv", "val.csv", "test.csv", "train_small.csv")

BASE_COLUMNS = [
    "Image Path", "Uncertainty Score Gender", "Uncertainty Score Age",
    "Uncertainty Score Race", "Ground Truth Gender", "Ground Truth Age",
    "Ground Truth Race", "Intersection", "Target",
]


def path_components(image_path):
    """(source, second_component) for a manifest `Image Path`."""
    parts = [part for part in str(image_path).replace("\\", "/").split("/") if part]
    if not parts:
        return "", ""
    return parts[0], (parts[1] if len(parts) > 1 else "")


def correct_target(image_path):
    """The label this path should carry: 0 real, 1 fake."""
    source, second = path_components(image_path)
    if source in REAL_ONLY_SOURCES:
        return 0
    if source in MIXED_SOURCES and second == REAL_SUBDIR:
        return 0
    return 1


def should_drop(image_path):
    return path_components(image_path)[0] in DROP_SOURCES


def analyze(frame):
    """Per-source before/after counts for one manifest."""
    sources = frame["Image Path"].map(lambda path: path_components(path)[0])
    corrected = frame["Image Path"].map(correct_target)
    dropped = frame["Image Path"].map(should_drop)

    report = defaultdict(lambda: {
        "rows": 0, "dropped": 0, "target0_before": 0, "target1_before": 0,
        "target0_after": 0, "target1_after": 0, "flip_0_to_1": 0, "flip_1_to_0": 0,
    })
    for source, before, after, drop in zip(
        sources, frame["Target"].astype(int), corrected, dropped
    ):
        entry = report[source]
        entry["rows"] += 1
        entry[f"target{before}_before"] += 1
        if drop:
            entry["dropped"] += 1
            continue
        entry[f"target{after}_after"] += 1
        if before != after:
            entry["flip_0_to_1" if after == 1 else "flip_1_to_0"] += 1
    return dict(report), corrected, dropped


def detect_line_ending(path):
    with open(path, "rb") as handle:
        first = handle.readline()
    return "CRLF" if first.endswith(b"\r\n") else "LF"


def process(root, name, apply_changes, backup_dir):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        print(f"  {name}: not present, skipping")
        return None

    original_ending = detect_line_ending(path)
    # pandas handles either convention on read; the manifests disagree (test.csv is
    # CRLF, the rest LF) and we normalize everything to LF on write.
    frame = pd.read_csv(path)
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing expected columns {missing}")

    report, corrected, dropped = analyze(frame)
    rows_before = len(frame)
    keep = ~dropped
    result = frame.loc[keep].copy()
    result["Target"] = corrected.loc[keep].astype(int).values
    rows_after = len(result)

    total_dropped = int(dropped.sum())
    flips_0_1 = sum(entry["flip_0_to_1"] for entry in report.values())
    flips_1_0 = sum(entry["flip_1_to_0"] for entry in report.values())
    real_after = int((result["Target"] == 0).sum())

    print(f"\n  {name}  ({original_ending} -> LF)")
    print(f"    rows: {rows_before:,} -> {rows_after:,}  (dropped {total_dropped:,})")
    print(f"    Target corrections among kept rows: {flips_0_1:,} real->fake, "
          f"{flips_1_0:,} fake->real")
    print(f"    real after: {real_after:,} / {rows_after:,} "
          f"({100.0 * real_after / max(1, rows_after):.1f}%)")

    changed_sources = {
        source: entry for source, entry in sorted(report.items())
        if entry["dropped"] or entry["flip_0_to_1"] or entry["flip_1_to_0"]
    }
    for source, entry in changed_sources.items():
        note = []
        if entry["dropped"]:
            note.append(f"dropped {entry['dropped']:,}")
        if entry["flip_0_to_1"]:
            note.append(f"{entry['flip_0_to_1']:,} real->fake")
        if entry["flip_1_to_0"]:
            note.append(f"{entry['flip_1_to_0']:,} fake->real")
        print(f"      {source:28s} {', '.join(note)}")

    # Post-conditions: the whole point is that Target becomes a pure function of the
    # path, with no dropped source surviving.
    recomputed = result["Image Path"].map(correct_target)
    assert (result["Target"].astype(int) == recomputed).all(), (
        f"{name}: Target is not a pure function of the path after correction"
    )
    assert not result["Image Path"].map(should_drop).any(), (
        f"{name}: rows from a dropped source survived"
    )
    assert list(result.columns) == list(frame.columns), f"{name}: column set changed"

    if apply_changes:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, name)
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)
            print(f"    backed up -> {backup_path}")
        else:
            print(f"    backup already exists, keeping it: {backup_path}")

        # Write to a temp file then rename, so an interrupted run cannot leave a
        # partial manifest in place of a good one.
        temporary = path + ".tmp"
        result.to_csv(temporary, index=False, lineterminator="\n")
        verify = pd.read_csv(temporary)
        assert len(verify) == rows_after, "row count changed on write"
        assert list(verify.columns) == list(frame.columns), "columns changed on write"
        assert detect_line_ending(temporary) == "LF", "temp file is not LF"
        os.replace(temporary, path)
        print(f"    written ({rows_after:,} rows, LF)")

    return {
        "file": name,
        "line_ending_before": original_ending,
        "line_ending_after": "LF",
        "rows_before": rows_before,
        "rows_after": rows_after,
        "dropped": total_dropped,
        "flips_real_to_fake": flips_0_1,
        "flips_fake_to_real": flips_1_0,
        "real_after": real_after,
        "per_source": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default=None,
                        help="AI-Face root (default: auto-discovered)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report without writing")
    group.add_argument("--apply", action="store_true", help="rewrite the manifests")
    parser.add_argument("--report", default=None,
                        help="write a JSON summary here (default: <root>/label_correction.json)")
    args = parser.parse_args()

    from test_helpers.data_graph_utils import resolve_ai_face_data_root
    root = args.data_root or resolve_ai_face_data_root()

    print(f"AI-Face root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print("\nRule: real iff source in {FFHQ, wiki} or (source in "
          "{ff++, dfdc, dfd, celebdf} and second component == 'real'); "
          "rows from {CelebA, casia-webface} dropped.")

    backup_dir = os.path.join(root, "csv_backups")
    summaries = []
    for name in MANIFESTS:
        summary = process(root, name, args.apply, backup_dir)
        if summary:
            summaries.append(summary)

    total_before = sum(entry["rows_before"] for entry in summaries)
    total_after = sum(entry["rows_after"] for entry in summaries)
    total_real = sum(entry["real_after"] for entry in summaries)
    print(f"\nTotals: {total_before:,} -> {total_after:,} rows; "
          f"real {total_real:,} ({100.0 * total_real / max(1, total_after):.1f}%)")
    print(f"  dropped: {sum(e['dropped'] for e in summaries):,}")
    print(f"  corrections: {sum(e['flips_real_to_fake'] for e in summaries):,} real->fake, "
          f"{sum(e['flips_fake_to_real'] for e in summaries):,} fake->real")
    print("\nNote: AI-Face v2 publishes 400,885 real images; this copy yields "
          f"{total_real:,}. The corrected split is more imbalanced than the published "
          "benchmark, which will move Brier, NLL, and ECE via the base rate.")

    if args.apply:
        report_path = args.report or os.path.join(root, "label_correction.json")
        with open(report_path, "w") as handle:
            json.dump({
                "rule": {
                    "real_only_sources": sorted(REAL_ONLY_SOURCES),
                    "mixed_sources": sorted(MIXED_SOURCES),
                    "real_subdir": REAL_SUBDIR,
                    "dropped_sources": sorted(DROP_SOURCES),
                },
                "totals": {
                    "rows_before": total_before, "rows_after": total_after,
                    "real_after": total_real,
                },
                "files": summaries,
            }, handle, indent=2, sort_keys=True)
        print(f"\nSummary written to {report_path}")
    else:
        print("\nNothing written. Re-run with --apply to rewrite the manifests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
