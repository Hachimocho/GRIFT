#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

OLD_PREFIX = '/home/brg2890/major/preprocessed/FaceForensics++_All'
NEW_PREFIX = '/shared/rc/defake/FaceForensics++_All'


def remap_paths(df: pd.DataFrame, old_prefix: str, new_prefix: str) -> pd.DataFrame:
    # Replace prefix in all string/object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(old_prefix, new_prefix, regex=False)
        df[col] = df[col].astype(str).str.replace("OUTDATED_FaceShifter", "FaceShifter", regex=False)
        df[col] = df[col].astype(str).str.replace("DeepFakes/train", "DeepFakes/train/altered", regex=False)
        df[col] = df[col].astype(str).str.replace("DeepFakes/val", "DeepFakes/val/altered", regex=False)
        df[col] = df[col].astype(str).str.replace("DeepFakes/test", "DeepFakes/test/altered", regex=False)
    return df


def drop_image_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'image_id' in df.columns:
        df = df.drop(columns=['image_id'])
    return df


def filter_no_errors(df: pd.DataFrame) -> pd.DataFrame:
    if 'error' in df.columns:
        # Keep rows where error is null/empty
        no_err = df[(df['error'].isna()) | (df['error'].astype(str).str.strip() == '')]
        return no_err
    return df


def filter_misc_deepfakes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out entries with misc/DeepFakes paths from all string/object columns"""
    original_len = len(df)
    
    # Create a mask to identify rows with misc/DeepFakes paths
    mask = pd.Series([False] * len(df), index=df.index)
    
    # Check all string/object columns for misc/DeepFakes paths
    for col in df.select_dtypes(include=['object']).columns:
        col_mask = df[col].astype(str).str.contains('misc/DeepFakes', na=False)
        mask = mask | col_mask
    
    # Keep rows that don't have misc/DeepFakes paths
    filtered_df = df[~mask]
    filtered_len = len(filtered_df)
    
    print(f"Filtered out {original_len - filtered_len} entries with misc/DeepFakes paths")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description='Remap FaceForensics CSV paths and generate copies')
    parser.add_argument('--input', type=str, default='faceforensics.csv', help='Input CSV path')
    parser.add_argument('--out', type=str, default='faceforensics_shared.csv', help='Output CSV with remapped paths')
    parser.add_argument('--out-no-errors', type=str, default='faceforensics_shared_no_errors.csv', help='Output CSV with remapped paths and rows with errors removed')
    parser.add_argument('--old-prefix', type=str, default=OLD_PREFIX, help='Old path prefix to replace')
    parser.add_argument('--new-prefix', type=str, default=NEW_PREFIX, help='New path prefix to use')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input CSV not found: {args.input}")
        sys.exit(1)

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    n0 = len(df)

    print(f"Remapping paths: '{args.old_prefix}' -> '{args.new_prefix}'")
    df = remap_paths(df, args.old_prefix, args.new_prefix)

    print("Dropping column: image_id (if present)")
    df = drop_image_id(df)

    print("Filtering out entries with misc/DeepFakes paths...")
    df = filter_misc_deepfakes(df)
    n_filtered = len(df)

    print(f"Writing remapped CSV: {args.out}")
    df.to_csv(args.out, index=False)

    print("Filtering rows with errors (if 'error' column exists)...")
    df_no_err = filter_no_errors(df)
    n1 = len(df_no_err)

    print(f"Writing no-errors CSV: {args.out_no_errors} (kept {n1}/{n_filtered} rows)")
    df_no_err.to_csv(args.out_no_errors, index=False)

    print("Done.")


if __name__ == '__main__':
    main()
