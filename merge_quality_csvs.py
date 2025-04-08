#!/usr/bin/env python3
"""
Merge script for quality CSV parts.
Combines multiple partial quality CSV files into a single final CSV file.
"""

import os
import sys
import glob
import argparse
import pandas as pd


def merge_csv_parts(input_dir, output_path, pattern=None):
    """Merge multiple CSV files into one.
    
    Args:
        input_dir: Directory containing the CSV parts
        output_path: Path where the merged CSV will be saved
        pattern: Glob pattern to match the CSV parts (e.g., "train_part*_quality.csv")
    
    Returns:
        Number of rows in the merged CSV
    """
    # Determine the pattern if not provided
    if pattern is None:
        # Try to guess from the files in the directory
        csvs = glob.glob(os.path.join(input_dir, "*_part*_quality.csv"))
        if not csvs:
            print(f"Error: No CSV parts found in {input_dir}")
            return 0
            
        # Extract the base pattern from the first file
        base_name = os.path.basename(csvs[0])
        prefix = base_name.split("_part")[0]
        pattern = f"{prefix}_part*_quality.csv"
    
    # Get all matching files
    file_pattern = os.path.join(input_dir, pattern)
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"Error: No files found matching pattern: {file_pattern}")
        return 0
    
    print(f"Found {len(csv_files)} CSV parts to merge:")
    for csv_file in sorted(csv_files):
        file_size = os.path.getsize(csv_file) / (1024 * 1024)  # Size in MB
        try:
            row_count = len(pd.read_csv(csv_file))
            print(f"  - {os.path.basename(csv_file)}: {row_count} rows, {file_size:.2f} MB")
        except Exception as e:
            print(f"  - {os.path.basename(csv_file)}: Error reading file: {str(e)}")
    
    # Read and concatenate all CSV files
    print(f"\nMerging files...")
    dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  - Added {os.path.basename(csv_file)}: {len(df)} rows")
        except Exception as e:
            print(f"  - Error reading {os.path.basename(csv_file)}: {str(e)}")
    
    if not dfs:
        print("Error: No valid CSV files found to merge")
        return 0
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save the merged dataframe
    print(f"\nSaving merged CSV to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    print(f"Successfully merged {len(dfs)} CSV files into {output_path}")
    print(f"Total rows in merged CSV: {len(merged_df)}")
    
    return len(merged_df)


def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV parts into a single file")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing the CSV parts")
    parser.add_argument("--output", "-o", required=True, help="Path for the merged output CSV")
    parser.add_argument("--pattern", "-p", help="Glob pattern to match CSV parts (e.g., 'train_part*_quality.csv')")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Merge the CSV parts
    row_count = merge_csv_parts(
        args.input_dir,
        args.output,
        args.pattern
    )
    
    if row_count == 0:
        print("Merge failed or resulted in an empty file")
        sys.exit(1)
    else:
        print(f"Merge completed successfully with {row_count} total rows")


if __name__ == "__main__":
    main()
