#!/usr/bin/env python3
"""
CSV Splitter for parallel attribute generation.
This script divides a large CSV file into multiple parts for parallel processing.
"""

import os
import sys
import math
import argparse
import pandas as pd
from pathlib import Path


def split_csv(input_path, output_dir, num_parts=2, prefix="part"):
    """Split a CSV file into multiple parts.
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save the output files
        num_parts: Number of parts to split the CSV into
        prefix: Prefix for the output files
    
    Returns:
        List of paths to the generated CSV files
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Calculate rows per part (rounded up to ensure all rows are included)
    rows_per_part = math.ceil(len(df) / num_parts)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    output_paths = []
    
    # Split the DataFrame and save each part
    for i in range(num_parts):
        start_idx = i * rows_per_part
        end_idx = min((i + 1) * rows_per_part, len(df))
        
        if start_idx >= len(df):
            break
            
        part_df = df.iloc[start_idx:end_idx]
        
        # Create output path
        output_path = os.path.join(output_dir, f"{name_without_ext}_{prefix}{i+1}.csv")
        
        # Save the part
        part_df.to_csv(output_path, index=False)
        output_paths.append(output_path)
        
        print(f"Part {i+1}: {len(part_df)} rows saved to {output_path}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple parts for parallel processing")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the output files")
    parser.add_argument("--num-parts", "-n", type=int, default=2, help="Number of parts to split into (default: 2)")
    parser.add_argument("--prefix", "-p", default="part", help="Prefix for output files (default: 'part')")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist or is not a file.")
        sys.exit(1)
    
    # Split the CSV
    output_paths = split_csv(
        args.input,
        args.output_dir,
        args.num_parts,
        args.prefix
    )
    
    print(f"\nSuccessfully split {args.input} into {len(output_paths)} parts.")
    print(f"Total rows: {sum([len(pd.read_csv(path)) for path in output_paths])}")


if __name__ == "__main__":
    main()
