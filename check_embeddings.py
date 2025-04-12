#!/usr/bin/env python3
"""
Check embeddings in quality CSV files to verify if they're valid or all zeros.
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from collections import defaultdict
import ast

def analyze_embeddings(csv_path, sample_limit=10):
    """
    Analyze the face embeddings in a quality CSV file
    
    Args:
        csv_path: Path to the quality CSV file
        sample_limit: Number of embeddings to sample for detailed analysis
    
    Returns:
        Dictionary with embedding statistics
    """
    print(f"Analyzing embeddings in: {csv_path}")
    
    try:
        # First try to load with default settings
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV with default settings: {e}")
        try:
            # Try with different encoding
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"Error loading CSV with utf-8 encoding: {e}")
            try:
                # Try with latin1 encoding
                df = pd.read_csv(csv_path, encoding='latin1')
            except Exception as e:
                print(f"Failed to load CSV file: {e}")
                return None
    
    print(f"CSV loaded successfully. Shape: {df.shape}")
    
    # Identify potential embedding columns
    embedding_columns = [col for col in df.columns if 'embedding' in col.lower()]
    print(f"Potential embedding columns: {embedding_columns}")
    
    if not embedding_columns:
        print("No embedding columns found!")
        return None
    
    results = {}
    
    for col in embedding_columns:
        print(f"\nAnalyzing column: {col}")
        
        # Check if column exists
        if col not in df.columns:
            print(f"Column {col} not found in DataFrame")
            continue
        
        # Get non-null count
        non_null_count = df[col].count()
        print(f"Non-null values: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.2f}%)")
        
        # Initialize counters
        zero_embeddings = 0
        valid_embeddings = 0
        error_count = 0
        embedding_lengths = defaultdict(int)
        embedding_sample = []
        magnitudes = []
        
        # Sample some rows for more detailed analysis
        sampled_rows = min(len(df), sample_limit)
        
        # Check each embedding
        for idx, embedding_val in enumerate(df[col].dropna()):
            try:
                # Handle various formats
                if isinstance(embedding_val, str):
                    # First try standard formats
                    try:
                        # Try to parse as JSON
                        embedding = json.loads(embedding_val)
                    except:
                        try:
                            # Try to parse as Python literal
                            embedding = ast.literal_eval(embedding_val)
                        except:
                            try:
                                # Try to handle comma-separated string
                                embedding = [float(x) for x in embedding_val.strip('[]').split(',')]
                            except:
                                # As per user instructions for scientific notation format with newlines
                                try:
                                    # 1. Strip all newline characters
                                    clean_str = embedding_val.replace('\n', '')
                                    
                                    # 2. Replace all multiple spaces with single spaces
                                    while '  ' in clean_str:
                                        clean_str = clean_str.replace('  ', ' ')
                                    
                                    # Trim any surrounding whitespace or brackets
                                    clean_str = clean_str.strip().strip('[]')
                                    
                                    # 3. Split by spaces and filter out empty strings
                                    parts = [p for p in clean_str.split(' ') if p]
                                    
                                    # 4. Convert to floats
                                    embedding = [float(x) for x in parts]
                                    
                                    if not embedding:  # If we got an empty list
                                        print(f"Warning: Parsed to empty list from: {embedding_val[:100]}...")
                                        embedding = np.zeros(512)  # Default size for face embeddings
                                        
                                except Exception as e:
                                    print(f"Could not parse embedding: {str(e)[:100]}...")
                                    print(f"Sample of problematic string: {embedding_val[:100]}...")
                                    # Default to zeros but don't crash
                                    embedding = np.zeros(512)  # Default size for face embeddings
                else:
                    embedding = embedding_val
                
                # Convert to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Record the length
                embedding_lengths[len(embedding)] += 1
                
                # Check if it's all zeros
                if np.all(np.isclose(embedding, 0)):
                    zero_embeddings += 1
                else:
                    valid_embeddings += 1
                    
                # Calculate magnitude
                magnitude = np.linalg.norm(embedding)
                magnitudes.append(magnitude)
                
                # Save sample
                if idx < sampled_rows:
                    embedding_sample.append({
                        'length': len(embedding),
                        'magnitude': magnitude,
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding)),
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'zeros': int(np.sum(np.isclose(embedding, 0))),
                        'is_all_zeros': bool(np.all(np.isclose(embedding, 0)))
                    })
                
            except Exception as e:
                error_count += 1
                if idx < sampled_rows:
                    print(f"Error processing row {idx}: {e}")
                    print(f"Value: {type(embedding_val)} - {embedding_val[:100]}...")
        
        # Compile statistics
        total_processed = zero_embeddings + valid_embeddings + error_count
        
        col_results = {
            'total_rows': len(df),
            'processed_embeddings': total_processed,
            'valid_embeddings': valid_embeddings,
            'zero_embeddings': zero_embeddings,
            'error_count': error_count,
            'embedding_lengths': dict(embedding_lengths),
            'samples': embedding_sample,
        }
        
        if magnitudes:
            col_results['magnitude_stats'] = {
                'min': float(np.min(magnitudes)),
                'max': float(np.max(magnitudes)),
                'mean': float(np.mean(magnitudes)),
                'median': float(np.median(magnitudes)),
                'std': float(np.std(magnitudes))
            }
        
        results[col] = col_results
        
        # Print summary
        print(f"Total processed: {total_processed}")
        print(f"Valid embeddings: {valid_embeddings} ({valid_embeddings/total_processed*100:.2f}%)")
        print(f"Zero embeddings: {zero_embeddings} ({zero_embeddings/total_processed*100:.2f}%)")
        print(f"Processing errors: {error_count} ({error_count/total_processed*100:.2f}%)")
        print(f"Embedding lengths: {dict(embedding_lengths)}")
        
        if magnitudes:
            print(f"Magnitude stats:")
            print(f"  Min: {np.min(magnitudes):.4f}")
            print(f"  Max: {np.max(magnitudes):.4f}")
            print(f"  Mean: {np.mean(magnitudes):.4f}")
            print(f"  Median: {np.median(magnitudes):.4f}")
        
        # Print some samples
        if embedding_sample:
            print("\nSample embeddings:")
            for i, sample in enumerate(embedding_sample):
                print(f"  Sample {i+1}:")
                print(f"    Length: {sample['length']}")
                print(f"    Magnitude: {sample['magnitude']:.4f}")
                print(f"    Range: [{sample['min']:.4f}, {sample['max']:.4f}]")
                print(f"    Mean: {sample['mean']:.4f}")
                print(f"    Std: {sample['std']:.4f}")
                print(f"    Zeros: {sample['zeros']}/{sample['length']} ({sample['zeros']/sample['length']*100:.2f}%)")
                print(f"    All zeros: {sample['is_all_zeros']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze embeddings in quality CSV files")
    parser.add_argument("--csv", type=str, help="Path to the quality CSV file to analyze")
    parser.add_argument("--data-root", type=str, default=".", help="Path to the data root directory to search for quality CSVs")
    parser.add_argument("--samples", type=int, default=10, help="Number of embeddings to sample for detailed analysis")
    parser.add_argument("--check-filenames", action="store_true", help="Check filename formats in the CSV")
    
    args = parser.parse_args()
    
    if args.csv:
        # Analyze specific CSV file
        if args.check_filenames:
            analyze_csv_filenames(args.csv)
        else:
            analyze_embeddings(args.csv, args.samples)
    else:
        # Search for quality CSVs in data root
        print(f"Searching for quality CSVs in: {args.data_root}")
        quality_csvs = []
        
        for root, _, files in os.walk(args.data_root):
            for file in files:
                if file.endswith("_quality.csv") or "quality" in file.lower() and file.endswith(".csv"):
                    quality_csvs.append(os.path.join(root, file))
        
        print(f"Found {len(quality_csvs)} quality CSV files")
        
        if not quality_csvs:
            print("No quality CSV files found!")
            return
        
        # Analyze each CSV
        for csv_path in quality_csvs:
            if args.check_filenames:
                analyze_csv_filenames(csv_path)
            else:
                analyze_embeddings(csv_path, args.samples)
            print("\n" + "="*80 + "\n")

def analyze_csv_filenames(csv_path):
    """Analyze the filename formats in a CSV file to identify potential matching issues
    
    Args:
        csv_path: Path to the CSV file to analyze
    """
    print(f"Analyzing filename formats in: {csv_path}")
    
    try:
        # First try to load with default settings
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV with default settings: {e}")
        try:
            # Try with different encoding
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"Error loading CSV with utf-8 encoding: {e}")
            try:
                # Try with latin1 encoding
                df = pd.read_csv(csv_path, encoding='latin1')
            except Exception as e:
                print(f"Failed to load CSV file: {e}")
                return
    
    print(f"CSV loaded successfully. Shape: {df.shape}")
    
    # Identify filename columns
    filename_columns = []
    for col in df.columns:
        if any(name in col.lower() for name in ['file', 'path', 'image', 'img', 'id']):
            filename_columns.append(col)
    
    print(f"Potential filename columns: {filename_columns}")
    
    for col in filename_columns:
        print(f"\nAnalyzing column: {col}")
        
        # Check for null values
        null_count = df[col].isnull().sum()
        print(f"Null values: {null_count}/{len(df)} ({null_count/len(df)*100:.2f}%)")
        
        # Check for unique values
        unique_count = df[col].nunique()
        print(f"Unique values: {unique_count}/{len(df)} ({unique_count/len(df)*100:.2f}%)")
        
        # Get sample of values
        sample_values = df[col].dropna().sample(min(5, len(df))).tolist()
        print(f"Sample values: {sample_values}")
        
        # Check format characteristics
        has_full_path = False
        has_relative_path = False
        has_basename_only = False
        has_extension = False
        
        # Get all values - limit to 1000 for performance
        values = df[col].dropna().values[:1000]
        
        for val in values:
            if not isinstance(val, str):
                continue
                
            # Check for path separators
            if '/' in val:
                if val.startswith('/'):
                    has_full_path = True
                else:
                    has_relative_path = True
            else:
                has_basename_only = True
                
            # Check for file extension
            if '.' in os.path.basename(val):
                has_extension = True
        
        print(f"Format characteristics:")
        print(f"  Contains absolute paths: {has_full_path}")
        print(f"  Contains relative paths: {has_relative_path}")
        print(f"  Contains basename only: {has_basename_only}")
        print(f"  Contains file extensions: {has_extension}")
        
        # Check for os.path.basename conversion impact
        if has_full_path or has_relative_path:
            original_unique = len(set(values))
            basename_values = [os.path.basename(val) if isinstance(val, str) else val for val in values]
            basename_unique = len(set(basename_values))
            
            collision_rate = 1 - (basename_unique / original_unique) if original_unique > 0 else 0
            print(f"\nBasename conversion impact:")
            print(f"  Original unique values: {original_unique}")
            print(f"  After basename conversion: {basename_unique}")
            print(f"  Collision rate: {collision_rate*100:.2f}%")
            
            if collision_rate > 0:
                print(f"WARNING: Converting to basename causes {collision_rate*100:.2f}% of filenames to collide!")
                print(f"This could be the reason embeddings aren't being properly associated with nodes.")

if __name__ == "__main__":
    main()
