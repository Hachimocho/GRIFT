import pandas as pd
import os
import numpy as np
from typing import List
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path

def split_dataframe(df: pd.DataFrame, num_segments: int = 100) -> List[pd.DataFrame]:
    """Split a dataframe into approximately equal segments."""
    segment_size = len(df) // num_segments
    remainder = len(df) % num_segments
    
    segments = []
    start_idx = 0
    
    for i in range(num_segments):
        # Add one extra row to some segments to handle the remainder
        current_size = segment_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        segment = df.iloc[start_idx:end_idx].copy()
        segments.append(segment)
        start_idx = end_idx
    
    return segments

def process_segment(segment_df: pd.DataFrame, segment_id: int, data_root: str, output_dir: str) -> str:
    """Process a single segment and save results to a CSV file."""
    # Create segment output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path for this segment's results
    output_path = os.path.join(output_dir, f'segment_{segment_id}_results.csv')
    
    # Check if this segment has already been processed
    if os.path.exists(output_path):
        logging.info(f"Segment {segment_id} already processed, skipping...")
        return output_path
    
    # Save segment to temporary CSV
    temp_input_path = os.path.join(output_dir, f'segment_{segment_id}_input.csv')
    segment_df.to_csv(temp_input_path, index=False)
    
    # Run the original script on this segment
    cmd = [
        'python', 
        os.path.join(os.path.dirname(__file__), 'additional_attributes.py'),
        '--data_root', data_root,
        '--metadata_path', temp_input_path,
        '--output_path', output_path,
        '--batch_size', '256'  # Using default batch size
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Clean up temporary input file
        os.remove(temp_input_path)
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing segment {segment_id}: {str(e)}")
        return None

def combine_results(result_files: List[str], output_path: str):
    """Combine multiple CSV result files into a single file."""
    dfs = []
    for file in result_files:
        if file and os.path.exists(file):
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error reading {file}: {str(e)}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        logging.info(f"Combined results saved to {output_path}")
        
        # Only remove segment files after successful combination
        for file in result_files:
            if file and os.path.exists(file):
                os.remove(file)
    else:
        logging.error("No valid result files to combine")

def get_completed_segments(temp_dir: str) -> List[int]:
    """Get list of segment IDs that have already been processed."""
    completed = []
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            if file.startswith('segment_') and file.endswith('_results.csv'):
                try:
                    segment_id = int(file.split('_')[1])
                    completed.append(segment_id)
                except (ValueError, IndexError):
                    continue
    return completed

def parallel_process(input_csv: str, data_root: str, output_csv: str, num_segments: int = 100, max_workers: int = None):
    """Main function to handle parallel processing of the input CSV."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    logging.info(f"Processing {len(df)} rows from {input_csv}")
    
    # Create segments
    segments = split_dataframe(df, num_segments)
    logging.info(f"Split data into {len(segments)} segments")
    
    # Create temporary directory for segment results
    temp_dir = os.path.join(os.path.dirname(output_csv), 'temp_segments')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get list of already completed segments
    completed_segments = get_completed_segments(temp_dir)
    logging.info(f"Found {len(completed_segments)} previously completed segments")
    
    # Process segments in parallel
    result_files = []
    # Add paths of completed segments to result_files
    for segment_id in completed_segments:
        result_files.append(os.path.join(temp_dir, f'segment_{segment_id}_results.csv'))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, segment in enumerate(segments):
            if i not in completed_segments:
                future = executor.submit(process_segment, segment, i, data_root, temp_dir)
                futures.append(future)
        
        # Track progress with tqdm
        if futures:  # Only show progress bar if there are segments to process
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing remaining segments"):
                result_file = future.result()
                if result_file:
                    result_files.append(result_file)
        else:
            logging.info("All segments already processed")
    
    # Combine results
    combine_results(result_files, output_csv)
    
    # Clean up temporary directory if all processing is complete
    try:
        if len(os.listdir(temp_dir)) == 0:
            os.rmdir(temp_dir)
    except OSError:
        logging.warning(f"Could not remove temporary directory {temp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process large CSV files in parallel segments")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--num_segments", type=int, default=100, help="Number of segments to split the data into")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    parallel_process(
        args.input_csv,
        args.data_root,
        args.output_csv,
        args.num_segments,
        args.max_workers
    )
