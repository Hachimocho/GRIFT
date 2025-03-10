import os
from datasets.Dataset import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import gc
import ast  # For safer and faster dictionary parsing
import json
import re

class AIFaceDataset(Dataset):
    tags = ["deepfakes", "image", "attributes"]
    hyperparameters = None

    def _parse_quality_attributes(self, row_data):
        """Process a single row of quality attribute data"""
        filename, row = row_data
        attrs = {}
        
        # Handle quality metrics
        quality_str = row.get('quality_metrics')
        if isinstance(quality_str, str):
            try:
                # Use ast.literal_eval instead of eval for safety and speed
                quality_dict = ast.literal_eval(quality_str)
                attrs.update({
                    'blur': quality_dict.get('blur_score', 0),
                    'brightness': quality_dict.get('brightness', 0),
                    'contrast': quality_dict.get('contrast', 0),
                    'compression': quality_dict.get('compression_score', 0)
                })
            except:
                pass
        
        # Handle symmetry metrics
        symmetry_str = row.get('symmetry')
        if isinstance(symmetry_str, str):
            try:
                symmetry_dict = ast.literal_eval(symmetry_str)
                attrs.update({
                    'symmetry_eye': symmetry_dict.get('eye_ratio', 0),
                    'symmetry_mouth': symmetry_dict.get('mouth_ratio', 0),
                    'symmetry_nose': symmetry_dict.get('nose_ratio', 0),
                    'symmetry_overall': symmetry_dict.get('overall_symmetry', 0)
                })
            except:
                pass
        
        # Handle emotion scores
        emotion_str = row.get('emotion_scores')
        if isinstance(emotion_str, str):
            try:
                emotion_dict = ast.literal_eval(emotion_str)
                for emotion, score in emotion_dict.items():
                    attrs[f'emotion_{emotion}'] = float(score)
            except:
                pass
        
        # Handle face embedding
        embedding_str = row.get('face_embedding')
        if isinstance(embedding_str, str):
            try:
                # Faster embedding parsing using numpy directly
                embedding_str = embedding_str.strip('[]')
                # Use a regex to match all floating point numbers
                values = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', embedding_str)]
                if values:
                    attrs['face_embedding'] = np.array(values)
            except:
                pass
        
        return filename, attrs
        
    def _load_quality_attributes_parallel(self, df, filename_col):
        """Process quality attributes in parallel for faster loading"""
        # Prepare data for parallel processing
        rows_to_process = []
        for i, row in df.iterrows():
            # Use full path without any special treatment
            filename = row[filename_col]
            if pd.isna(filename) or not isinstance(filename, str):
                continue
            rows_to_process.append((filename, row))
        
        # Process in parallel using all available cores except one
        workers = max(1, cpu_count() - 1)
        print(f"Processing quality attributes with {workers} parallel workers...")
        
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(self._parse_quality_attributes, rows_to_process, chunksize=1000),
                total=len(rows_to_process),
                desc="Processing quality attributes"
            ))
        
        # Collect results
        attributes_map = {}
        quality_count = 0
        symmetry_count = 0
        embedding_count = 0
        
        for filename, attrs in results:
            if attrs:
                if any(k in attrs for k in ['blur', 'brightness', 'contrast', 'compression']):
                    quality_count += 1
                if any(k.startswith('symmetry_') for k in attrs):
                    symmetry_count += 1
                if 'face_embedding' in attrs:
                    embedding_count += 1
                attributes_map[filename] = attrs
        
        print(f"Processed attributes: {quality_count} quality metrics, {symmetry_count} symmetry metrics, {embedding_count} face embeddings")
        return attributes_map

    def _normalize_filename(self, filename):
        """Normalize filename to ensure consistent matching between different CSVs"""
        # Skip normalization for invalid filenames
        if pd.isna(filename) or not isinstance(filename, str):
            return None
            
        # Instead of extracting basename, preserve the full path
        # Just do minimal normalization to handle edge cases
        normalized = filename.strip()
        
        # Handle duplicate slashes and other path normalization if needed
        # normalized = os.path.normpath(normalized)
        
        return normalized

    def _analyze_csv_filename_stats(self, csv_path):
        """Analyze statistics about filenames in a CSV file"""
        if not os.path.exists(csv_path):
            print(f"CSV file does not exist: {csv_path}")
            return
            
        try:
            # Read only the first few rows to detect the column names
            df_sample = pd.read_csv(csv_path, nrows=5)
            
            # Determine filename column
            filename_col = None
            for possible_col in ['image_id', 'image_path', 'Image Path', 'filename']:
                if possible_col in df_sample.columns:
                    filename_col = possible_col
                    break
            
            if filename_col is None:
                print(f"Warning: Could not find filename column in {csv_path}. Available columns: {df_sample.columns.tolist()}")
                # Try to use the first column as a fallback
                filename_col = df_sample.columns[0]
                print(f"Using '{filename_col}' as the filename column")
            
            # Now read just the filename column for the entire file
            df = pd.read_csv(csv_path, usecols=[filename_col])
            
            # Count total rows
            total_rows = len(df)
            
            # Count NaN values
            nan_count = df[filename_col].isna().sum()
            
            # Count non-string values (shouldn't be many, but could cause issues)
            non_string_count = sum(1 for item in df[filename_col] if not pd.isna(item) and not isinstance(item, str))
            
            # Use filenames directly without normalization
            valid_filenames = [f for f in df[filename_col] if not pd.isna(f) and isinstance(f, str)]
            unique_filenames = set(valid_filenames)
            unique_count = len(unique_filenames)
            
            # Check for duplicates
            dup_count = len(valid_filenames) - unique_count
            
            print(f"\nFilename Statistics for {os.path.basename(csv_path)}:")
            print(f"  Total rows: {total_rows}")
            print(f"  Valid filenames: {len(valid_filenames)} ({len(valid_filenames)/total_rows*100:.2f}%)")
            print(f"  NaN filenames: {nan_count} ({nan_count/total_rows*100:.2f}%)")
            print(f"  Non-string filenames: {non_string_count} ({non_string_count/total_rows*100:.2f}%)")
            print(f"  Unique filenames: {unique_count} ({unique_count/len(valid_filenames)*100:.2f}% of valid)")
            print(f"  Duplicate filenames: {dup_count} ({dup_count/len(valid_filenames)*100:.2f}% of valid)")
            
            # Sample of duplicates if any exist
            if dup_count > 0:
                # Get counts of each filename
                from collections import Counter
                filename_counts = Counter(valid_filenames)
                duplicates = {name: count for name, count in filename_counts.items() if count > 1}
                
                # Show top 5 duplicates
                print("\n  Top duplicated filenames:")
                for name, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    '{name}': {count} occurrences")
            
            return unique_filenames
                
        except Exception as e:
            print(f"Error analyzing CSV: {str(e)}")
            return set()

    def _load_additional_attributes(self, subset):
        """Load additional attributes from CSV file for the given split"""
        print(f"Loading additional attributes for {subset}...")
        # Load both base and quality attributes
        base_csv = os.path.join(self.data_root, f"{subset}.csv")
        quality_csv = os.path.join(self.data_root, f"{subset}_quality.csv")
        
        attributes_map = {}
        
        # Perform analysis on the CSV files to understand filename patterns
        print(f"Analyzing base CSV file: {base_csv}")
        base_unique_filenames = self._analyze_csv_filename_stats(base_csv)
        
        if os.path.exists(quality_csv):
            print(f"Analyzing quality CSV file: {quality_csv}")
            quality_unique_filenames = self._analyze_csv_filename_stats(quality_csv)
            
            # Calculate potential overlap
            if base_unique_filenames and quality_unique_filenames:
                overlap = base_unique_filenames.intersection(quality_unique_filenames)
                print(f"\nPotential filename overlap between base and quality:")
                print(f"  Base CSV unique filenames: {len(base_unique_filenames)}")
                print(f"  Quality CSV unique filenames: {len(quality_unique_filenames)}")
                print(f"  Overlap: {len(overlap)} files ({len(overlap)/len(base_unique_filenames)*100:.2f}% of base)")
                print(f"  Base only: {len(base_unique_filenames - quality_unique_filenames)} files")
                print(f"  Quality only: {len(quality_unique_filenames - base_unique_filenames)} files")
        
        # Load base attributes
        if os.path.exists(base_csv):
            try:
                print(f"Loading base attributes for {subset} from {base_csv}...")
                df = pd.read_csv(base_csv)
                
                # Determine filename column - could be 'image_id', 'image_path', 'Image Path', etc.
                filename_col = None
                for possible_col in ['image_id', 'image_path', 'Image Path', 'filename']:
                    if possible_col in df.columns:
                        filename_col = possible_col
                        break
                
                if filename_col is None:
                    print(f"Warning: Could not find filename column in {base_csv}. Available columns: {df.columns.tolist()}")
                    # Try to use the first column as a fallback
                    filename_col = df.columns[0]
                    print(f"Using '{filename_col}' as the filename column")
                
                # Track valid vs invalid entries 
                valid_count = 0
                invalid_count = 0
                
                # Process attributes
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing base attributes for {subset}"):
                    raw_filename = row[filename_col]
                    
                    # Skip invalid filenames
                    if pd.isna(raw_filename) or not isinstance(raw_filename, str):
                        invalid_count += 1
                        continue
                    
                    # Use the full path directly - no normalization
                    filename = raw_filename.strip()
                    
                    # Convert row to dict, excluding special columns and handling NaN values
                    attrs = {}
                    for k, v in row.items():
                        if k not in [filename_col, '_debug'] and not k.startswith('Unnamed:'):
                            # Check if value is not NaN/None
                            if isinstance(v, (str, bool, int, float)) and not pd.isna(v):
                                attrs[k] = v
                    
                    if attrs:  # Only add if we have valid attributes
                        attributes_map[filename] = attrs
                        valid_count += 1
                
                print(f"Loaded {len(attributes_map)} base attributes for {subset}")
                print(f"Valid entries: {valid_count}, Invalid entries: {invalid_count}")
            except Exception as e:
                print(f"Warning: Error loading base attributes for {subset}: {str(e)}")
                # Print available columns for debugging
                try:
                    columns = pd.read_csv(base_csv, nrows=0).columns.tolist()
                    print(f"Available columns in {base_csv}: {columns}")
                except:
                    pass
        
        # Load quality attributes
        if os.path.exists(quality_csv):
            try:
                print(f"Loading quality attributes for {subset} from {quality_csv}...")
                
                # Read the CSV file once
                df = pd.read_csv(quality_csv)
                
                # Determine filename column for quality CSV
                filename_col = None
                for possible_col in ['image_id', 'image_path', 'Image Path', 'filename']:
                    if possible_col in df.columns:
                        filename_col = possible_col
                        break
                
                if filename_col is None:
                    print(f"Warning: Could not find filename column in {quality_csv}. Available columns: {df.columns.tolist()}")
                    # Try to use the first column as a fallback
                    filename_col = df.columns[0]
                    print(f"Using '{filename_col}' as the filename column")
                
                # Use parallel processing for quality attributes
                quality_attributes = self._load_quality_attributes_parallel(df, filename_col)
                
                # Check how many quality attributes can actually be merged
                mergeable_count = 0
                standalone_count = 0
                for filename in quality_attributes.keys():
                    # Don't normalize, just use the filename as is
                    if filename in attributes_map:
                        mergeable_count += 1
                    else:
                        standalone_count += 1
                
                print(f"Found {mergeable_count} quality attributes that can be merged with base attributes")
                print(f"Found {standalone_count} quality attributes without matching base attributes")
                
                # Merge with existing attributes
                merge_count = 0
                standalone_count = 0
                for filename, attrs in quality_attributes.items():
                    # Don't normalize, use the filename as is                   
                    if filename in attributes_map:
                        attributes_map[filename].update(attrs)
                        merge_count += 1
                    else:
                        # Add quality attributes even if there's no matching base entry
                        attributes_map[filename] = attrs
                        standalone_count += 1
                
                print(f"Quality attribute merging: {merge_count} merged, {standalone_count} standalone")
                print(f"Added quality attributes for {subset}: {len(quality_attributes)} files total")
            except Exception as e:
                print(f"Warning: Error loading quality attributes for {subset}: {str(e)}")
                # Print available columns for debugging
                try:
                    columns = pd.read_csv(quality_csv, nrows=0).columns.tolist()
                    print(f"Available columns in {quality_csv}: {columns}")
                except:
                    pass
        
        print(f"Total attributes loaded for {subset}: {len(attributes_map)} files")
        return attributes_map

    def create_node(self, args):
        subset, path, target, gender, race, age, data_class, data_args, node_class, node_args, additional_attrs = args
        # Start with base attributes using correct prefixes
        attributes = {
            "gender_" + str(gender).lower(): True,
            "race_" + str(race).lower(): True,
            "age_" + str(age).lower(): True
        }
        
        # Add additional attributes if available
        if isinstance(additional_attrs, dict) and len(additional_attrs) > 0:
            filename = os.path.basename(path)
            if filename in additional_attrs:
                # Add emotion attributes with prefix
                for key, value in additional_attrs[filename].items():
                    if key.startswith('emotion_'):
                        attributes[key] = value
                    elif key in ['blur', 'brightness', 'contrast', 'compression']:
                        # Quality metrics should be added as is
                        attributes[key] = value
                    elif key.startswith('symmetry_'):
                        # Symmetry metrics should be added as is
                        attributes[key] = value
                    elif key == 'face_embedding' and isinstance(value, np.ndarray):
                        # Face embedding should be added as is
                        attributes[key] = value
                    else:
                        # For any other attributes, add them with appropriate prefix
                        if isinstance(value, bool):
                            prefix = next((p for p in ['gender_', 'race_', 'age_', 'emotion_'] 
                                        if key.startswith(p)), '')
                            if prefix:
                                attributes[key] = value
        
        node = node_class(
            subset,
            data_class(path, **data_args),
            [],
            int(target),
            attributes,
            **node_args
        )
        # Explicitly set the split attribute
        node.split = subset
        return node

    def load(self):
        print("Load called.")
        data_root = self.data_root
        train_csv = os.path.join(data_root, "train.csv")
        val_csv = os.path.join(data_root, "val.csv")
        test_csv = os.path.join(data_root, "test.csv")
        
        # Check if we need to create splits
        if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
            try:
                # Load all data
                all_data = pd.read_csv(os.path.join(data_root, "data.csv"))
                
                # Create splits
                print("Creating train/val/test splits...")
                train_ratio, val_ratio = 0.7, 0.15  # This leaves 0.15 for test
                
                # Randomly shuffle the data
                all_data = all_data.sample(frac=1, random_state=42)
                
                # Calculate split indices
                n = len(all_data)
                train_end = int(train_ratio * n)
                val_end = int((train_ratio + val_ratio) * n)
                
                # Split the data
                train_data = all_data[:train_end]
                val_data = all_data[train_end:val_end]
                test_data = all_data[val_end:]
                
                # Save splits
                train_data.to_csv(train_csv, index=False)
                val_data.to_csv(val_csv, index=False)
                test_data.to_csv(test_csv, index=False)
                
                print(f"Created splits:")
                print(f"Train: {len(train_data)} samples")
                print(f"Val: {len(val_data)} samples")
                print(f"Test: {len(test_data)} samples")
                
                del all_data, train_data, val_data, test_data
                gc.collect()
            except Exception as e:
                raise Exception(f"Error creating splits: {str(e)}")
        
        # Store nodes for each subset separately
        all_nodes = []
        num_processes = min(4, cpu_count() - 1)
        
        # Load additional attributes for each split
        print("Loading additional attributes...")
        train_attrs = self._load_additional_attributes("train")
        val_attrs = self._load_additional_attributes("val")
        test_attrs = self._load_additional_attributes("test")
        
        # Process train, val, test in order
        for subset, csv, attrs in tqdm(list(zip(
            ["train", "val", "test"], 
            [train_csv, val_csv, test_csv],
            [train_attrs, val_attrs, test_attrs]
        )), desc="Loading datasets"):
            print(f"\nLoading {subset} data for AIFace dataset...")
            
            if not os.path.exists(csv):
                print(f"Warning: {csv} not found, skipping {subset} split")
                continue
                
            try:
                df = pd.read_csv(csv)
                if len(df) == 0:
                    print(f"Warning: {csv} is empty, skipping {subset} split")
                    continue
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find input file: {csv}")
            except pd.errors.EmptyDataError:
                raise ValueError("The input file is empty")
                
            chunk_size = 100000
            subset_nodes = []
            
            for chunk in pd.read_csv(csv, chunksize=chunk_size):
                chunk['image_path'] = chunk['Image Path'].apply(lambda x: os.path.join(data_root, x.lstrip('/')))
                args_list = [
                    (subset, path, target, gender, race, age, self.data_class, self.data_args, self.node_class, self.node_args, attrs)
                    for path, target, gender, race, age in zip(
                        chunk['image_path'],
                        chunk['Target'],
                        chunk['Ground Truth Gender'],
                        chunk['Ground Truth Race'],
                        chunk['Ground Truth Age']
                    )
                ]
                
                with Pool(num_processes) as pool:
                    chunk_nodes = list(tqdm(
                        pool.imap_unordered(self.create_node, args_list, chunksize=1000),
                        total=len(args_list),
                        desc=f"Processing {subset} chunk"
                    ))
                    subset_nodes.extend(chunk_nodes)
                
                del chunk
                del args_list
                del chunk_nodes
                gc.collect()
            
            # Add all nodes from this subset to the final list
            all_nodes.extend(subset_nodes)
            del subset_nodes
            gc.collect()
            
        self.nodes = all_nodes
        return self.nodes