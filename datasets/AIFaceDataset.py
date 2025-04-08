import os
from datasets.Dataset import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import ast  # For safer and faster dictionary parsing
import json
import re
import random
import time
from profiler import profiler

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
            filename = row[filename_col]
            filename = filename.split('ai-face')[-1].strip()
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
            for possible_col in ['image_path', 'Image Path']:
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
            valid_filenames = [f.split('ai-face')[-1] for f in df[filename_col] if not pd.isna(f) and isinstance(f, str)]
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
            
            print("\n Example filenames:")
            print(random.sample(sorted(unique_filenames), 5))

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
                filename_col = 'image_path'
                # for possible_col in ['image_id', 'image_path', 'Image Path', 'filename']:
                #     if possible_col in df.columns:
                #         filename_col = possible_col
                #         break
                
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
                raise e
        
        # Load quality attributes
        if os.path.exists(quality_csv):
            try:
                print(f"Loading quality attributes for {subset} from {quality_csv}...")
                    
                # Read the CSV file once
                df = pd.read_csv(quality_csv)
                    
                # Determine filename column for quality CSV
                filename_col = 'image_path'
                # for possible_col in ['image_id', 'image_path', 'Image Path', 'filename']:
                #     if possible_col in df.columns:
                #         filename_col = possible_col
                #         break
                    
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
                raise e
            
        print(f"Total attributes loaded for {subset}: {len(attributes_map)} files")
        return attributes_map

    def create_nodes_threaded(self, chunk_args):
        """Process nodes using threading instead of multiprocessing
        This eliminates the expensive serialization/deserialization overhead
        """
        # Unpack the arguments
        subset, paths, targets, genders, races, ages, additional_attrs = chunk_args
        
        # Pre-calculate total nodes for this chunk
        total_nodes = len(paths)
        if total_nodes == 0:
            return []
            
        # Initialize result nodes list with the right size to avoid resizing
        nodes = [None] * total_nodes
        
        # Track timing for the first chunk only
        first_run = not hasattr(self, '_threaded_first_run')
        if first_run:
            self._threaded_first_run = True
            chunk_start = time.time()
        
        # Track problematic indices for debugging
        problematic_indices = []
        node_lock = threading.Lock()  # For thread-safety when updating shared data
        
        # Function to process a slice of nodes in each thread
        def process_slice(start_idx, end_idx, thread_id):
            # Process all nodes in this slice
            for i in range(start_idx, end_idx):
                # Skip already processed nodes (defensive programming)
                if nodes[i] is not None:
                    continue
                    
                try:
                    # Fast attribute creation
                    g_str = str(genders[i]).lower()
                    r_str = str(races[i]).lower()
                    a_str = str(ages[i]).lower()
                    
                    attributes = {
                        "gender_" + g_str: True,
                        "race_" + r_str: True,
                        "age_" + a_str: True
                    }
                    
                    # Extract filename and add additional attributes
                    path = paths[i]
                    filename = os.path.basename(path)
                    if additional_attrs and filename in additional_attrs:
                        file_attrs = additional_attrs[filename]
                        if file_attrs:
                            attributes.update(file_attrs)
                            if 'face_embedding' in file_attrs:
                                attributes['face_embedding'] = file_attrs['face_embedding']
                    
                    # Create data and node
                    data = self.data_class(path, **self.data_args)
                    label = int(targets[i]) if targets[i] is not None else 0
                    threshold = self.node_args.get('threshold', 80)
                    
                    # Create the node with all required arguments
                    node = self.node_class(subset, data, [], label, attributes, threshold)
                    node.attributes["Target"] = targets[i]
                    node.attributes["subset"] = subset
                    
                    # Explicitly set the split property needed by dataloader
                    node.split = subset
                    
                    # Store the node directly in the pre-allocated list
                    nodes[i] = node
                    
                except Exception as e:
                    # Log the exception and add to the problematic indices list
                    with node_lock:
                        problematic_indices.append((i, str(e)))  
                    # Don't re-raise - we want to continue processing other nodes
        
        # Determine optimal thread count - using more threads than CPUs helps with I/O bound tasks
        # but we want to avoid excessive thread creation overhead
        optimal_thread_count = min(32, max(cpu_count() * 2, 4))
        
        # Use a more reliable approach: assign indices to threads in a round-robin fashion
        # This avoids boundary issues with slicing
        all_indices = list(range(total_nodes))
        thread_indices = [[] for _ in range(optimal_thread_count)]
        
        # Distribute indices evenly across threads
        for i, idx in enumerate(all_indices):
            thread_indices[i % optimal_thread_count].append(idx)
        
        # Create and start threads
        threads = []
        for t in range(optimal_thread_count):
            # Skip empty assignments
            if not thread_indices[t]:
                continue
                
            thread = threading.Thread(
                target=lambda indices, tid: [process_slice(idx, idx+1, tid) for idx in indices],
                args=(thread_indices[t], t)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Report any problematic indices
        if problematic_indices:
            print(f"Found {len(problematic_indices)} problematic nodes in {subset} chunk")
            # Show the first few problematic indices and errors for diagnosis
            for idx, error in problematic_indices[:5]:
                print(f"  - Index {idx}: {error}")
            
        # For the first chunk, print some timing metrics
        if first_run:
            elapsed = time.time() - chunk_start
            throughput = total_nodes / elapsed if elapsed > 0 else 0
            print(f"Threaded node creation: {total_nodes} nodes in {elapsed:.4f}s ({throughput:.2f} nodes/sec)")
            for i in range(10):
                print(f"Node {i}: {nodes[i]}")

        return nodes
        
    def create_nodes_batch(self, batch_args):
        """Process a batch of nodes at once to reduce IPC overhead"""
        batch_start = time.time()
        batch_nodes = []
        
        for args in batch_args:
            subset, path, target, gender, race, age, data_class, data_args, node_class, node_args, additional_attrs = args
            
            # Fast string conversion and base attribute creation
            g_str = str(gender).lower()
            r_str = str(race).lower()
            a_str = str(age).lower()
            
            attributes = {
                "gender_" + g_str: True,
                "race_" + r_str: True,
                "age_" + a_str: True
            }
            
            # Extract filename and add additional attributes 
            filename = os.path.basename(path)
            if additional_attrs and filename in additional_attrs:
                file_attrs = additional_attrs[filename]
                if file_attrs:
                    attributes.update(file_attrs)
                    if 'face_embedding' in file_attrs:
                        attributes['face_embedding'] = file_attrs['face_embedding']
            
            # Create data and node
            data = data_class(path, **data_args)
            label = int(target) if target is not None else 0
            threshold = node_args.get('threshold', 80)
            
            # Create the node with all required arguments
            node = node_class(subset, data, [], label, attributes, threshold)
            node.attributes["Target"] = target
            node.attributes["subset"] = subset
            
            batch_nodes.append(node)
        
        if batch_nodes and not hasattr(self, '_batch_count'):
            self._batch_count = 0
            print(f"Batch processing: {len(batch_nodes)} nodes in {time.time() - batch_start:.4f}s")
            self._batch_count += 1
            
        return batch_nodes
    
    def create_node(self, args):
        # Start overall node creation timing
        node_start = time.time()
        
        subset, path, target, gender, race, age, data_class, data_args, node_class, node_args, additional_attrs = args
        
        # Time the attribute dictionary creation
        attr_start = time.time()
        g_str = str(gender).lower()
        r_str = str(race).lower()
        a_str = str(age).lower()
        
        # Use direct dictionary initialization (faster than individual assignments)
        attributes = {
            "gender_" + g_str: True,
            "race_" + r_str: True,
            "age_" + a_str: True
        }
        attr_time = time.time() - attr_start
        
        # Extract image filename for attribute lookup
        lookup_start = time.time()
        filename = os.path.basename(path)
        basename_time = time.time() - lookup_start
        
        # Time the additional attributes lookup
        attrs_lookup_start = time.time()
        # Add additional attributes if available (significantly more efficient lookup)
        if additional_attrs and filename in additional_attrs:
            # Get all the additional attributes for this file
            file_attrs = additional_attrs[filename]
            if file_attrs:
                # Use dictionary update for bulk attribute addition
                attributes.update(file_attrs)
                
                # Direct attribute access for face embedding if available
                if 'face_embedding' in file_attrs:
                    attributes['face_embedding'] = file_attrs['face_embedding']
        attrs_lookup_time = time.time() - attrs_lookup_start
        
        # Create the node - pass path as positional argument
        data_creation_start = time.time()
        data = data_class(path, **data_args)
        data_creation_time = time.time() - data_creation_start
        
        # Convert target to int for label
        label = int(target) if target is not None else 0
        
        # Create node with correct positional arguments: split, data, edges, label, attributes, threshold
        # Extract threshold from node_args or use default
        threshold = node_args.get('threshold', 80)  # Default threshold of 80%
        
        # Pass args in correct order as required by AttributeNode.__init__
        node_init_start = time.time()
        node = node_class(subset, data, [], label, attributes, threshold)
        node_init_time = time.time() - node_init_start
        
        # Store additional info in attributes dict
        attr_update_start = time.time()
        node.attributes["Target"] = target
        node.attributes["subset"] = subset
        attr_update_time = time.time() - attr_update_start
        
        # Calculate total node creation time
        total_time = time.time() - node_start
        
        # Store profiling data with the node for analysis
        node._profiling = {
            'attr_init_time': attr_time,
            'basename_time': basename_time,
            'attrs_lookup_time': attrs_lookup_time,
            'data_creation_time': data_creation_time,
            'node_init_time': node_init_time,
            'attr_update_time': attr_update_time,
            'total_time': total_time
        }
        
        # Print times for first 10 nodes to identify bottlenecks
        if not hasattr(self, '_profile_count'):
            self._profile_count = 0
        
        if self._profile_count < 10:
            print(f"\nNode #{self._profile_count} Creation Profile:")
            print(f"  Attribute init: {attr_time:.5f}s")
            print(f"  Path basename: {basename_time:.5f}s")
            print(f"  Attribute lookup: {attrs_lookup_time:.5f}s")
            print(f"  Data creation: {data_creation_time:.5f}s")
            print(f"  Node init: {node_init_time:.5f}s")
            print(f"  Attribute update: {attr_update_time:.5f}s")
            print(f"  Total time: {total_time:.5f}s")
            self._profile_count += 1
        
        return node

    def load(self):
        print("Load called.")
        # Start overall profiling
        profiler.start('total_load_time')
        
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
        num_processes = min(8, cpu_count())  # Use more CPU cores for processing
        print(f"Using {num_processes} processes for parallel node creation")
    
        # Load additional attributes for each split
        print("Loading additional attributes...")
        profiler.start('loading_attributes')
        
        profiler.start('loading_train_attributes')
        train_attrs = self._load_additional_attributes("train")
        profiler.end('loading_train_attributes')
        
        profiler.start('loading_val_attributes')
        val_attrs = self._load_additional_attributes("val")
        profiler.end('loading_val_attributes')
        
        profiler.start('loading_test_attributes')
        test_attrs = self._load_additional_attributes("test")
        profiler.end('loading_test_attributes')
        
        profiler.end('loading_attributes')
    
        # Process train, val, test in order
        profiler.start('processing_all_splits')
        for subset, csv, attrs in tqdm(list(zip(
            ["train", "val", "test"],
            [train_csv, val_csv, test_csv],
            [train_attrs, val_attrs, test_attrs]
        )), desc="Loading datasets"):
            profiler.start(f'processing_{subset}_split')
            print(f"\nLoading {subset} data for AIFace dataset...")
            
            if not os.path.exists(csv):
                print(f"Warning: {csv} not found, skipping {subset} split")
                profiler.end(f'processing_{subset}_split')
                continue
            
            try:
                # Just check if the file exists and has content
                df_sample = pd.read_csv(csv, nrows=5)
                if len(df_sample) == 0:
                    print(f"Warning: {csv} is empty, skipping {subset} split")
                    continue
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find input file: {csv}")
            except pd.errors.EmptyDataError:
                raise ValueError("The input file is empty")
            
            # Calculate the total number of rows first
            with open(csv, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            
            # Optimize chunk size based on available memory and cores
            chunk_size = min(100000, max(10000, total_rows // (num_processes * 3)))
            subset_nodes = []
            
            start_time = time.time()
            processed_count = 0
            
            # Process chunks of CSV data
            profiler.start(f'{subset}_chunk_processing')
            for chunk in pd.read_csv(csv, chunksize=chunk_size):
                profiler.start(f'{subset}_vectorized_operations')
                # Vectorized path generation (much faster than apply)
                base_paths = chunk['Image Path'].values
                chunk['image_path'] = np.array([os.path.join(data_root, p.lstrip('/')) for p in base_paths])
                
                # Vectorized extraction of column data (avoids row iteration)
                paths = chunk['image_path'].values
                targets = chunk['Target'].values
                genders = chunk['Ground Truth Gender'].values
                races = chunk['Ground Truth Race'].values
                ages = chunk['Ground Truth Age'].values
                
                profiler.end(f'{subset}_vectorized_operations')
                profiler.start(f'{subset}_args_list_creation')
            
                # THREADED PROCESSING: Much more efficient for this workload
                # No need to create individual argument tuples - just pass chunk data directly
                profiler.end(f'{subset}_args_list_creation')
                profiler.start(f'{subset}_threaded_node_creation')
                
                print(f"Processing {len(paths)} nodes using threaded approach")
                
                # Package chunk data for threaded processing
                chunk_args = (subset, paths, targets, genders, races, ages, attrs)
                
                # Process this chunk with threading
                nodes = self.create_nodes_threaded(chunk_args)
                
                # Validate nodes - check for None values
                if None in nodes:
                    none_count = nodes.count(None)
                    print(f"Warning: Found {none_count} None nodes in {subset} chunk")
                    nodes = [n for n in nodes if n is not None]
                
                # Make sure each node has the split attribute properly set
                for node in nodes:
                    node.split = subset  # Explicitly set the split attribute
                    
                subset_nodes.extend(nodes)
                
                profiler.end(f'{subset}_threaded_node_creation')
                
                # Track performance and provide feedback
                processed_count += len(chunk)
                elapsed = time.time() - start_time
                nodes_per_second = processed_count / elapsed if elapsed > 0 else 0
                estimated_total = total_rows / nodes_per_second if nodes_per_second > 0 else 0
                
                print(f"Performance: {nodes_per_second:.2f} nodes/sec, " 
                      f"Processed: {processed_count}/{total_rows}, "
                      f"Estimated total time: {estimated_total/60:.1f} min")
                
                # Clean up to avoid memory leaks
                del chunk
                del paths, targets, genders, races, ages
                gc.collect()
            
            # Add all nodes from this subset to the final list
            all_nodes.extend(subset_nodes)
            print(f"Completed {subset} split: {len(subset_nodes)} nodes created")
            
            # Clean up subset data
            profiler.start(f'{subset}_cleanup')
            del subset_nodes
            gc.collect()
            profiler.end(f'{subset}_cleanup')
            
            # End profiling for this split
            profiler.end(f'processing_{subset}_split')
        
        self.nodes = all_nodes
        
        # End all profiling timers
        profiler.end('processing_all_splits')
        profiler.end('total_load_time')
        
        # Print profiling report
        print(profiler)
        
        return self.nodes