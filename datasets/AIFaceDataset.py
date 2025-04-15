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
                # 1. Strip all newline characters
                clean_str = embedding_str.replace('\n', ' ')
                
                # 2. Replace all multiple spaces with single spaces
                while '  ' in clean_str:
                    clean_str = clean_str.replace('  ', ' ')
                
                # Trim any surrounding whitespace or brackets
                clean_str = clean_str.strip().strip('[]')
                
                # 3. Split by spaces and filter out empty strings
                parts = [p for p in clean_str.split(' ') if p]
                
                # 4. Convert to floats
                values = [float(x) for x in parts]
                #print(values)
                
                # Only store if we got a non-empty array
                if values:
                    #print("Got values!")
                    embedding = np.array(values)
                    # We want ALL embeddings, as long as they're not all zeros
                    if not np.all(np.isclose(embedding, 0)):
                        attrs['face_embedding'] = embedding
                        # Add debug info about embeddings of different lengths
                        if len(embedding) != 512:
                            print(f"WARNING: Found embedding with non-standard length {len(embedding)} for {filename}")
                        # No need for detailed logging to avoid errors
            except Exception as e:
                print(f"Error parsing face embedding for {filename}: {str(e)}")
        
        return filename, attrs
        
    def _load_quality_attributes_parallel(self, df, filename_col):
        """Process quality attributes in parallel for faster loading"""
        # Prepare data for parallel processing
        rows_to_process = []
        for i, row in df.iterrows():
            # Use image_path if available, fall back to specified column otherwise
            if 'image_path' in row and not pd.isna(row['image_path']):
                filename = row['image_path']
            else:
                filename = row[filename_col]
                
            # Don't modify the path - use it exactly as stored in the CSV
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
        
        # DEBUG: Add counts of embeddings in the final attributes dictionary
        embedding_count = 0
        non_zero_count = 0
        sample_attrs = []
        for filename, attrs in attributes_map.items():
            if 'face_embedding' in attrs and isinstance(attrs['face_embedding'], np.ndarray):
                embedding_count += 1
                if not np.all(np.isclose(attrs['face_embedding'], 0)):
                    non_zero_count += 1
                    if len(sample_attrs) < 2:
                        sample_attrs.append((filename, attrs['face_embedding'].shape))
        
        print(f"DEBUG: Found {embedding_count} embeddings in final attributes_map dictionary")
        print(f"DEBUG: Of these, {non_zero_count} are non-zero embeddings")
        if sample_attrs:
            print(f"DEBUG: Sample embeddings: {sample_attrs}")
        
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
                
                # Update base attributes with root directory
                for filename in list(attributes_map.keys()):
                    full_path = os.path.join(self.data_root, filename.lstrip('/'))
                    attributes_map[full_path] = attributes_map.pop(filename)

                print(f"Updated base attributes with root directory for {subset}")
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
                
                # Process quality attributes
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing quality attributes for {subset}"):
                    raw_filename = row[filename_col]
                    
                    # Skip invalid filenames
                    if pd.isna(raw_filename) or not isinstance(raw_filename, str):
                        continue
                    
                    # Use the full path directly - no normalization
                    filename = raw_filename.strip()
                    
                    # Parse quality attributes
                    _, quality_attrs = self._parse_quality_attributes((filename, row))
                    
                    # Merge with existing attributes
                    if filename in attributes_map:
                        attributes_map[filename].update(quality_attrs)
                    else:
                        attributes_map[filename] = quality_attrs
                
                print(f"Loaded {len(attributes_map)} total attributes for {subset}")
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

    def create_nodes_threaded(self, chunk_args):
        """Process nodes using threading instead of multiprocessing
        This eliminates the expensive serialization/deserialization overhead
        """
        # Unpack the arguments
        subset, paths, targets, genders, races, ages, additional_attrs = chunk_args
        
        # Pre-calculate total nodes for this chunk
        total_nodes = len(paths)
        if not additional_attrs:
            print("DEBUG: additional_attrs is empty or None in create_nodes_threaded")
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
                    orig_path = base_paths[i] if 'base_paths' in locals() else path
                    
                    # CRITICAL: Find the original CSV path which is needed for attribute lookup
                    # this is the path as it appears in the CSV file - without the data_root prepended
                    csv_path = None
                    # We can't access chunk directly here, but we can reconstruct the path
                    if hasattr(self, 'data_root') and path.startswith(self.data_root):
                        # Try to extract the relative path by removing the data_root
                        rel_path = path[len(self.data_root):].lstrip('/')
                        if rel_path:
                            csv_path = '/' + rel_path  # Add leading slash as in original CSV
                    
                    # Debug for the first few entries
                    if i < 3 and csv_path:
                        print(f"DEBUG Path recovery - path: {path}, filename: {filename}, csv_path: {csv_path}")
                    
                    # Try multiple matching strategies including the newly computed csv_path
                    match_found = False
                    file_attrs = None
                    matched_key = None
                    
                    # Try direct lookup in additional_attrs using the CSV path which should match
                    # the key format in the quality attributes CSV
                    if additional_attrs and csv_path and csv_path in additional_attrs:
                        file_attrs = additional_attrs[csv_path]
                        match_found = True
                        matched_key = csv_path
                    else:
                        # Fall back to trying all path variations
                        candidates = [
                            orig_path,                    # Original path from CSV
                            orig_path.lstrip('/'),       # Without leading slash
                            '/' + orig_path.lstrip('/'), # With leading slash
                            os.path.basename(orig_path), # Just the filename
                            path,                        # Full path after joining
                            filename,                    # Just basename of full path
                        ]
                        
                        # Add the csv_path if we have it
                        if csv_path:
                            candidates.append(csv_path)
                        
                        # Add forward slash variants
                        slash_candidates = [c.replace('\\', '/') for c in candidates]
                        candidates.extend(slash_candidates)
                        
                        # Try each candidate path
                        for candidate in candidates:
                            if additional_attrs and candidate in additional_attrs:
                                file_attrs = additional_attrs[candidate]
                                match_found = True
                                matched_key = candidate
                                break
                    
                    # Debug all attribute keys for the first few nodes, to help with troubleshooting
                    if i < 3 and additional_attrs:
                        print(f"DEBUG Node {i}: Attribute lookup - match_found: {match_found}, matched_key: {matched_key}")
                        if match_found:
                            print(f"DEBUG Node {i}: Found attributes: {list(file_attrs.keys())[:5]}...")
                        else:
                            sample_keys = list(additional_attrs.keys())[:5]
                            print(f"DEBUG Node {i}: No match found. Sample attr keys: {sample_keys}")
                            print(f"DEBUG Node {i}: Candidates tried: {candidates[:5]}...")
                    
                    # Apply attributes if a match was found
                    if match_found and file_attrs:
                        # Debug first few nodes
                        if i < 5:
                            print(f"DEBUG Node {i}: Adding {len(file_attrs)} attributes from key '{matched_key}'")
                            quality_keys = [k for k in file_attrs.keys() if k.startswith('symmetry_') or k in ('blur', 'brightness', 'contrast', 'compression')]
                            print(f"DEBUG Node {i}: Quality attributes being added: {quality_keys}")
                            
                        # CRITICAL: Make sure to update the attributes dict
                        for k, v in file_attrs.items():
                            attributes[k] = v
                    
                    # Create data and node
                    data = self.data_class(path, **self.data_args)
                    label = int(targets[i]) if targets[i] is not None else 0
                    threshold = self.node_args.get('threshold', 80)
                    
                    # Create the node with all required arguments
                    node = self.node_class(path, subset, data, [], label, attributes, threshold)
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
            node = node_class(path, subset, data, [], label, attributes, threshold)
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
        # Try both full path and basename since the CSV may use either format
        full_path = path
        filename = os.path.basename(path)
        basename_time = time.time() - lookup_start
        
        # Time the additional attributes lookup
        attrs_lookup_start = time.time()
        # Try attribute lookup with multiple path formats to maximize matching
        file_attrs = None
        
        # First check if additional_attrs even has embeddings
        if additional_attrs:
            embedding_count = sum(1 for attrs in additional_attrs.values() 
                                if 'face_embedding' in attrs and isinstance(attrs['face_embedding'], np.ndarray))
            print(f"DEBUG in create_node: additional_attrs has {embedding_count} embeddings out of {len(additional_attrs)} entries")
            # Print a few sample keys
            sample_keys = list(additional_attrs.keys())[:3]
            print(f"DEBUG: Sample keys in additional_attrs: {sample_keys}")
        
        # Try several formats to maximize our chance of finding attributes
        lookup_paths = [
            path,                           # Full absolute path as is
            full_path,                      # Path extracted earlier
            filename,                       # Basename only
            path.replace('\\', '/'),        # Handle any Windows paths that might be mixed in
        ]
        
        # Try each path format until we find a match
        for lookup_path in lookup_paths:
            if additional_attrs and lookup_path in additional_attrs:
                file_attrs = additional_attrs[lookup_path]
                # For the first few matches, print debug info
                if not hasattr(self, '_debug_counter'):
                    self._debug_counter = 0
                if self._debug_counter < 5:
                    print(f"DEBUG #{self._debug_counter}: Found attributes for {os.path.basename(lookup_path)}")
                    self._debug_counter += 1
                break
        
        # If we still didn't find attributes, log a debug message
        if file_attrs is None and not hasattr(self, '_debug_miss_counter'):
            self._debug_miss_counter = 0
            if self._debug_miss_counter < 5:
                print(f"DEBUG MISS #{self._debug_miss_counter}: No attributes found for {os.path.basename(path)}")
                self._debug_miss_counter += 1
        
        # If we found attributes in either format, add them to the node
        if file_attrs:
            # Always show debug for first few nodes with embeddings
            if not hasattr(self, '_debug_attr_counter'):
                self._debug_attr_counter = 0
            
            has_embedding = 'face_embedding' in file_attrs
            
            if has_embedding and self._debug_attr_counter < 5:
                emb = file_attrs.get('face_embedding')
                emb_type = type(emb).__name__
                emb_shape = getattr(emb, 'shape', None)
                emb_sum = np.sum(emb) if isinstance(emb, np.ndarray) else 'N/A'
                print(f"DEBUG #{self._debug_attr_counter}: FOUND file_attrs with embedding: type={emb_type}, shape={emb_shape}, sum={emb_sum}")
                self._debug_attr_counter += 1
            
            # Use dictionary update for bulk attribute addition
            attributes.update(file_attrs)
            
            # Direct attribute access for face embedding if available
            if 'face_embedding' in file_attrs:
                emb = file_attrs['face_embedding']
                if isinstance(emb, np.ndarray) and len(emb) > 0:
                    # Copy to ensure no reference issues
                    attributes['face_embedding'] = np.array(emb)
                    
                    # CRITICAL: Verify the embedding was actually added
                    if 'face_embedding' in attributes:
                        added_emb = attributes['face_embedding']
                        print(f"SUCCESS: Added embedding of shape {added_emb.shape} to node attributes")
                    else:
                        print(f"ERROR: Failed to add embedding to node attributes!")
                    
                    # Count nodes with face_embedding
                    if not hasattr(self, '_face_embedding_count'):
                        self._face_embedding_count = 0
                    self._face_embedding_count += 1
                    
                    # Print summary more frequently for debugging
                    if self._face_embedding_count % 5 == 0:
                        print(f"Added face_embedding to {self._face_embedding_count} nodes so far")
                else:
                    print(f"WARNING: Found 'face_embedding' key but value is not a valid numpy array: {type(emb)}")
        else:
            # Limited debug output
            if not hasattr(self, '_debug_attr_counter'):
                self._debug_attr_counter = 0
            if self._debug_attr_counter < 5:
                print(f"DEBUG #{self._debug_attr_counter}: No attributes found for this file")
                self._debug_attr_counter += 1
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
        node = node_class(path, subset, data, [], label, attributes, threshold)
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
                # Print first few entries of attrs to verify they have quality attributes
                if not hasattr(self, '_debug_attrs_printed') or self._debug_attrs_printed < 3:
                    if not hasattr(self, '_debug_attrs_printed'):
                        self._debug_attrs_printed = 0
                    sample_keys = list(attrs.keys())[:5] if attrs else []
                    if sample_keys:
                        sample_attr = attrs[sample_keys[0]]
                        quality_keys = [k for k in sample_attr.keys() if k.startswith('symmetry_') or k in ('blur', 'brightness', 'contrast', 'compression')]
                        print(f"\nDEBUG: First few attribute keys: {sample_keys}")
                        print(f"DEBUG: Quality attribute keys in first item: {quality_keys}")
                        print(f"DEBUG: Total attributes loaded: {len(attrs)}")
                        
                        # Count how many entries have quality attributes
                        quality_count = 0
                        for _, attr_dict in list(attrs.items())[:100]:  # Check a subset for speed
                            if any(k.startswith('symmetry_') or k in ('blur', 'brightness', 'contrast', 'compression') for k in attr_dict):
                                quality_count += 1
                        print(f"DEBUG: {quality_count}/100 sample entries have quality attributes")
                    self._debug_attrs_printed += 1
                    
                chunk_args = (subset, paths, targets, genders, races, ages, attrs)
                
                # Process this chunk with threading
                nodes = self.create_nodes_threaded(chunk_args)
                
                # Debug check: Inspect the first few nodes to verify attribute inclusion
                if len(nodes) > 0 and (not hasattr(self, '_debug_node_printed') or self._debug_node_printed < 3):
                    if not hasattr(self, '_debug_node_printed'):
                        self._debug_node_printed = 0
                    node = nodes[0]
                    print(f"\nDEBUG: Node attributes: {sorted(node.attributes.keys())}")
                    quality_attrs = [k for k in node.attributes.keys() if k.startswith('symmetry_') or k in ('blur', 'brightness', 'contrast', 'compression')]
                    print(f"DEBUG: Node quality attributes: {quality_attrs}")
                    self._debug_node_printed += 1
                
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