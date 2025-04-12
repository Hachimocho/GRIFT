#!/usr/bin/env python
"""
Debugging script to directly examine the quality CSV and its data structure
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json

def main():
    # Path to quality CSV
    quality_csv = "/major/datasets/ai-face/train_quality.csv"
    
    print(f"Examining quality CSV: {quality_csv}")
    
    # Load the CSV
    try:
        df = pd.read_csv(quality_csv)
        print(f"Successfully loaded CSV. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for image ID / path columns
        id_cols = [col for col in df.columns if any(term in col.lower() for term in ['id', 'path', 'file', 'image'])]
        print(f"Identified ID/path columns: {id_cols}")
        
        if id_cols:
            # Analyze the first column
            col = id_cols[0]
            print(f"\nAnalyzing column: {col}")
            
            # Get sample values
            sample = df[col].head(5).tolist()
            print(f"Sample values: {sample}")
            
            # Check if they're absolute paths
            has_abs_path = any(val.startswith('/') for val in sample if isinstance(val, str))
            print(f"Contains absolute paths: {has_abs_path}")
            
            # Check for embeddings
            embedding_col = next((col for col in df.columns if 'embedding' in col.lower()), None)
            if embedding_col:
                print(f"\nFound embedding column: {embedding_col}")
                
                # Get sample embedding
                sample_emb = df[embedding_col].iloc[0]
                print(f"Sample embedding type: {type(sample_emb)}")
                print(f"Sample embedding (truncated): {str(sample_emb)[:200]}...")
                
                # Try to parse embedding
                if isinstance(sample_emb, str):
                    try:
                        # Clean up embedding string
                        clean_str = sample_emb.replace('\n', ' ')
                        while '  ' in clean_str:
                            clean_str = clean_str.replace('  ', ' ')
                        clean_str = clean_str.strip().strip('[]')
                        parts = [p for p in clean_str.split(' ') if p]
                        values = [float(x) for x in parts]
                        
                        # Convert to numpy array
                        embedding = np.array(values)
                        print(f"Successfully parsed embedding: shape={embedding.shape}")
                        print(f"Min: {np.min(embedding):.4f}, Max: {np.max(embedding):.4f}")
                        print(f"Mean: {np.mean(embedding):.4f}, Std: {np.std(embedding):.4f}")
                        print(f"Number of zeros: {np.sum(np.isclose(embedding, 0))}")
                        
                        # Create a dummy node with the embedding
                        test_node = {'attributes': {'face_embedding': embedding}}
                        
                        # Try to pickle it
                        with open('test_embedding.pkl', 'wb') as f:
                            pickle.dump(test_node, f)
                        print("Successfully pickled test node with embedding")
                        
                        # Load it back
                        with open('test_embedding.pkl', 'rb') as f:
                            loaded_node = pickle.load(f)
                        
                        loaded_emb = loaded_node['attributes']['face_embedding']
                        print(f"Successfully loaded embedding: shape={loaded_emb.shape}")
                        print(f"Min: {np.min(loaded_emb):.4f}, Max: {np.max(loaded_emb):.4f}")
                        print(f"Values match: {np.array_equal(embedding, loaded_emb)}")
                        
                    except Exception as e:
                        print(f"Error parsing embedding: {e}")
                
        # Check unique vs basename collision
        if id_cols:
            col = id_cols[0]
            full_paths = df[col].tolist()
            basenames = [os.path.basename(p) if isinstance(p, str) else p for p in full_paths]
            
            unique_full = len(set(full_paths))
            unique_base = len(set(basenames))
            
            print(f"\nPath uniqueness analysis:")
            print(f"Unique full paths: {unique_full}")
            print(f"Unique basenames: {unique_base}")
            print(f"Collision rate: {100 - (unique_base / unique_full * 100):.2f}%")
                
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
if __name__ == "__main__":
    main()
