import os
import sys
from datasets.AIFaceDataset import AIFaceDataset
from nodes.atrnode import AttributeNode
from graphs.HyperGraph import HyperGraph

def main():
    print("Testing AIFaceDataset attribute loading directly...")
    data_root = "/home/brg2890/major/datasets/ai-face"
    
    # Create dataset instance
    edge_class = None
    node_class = AttributeNode
    node_args = {}
    data_class = None
    data_args = {}
    
    # Initialize the dataset
    dataset = AIFaceDataset(data_root, data_class, data_args, node_class, node_args)
    
    # Load attributes
    print("Loading nodes from datasets...")
    nodes = dataset.load()
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total nodes loaded: {len(nodes)}")
    
    # Count attributes by type
    attr_counts = {}
    quality_metrics_count = 0
    symmetry_count = 0
    face_embedding_count = 0
    
    for node in nodes:
        for attr_name in node.attributes.keys():
            if attr_name not in attr_counts:
                attr_counts[attr_name] = 0
            attr_counts[attr_name] += 1
            
            # Count specific attribute types
            if attr_name in ['blur', 'brightness', 'contrast', 'compression']:
                quality_metrics_count += 1
            elif attr_name.startswith('symmetry_'):
                symmetry_count += 1
            elif attr_name == 'face_embedding':
                face_embedding_count += 1
    
    # Count nodes with quality attributes
    nodes_with_quality = sum(1 for node in nodes if any(attr in node.attributes for attr in ['blur', 'brightness', 'contrast', 'compression']))
    nodes_with_symmetry = sum(1 for node in nodes if any(attr.startswith('symmetry_') for attr in node.attributes))
    nodes_with_embedding = sum(1 for node in nodes if 'face_embedding' in node.attributes)
    
    print(f"\nAttribute Type Statistics:")
    print(f"Nodes with quality metrics: {nodes_with_quality} ({nodes_with_quality/len(nodes)*100:.2f}%)")
    print(f"Nodes with symmetry: {nodes_with_symmetry} ({nodes_with_symmetry/len(nodes)*100:.2f}%)")
    print(f"Nodes with face embeddings: {nodes_with_embedding} ({nodes_with_embedding/len(nodes)*100:.2f}%)")
    
    print(f"\nTotal attribute counts:")
    print(f"Quality metric attributes: {quality_metrics_count}")
    print(f"Symmetry attributes: {symmetry_count}")
    print(f"Face embedding attributes: {face_embedding_count}")
    
    print("\nAttribute Distribution:")
    for attr_name, count in sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)[:20]:  # Top 20 attributes
        print(f"  {attr_name}: {count} nodes ({count/len(nodes)*100:.2f}%)")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
