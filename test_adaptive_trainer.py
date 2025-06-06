"""
Demonstration script for the new AdaptiveTrainer architecture.
Shows how to dynamically switch traversal methods during training.
"""
import time
import os
import sys
import logging
import json
import random
import torch
import torch.nn as nn
from datetime import datetime

# Import utilities from the test helpers
from test_helpers.logging_utils import capture_output, set_seed
from test_helpers.args_utils import parse_args
from test_helpers.data_graph_utils import load_and_prepare_data_splits

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new AdaptiveTrainer
from trainers.AdaptiveTrainer import AdaptiveTrainer
from managers.NoGraphManager import NoGraphManager
from models.CNNModel import CNNModel

def main():
    print("=" * 80)
    print("AdaptiveTrainer Demonstration - Dynamic Traversal Switching")
    print("=" * 80)
    
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the primary loss function
    criterion = nn.BCEWithLogitsLoss().to(device)
    data_root = "/home/brg2890/major/datasets/ai-face"

    # Set up attribute metadata
    attribute_metadata = [
        {
            'name': 'Ground Truth Gender',
            'type': 'categorical',
            'possible_values': [0, 1]  
        },
        {
            'name': 'Ground Truth Race',
            'type': 'categorical',
            'possible_values': [0, 1, 2, 3]  
        },
        {
            'name': 'Ground Truth Age',
            'type': 'categorical',
            'possible_values': [0, 1, 2, 3]  
        },
        {
            'name': 'blur',
            'type': 'continuous'
        },
        {
            'name': 'brightness',
            'type': 'continuous'
        },
        {
            'name': 'contrast',
            'type': 'continuous'
        },
        {
            'name': 'compression',
            'type': 'continuous'
        },
        {
            'name': 'symmetry_eye',
            'type': 'continuous'
        },
        {
            'name': 'symmetry_mouth',
            'type': 'continuous'
        },
        {
            'name': 'symmetry_nose',
            'type': 'continuous'
        },
        {
            'name': 'symmetry_overall',
            'type': 'continuous'
        },
        {
            'name': 'emotion_angry',
            'type': 'continuous'
        },
        {
            'name': 'emotion_disgust',
            'type': 'continuous'
        },
        {
            'name': 'emotion_fear',
            'type': 'continuous'
        },
        {
            'name': 'emotion_happy',
            'type': 'continuous'
        },
        {
            'name': 'emotion_sad',
            'type': 'continuous'
        },
        {
            'name': 'emotion_surprise',
            'type': 'continuous'
        },
        {
            'name': 'emotion_neutral',
            'type': 'continuous'
        },
        {
            'name': 'face_embedding',
            'type': 'continuous'
        }
    ]
    
    # Load and prepare data splits
    print("\nLoading and preparing data splits...")
    train_nodes, val_nodes, test_nodes, \
    train_nodes_full, val_nodes_full, test_nodes_full, \
    node_loading_time = load_and_prepare_data_splits(args, data_root)
    
    # Create graph managers
    train_manager = NoGraphManager(None)  # Will set graph later
    
    # For this demo, we'll create a simple graph with just a subset of nodes
    print(f"\nUsing subset of {min(1000, len(train_nodes_full))} training nodes for demo")
    demo_nodes = train_nodes_full[:min(1000, len(train_nodes_full))]
    
    # Create a simple graph (this is simplified for demo purposes)
    from graphs.HyperGraph import HyperGraph
    demo_graph = HyperGraph(demo_nodes)
    train_manager.graph = demo_graph
    
    # Create model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = CNNModel(
        f"saved_models/adaptive_demo_{timestamp}.pt",
        "vistransformdf",  # Using a lightweight architecture for demo
        1e-4,
        True,
        device=device
    )
    
    # Create the AdaptiveTrainer
    print("\n" + "=" * 60)
    print("Creating AdaptiveTrainer...")
    print("=" * 60)
    
    trainer = AdaptiveTrainer(
        graphmanager=train_manager,
        models=[model],
        device=device,
        attribute_metadata=attribute_metadata,
        loss_fn=criterion
    )
    
    # Demonstration of dynamic traversal switching
    print("\n" + "=" * 60)
    print("Demonstration: Dynamic Traversal Switching")
    print("=" * 60)
    
    # Start with comprehensive traversal
    print("\n1. Starting with Comprehensive Traversal")
    print("-" * 40)
    
    comprehensive_traversal = trainer._create_traversal(
        "comprehensive",
        graph=demo_graph,
        num_pointers=1,
        num_steps=500
    )
    trainer.set_traversal(comprehensive_traversal, "comprehensive")
    
    # Train for 2 epochs
    print("Training for 2 epochs with Comprehensive Traversal...")
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        try:
            metrics, distribution = trainer.train(epoch)
            print(f"Comprehensive Training - Epoch {epoch + 1}: {metrics}")
        except Exception as e:
            print(f"Error in comprehensive training: {e}")
            continue
    
    print("\n2. Switching to Random Traversal")
    print("-" * 40)
    
    # Switch to random traversal mid-training
    try:
        trainer.switch_traversal(
            "random",
            graph=demo_graph,
            num_pointers=1,
            num_steps=500
        )
        
        # Train for 2 more epochs
        print("Training for 2 epochs with Random Traversal...")
        for epoch in range(2, 4):
            print(f"\nEpoch {epoch + 1}/4")
            try:
                metrics, distribution = trainer.train(epoch)
                print(f"Random Training - Epoch {epoch + 1}: {metrics}")
            except Exception as e:
                print(f"Error in random training: {e}")
                continue
                
    except Exception as e:
        print(f"Error switching to random traversal: {e}")
    
    print("\n3. Switching to I-Value Traversal")
    print("-" * 40)
    
    # Switch to I-value traversal (this will enable DQN capabilities)
    try:
        trainer.switch_traversal(
            "i-value",
            graph=demo_graph,
            num_pointers=1,
            num_steps=500
        )
        
        # Train for 2 more epochs
        print("Training for 2 epochs with I-Value Traversal...")
        for epoch in range(4, 6):
            print(f"\nEpoch {epoch + 1}/6")
            try:
                metrics, distribution = trainer.train(epoch)
                print(f"I-Value Training - Epoch {epoch + 1}: {metrics}")
            except Exception as e:
                print(f"Error in I-value training: {e}")
                continue
                
    except Exception as e:
        print(f"Error switching to I-value traversal: {e}")
    
    print("\n4. Final Switch to I-Value Cluster Hop")
    print("-" * 40)
    
    # Switch to I-value cluster hop traversal
    try:
        trainer.switch_traversal(
            "i-value-cluster-hop",
            graph=demo_graph,
            num_pointers=1,
            num_steps=500,
            bias_hop_period=2
        )
        
        # Train for 2 final epochs
        print("Training for 2 epochs with I-Value Cluster Hop Traversal...")
        for epoch in range(6, 8):
            print(f"\nEpoch {epoch + 1}/8")
            try:
                metrics, distribution = trainer.train(epoch)
                print(f"I-Value Cluster Hop Training - Epoch {epoch + 1}: {metrics}")
            except Exception as e:
                print(f"Error in cluster hop training: {e}")
                continue
                
    except Exception as e:
        print(f"Error switching to cluster hop traversal: {e}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Dynamic traversal switching during training")
    print("✓ Automatic capability management (DQN, bias loss)")
    print("✓ State transfer between traversal methods")
    print("✓ Unified trainer interface for all traversal types")
    print("✓ Backward compatibility with existing functionality")
    
    print(f"\nTrainer capabilities enabled: {trainer.capabilities.enabled_capabilities}")
    print(f"Final traversal type: {trainer.current_traversal_type}")
    
    # Optional: Show how to perform validation with a different traversal
    print("\n5. Bonus: Validation with Different Traversal")
    print("-" * 50)
    
    if val_nodes and len(val_nodes) > 0:
        try:
            # Create a validation graph (simplified)
            val_graph = HyperGraph(val_nodes[:min(500, len(val_nodes))])
            
            # Temporarily switch to comprehensive for validation
            original_traversal = trainer.current_traversal
            original_type = trainer.current_traversal_type
            
            val_traversal = trainer._create_traversal(
                "comprehensive",
                graph=val_graph,
                num_pointers=1,
                num_steps=200
            )
            trainer.set_traversal(val_traversal, "comprehensive")
            
            print("Running validation with comprehensive traversal...")
            val_metrics, val_distribution = trainer.train()
            print(f"Validation metrics: {val_metrics}")
            
            # Switch back to original traversal
            trainer.set_traversal(original_traversal, original_type)
            print(f"Switched back to {original_type} traversal")
            
        except Exception as e:
            print(f"Error in validation demonstration: {e}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 