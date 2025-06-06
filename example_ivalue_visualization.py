#!/usr/bin/env python3
"""
Example script demonstrating I-value visualization during training.

This script shows various ways to run the hierarchical test with I-value 
visualization enabled for different traversal configurations.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {description} interrupted by user")
        return False

def main():
    """Run various I-value visualization examples."""
    
    print("I-VALUE VISUALIZATION EXAMPLES")
    print("This script demonstrates different ways to visualize I-value changes during training.")
    
    # Base arguments for all examples
    base_args = [
        sys.executable, "test_hierarchical.py",
        "--num-epochs", "5",  # Short training for demo
        "--batch-size", "50",
        "--viz-sample-size", "500",  # Smaller sample for faster processing
        "--viz-track-nodes", "20",   # Track fewer nodes for demo
        "--trainer-mode", "adaptive"
    ]
    
    examples = [
        {
            "name": "Basic I-value Traversal with Visualization",
            "description": "Train using I-value traversal with comprehensive visualization tracking",
            "args": base_args + [
                "--enable-ivalue-viz",
                "--traversal-type", "i-value",
                "--architectures", "vistransformdf"
            ]
        },
        {
            "name": "I-value Cluster Hop with Bias Analysis",
            "description": "Train using I-value cluster hop traversal with bias hop visualization",
            "args": base_args + [
                "--enable-ivalue-viz",
                "--traversal-type", "i-value-cluster-hop",
                "--bias_hop_period", "50",  # More frequent hops for demo
                "--architectures", "vistransformdf"
            ]
        },
        {
            "name": "Traversal Switching with Visualization",
            "description": "Switch from comprehensive to I-value-cluster-hop with visualization",
            "args": base_args + [
                "--enable-ivalue-viz",
                "--enable-traversal-switching",
                "--traversal-sequence", "comprehensive,i-value-cluster-hop",
                "--switch-epochs", "2",  # Switch early for demo
                "--bias_hop_period", "30",
                "--architectures", "vistransformdf"
            ]
        },
        {
            "name": "Compare All Traversals with Visualization",
            "description": "Test all traversal types with I-value visualization where applicable",
            "args": base_args + [
                "--enable-ivalue-viz",
                "--test-all-traversals",
                "--architectures", "vistransformdf"
            ]
        }
    ]
    
    print(f"\nFound {len(examples)} examples to run.")
    print("Each example will create visualization plots in the 'ivalue_visualizations' directory.")
    
    # Ask user which examples to run
    print("\nAvailable examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   {example['description']}")
    
    print(f"\n{len(examples)+1}. Run all examples")
    print(f"{len(examples)+2}. Exit")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(examples)+2}): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(examples) + 2:  # Exit
                print("Exiting...")
                return
            elif choice_num == len(examples) + 1:  # Run all
                selected_examples = examples
                break
            elif 1 <= choice_num <= len(examples):  # Run specific example
                selected_examples = [examples[choice_num - 1]]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(examples)+2}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Create visualization directory if it doesn't exist
    os.makedirs("ivalue_visualizations", exist_ok=True)
    
    # Run selected examples
    successful_runs = 0
    total_runs = len(selected_examples)
    
    for i, example in enumerate(selected_examples, 1):
        print(f"\n\nRunning example {i}/{total_runs}: {example['name']}")
        
        if run_command(example['args'], example['description']):
            successful_runs += 1
        
        # Ask user if they want to continue (except for last example)
        if i < total_runs:
            try:
                continue_choice = input("\nContinue to next example? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    print("Stopping at user request...")
                    break
            except KeyboardInterrupt:
                print("\nStopping at user request...")
                break
    
    print(f"\n\nSUMMARY:")
    print(f"Completed {successful_runs}/{total_runs} examples successfully")
    
    if successful_runs > 0:
        print(f"\n📊 Visualization plots saved in:")
        print(f"   - ivalue_visualizations/ (main directory)")
        print(f"   - Check subdirectories for each configuration")
        
        print(f"\n📋 Generated visualizations include:")
        print(f"   - Training progression plots (mean, distribution, ratios)")
        print(f"   - Subgroup analysis (bias across demographic groups)")
        print(f"   - Individual node tracking (detailed I-value evolution)")
        print(f"   - Bias hop analysis (for cluster hop traversal)")
        print(f"   - Raw data JSON files for further analysis")
        
    print(f"\nExample script completed!")

if __name__ == "__main__":
    main() 