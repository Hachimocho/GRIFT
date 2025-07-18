#!/usr/bin/env python3
"""
Test script for GPU Queue Manager

This script tests the basic functionality of the GPU queue manager
to ensure it works correctly before integrating with the web UI.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_ui.gpu_queue_manager import GPUQueueManager

def test_gpu_queue_manager():
    """Test the GPU queue manager functionality."""
    print("Testing GPU Queue Manager...")
    
    # Initialize the queue manager
    queue_manager = GPUQueueManager()
    
    try:
        # Test 1: Get GPU information
        print("\n1. Testing GPU information retrieval...")
        gpu_info = queue_manager.get_gpu_info()
        print(f"Found {len(gpu_info)} GPUs:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']} - {gpu['status']}")
            print(f"    Memory: {gpu['memory_free_gb']:.1f}GB free / {gpu['memory_total_gb']:.1f}GB total")
        
        # Test 2: Get available GPUs
        print("\n2. Testing available GPU detection...")
        available_gpus = queue_manager.get_available_gpus()
        print(f"Available GPUs: {available_gpus}")
        
        # Test 3: Test queue functionality
        print("\n3. Testing queue functionality...")
        
        # Create a test configuration
        test_config = {
            "architectures": ["resnestdf"],
            "num_epochs": 5,
            "batch_size": 32,
            "traversal_type": "random",
            "seed": 42
        }
        
        # Queue a test run
        run_id = queue_manager.queue_run("test_config", test_config, priority=1)
        print(f"Queued test run with ID: {run_id}")
        
        # Check queue status
        queue_status = queue_manager.get_queue_status()
        print(f"Queue status: {json.dumps(queue_status, indent=2)}")
        
        # Test 4: List runs
        print("\n4. Testing run listing...")
        runs = queue_manager.list_runs()
        print(f"Total runs: {len(runs)}")
        for run in runs:
            print(f"  {run['run_id']}: {run['status']}")
        
        # Test 5: Get specific run
        print("\n5. Testing run retrieval...")
        run_details = queue_manager.get_run(run_id)
        if run_details:
            print(f"Run details: {json.dumps(run_details, indent=2)}")
        
        # Test 6: Test GPU memory estimation
        print("\n6. Testing GPU memory estimation...")
        estimated_memory = queue_manager.estimate_gpu_memory_requirement(test_config)
        print(f"Estimated GPU memory requirement: {estimated_memory:.1f}GB")
        
        # Test 7: Test queue processing (if GPUs are available)
        print("\n7. Testing queue processing...")
        if available_gpus:
            print(f"GPUs are available, queue should process automatically")
            print("Waiting 5 seconds to see if queue processes...")
            time.sleep(5)
            
            # Check if run was started
            updated_run = queue_manager.get_run(run_id)
            if updated_run and updated_run.get('status') == 'running':
                print(f"✅ Run {run_id} was successfully started on GPU {updated_run.get('gpu_id')}")
                
                # Test stopping the run
                print("\n8. Testing run stopping...")
                success = queue_manager.stop_run(run_id)
                if success:
                    print(f"✅ Run {run_id} was successfully stopped")
                else:
                    print(f"❌ Failed to stop run {run_id}")
            else:
                print(f"Run {run_id} is still queued or failed to start")
        else:
            print("No GPUs available, run remains queued")
        
        # Final status check
        print("\n9. Final status check...")
        final_runs = queue_manager.list_runs()
        print(f"Final run count: {len(final_runs)}")
        for run in final_runs:
            print(f"  {run['run_id']}: {run['status']}")
        
        print("\n✅ GPU Queue Manager test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        queue_manager.shutdown()

if __name__ == "__main__":
    test_gpu_queue_manager() 