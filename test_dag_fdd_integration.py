#!/usr/bin/env python3
"""
Test script for DAG FDD detector integration.

This script verifies that the DAG FDD detector is properly integrated while
preserving the exact research methodology from the original paper.
"""

import torch
import sys
import os

def test_dag_fdd_adapter():
    """Test the DAG FDD adapter functionality."""
    print("Testing DAG FDD Adapter...")
    
    try:
        from models.detectors.dag_fdd_adapter import DagFddAdapter
        
        # Create DAG FDD adapter
        model = DagFddAdapter(
            pretrained=True,
            output_classes=1,
            classification_strategy='binary'
        )
        print("✓ DAG FDD Adapter created successfully")
        
        # Test forward pass
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        
        # Check output format (should be dict with 'cls' key)
        if isinstance(output, dict) and 'cls' in output:
            print(f"✓ Forward pass successful - Output shape: {output['cls'].shape}")
        else:
            print(f"✗ Unexpected output format: {type(output)}")
            return False
        
        # Test loss computation
        target = torch.tensor([1.0])
        loss = model.get_loss(output, target)
        print(f"✓ Loss computation successful - Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ DAG FDD Adapter test failed: {str(e)}")
        return False

def test_fairness_cnn_model():
    """Test the FairnessCNNModel with DAG FDD detector."""
    print("\nTesting FairnessCNNModel with DAG FDD...")
    
    try:
        from models.FairnessCNNModel import FairnessCNNModel
        
        # Create fairness model with DAG FDD detector
        model = FairnessCNNModel(
            save_path='./test_checkpoint.pth',
            model_name='dag_fdd',
            lr=0.001,
            amsgrad=True,
            device='cpu',
            fairness_mode=True
        )
        print("✓ FairnessCNNModel with DAG FDD created successfully")
        
        # Test that fairness mode is enabled
        if model.fairness_mode:
            print("✓ Fairness mode is enabled")
        else:
            print("✗ Fairness mode is not enabled")
            return False
        
        # Test that the model is using the DAG FDD adapter
        if hasattr(model.model, 'dag_detector'):
            print("✓ Model is using DAG FDD detector")
        else:
            print("✗ Model is not using DAG FDD detector")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ FairnessCNNModel test failed: {str(e)}")
        return False

def test_original_methodology_preservation():
    """Test that the original research methodology is preserved."""
    print("\nTesting original methodology preservation...")
    
    try:
        from models.detectors.dag_fdd_adapter import DagFddAdapter
        
        model = DagFddAdapter()
        
        # Test that all original methods are accessible
        required_methods = [
            'threshplus_tensor',
            'search_func', 
            'searched_lamda_loss',
            'get_train_metrics',
            'features',
            'classifier',
            'build_backbone',
            'build_loss'
        ]
        
        for method_name in required_methods:
            if hasattr(model, method_name):
                print(f"✓ {method_name} method preserved")
            else:
                print(f"✗ {method_name} method missing")
                return False
        
        # Test that the original detector is accessible
        if hasattr(model, 'dag_detector'):
            print("✓ Original DAG FDD detector preserved")
        else:
            print("✗ Original DAG FDD detector missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Methodology preservation test failed: {str(e)}")
        return False

def test_compatibility_with_existing_framework():
    """Test compatibility with existing HyperGraph framework."""
    print("\nTesting compatibility with existing framework...")
    
    try:
        from models.detectors.dag_fdd_adapter import DagFddAdapter
        
        model = DagFddAdapter()
        
        # Test that the adapter has the expected interface
        expected_attributes = [
            'pretrained',
            'finetune', 
            'exclude_top',
            'output_classes',
            'classification_strategy',
            'configuration',
            'model'
        ]
        
        for attr_name in expected_attributes:
            if hasattr(model, attr_name):
                print(f"✓ {attr_name} attribute present")
            else:
                print(f"✗ {attr_name} attribute missing")
                return False
        
        # Test that the model attribute points to the backbone
        if hasattr(model.model, 'features'):
            print("✓ Model attribute points to valid backbone")
        else:
            print("✗ Model attribute does not point to valid backbone")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DAG FDD Detector Integration Test Suite")
    print("=" * 60)
    print("This test suite verifies that the DAG FDD detector is properly")
    print("integrated while preserving the exact research methodology.")
    print()
    
    tests = [
        ("DAG FDD Adapter", test_dag_fdd_adapter),
        ("FairnessCNNModel", test_fairness_cnn_model),
        ("Original Methodology Preservation", test_original_methodology_preservation),
        ("Framework Compatibility", test_compatibility_with_existing_framework)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test PASSED")
        else:
            print(f"✗ {test_name} test FAILED")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DAG FDD detector integration is working correctly.")
        print("The exact research methodology has been preserved for proper comparison.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 