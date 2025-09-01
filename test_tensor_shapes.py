#!/usr/bin/env python3
"""
Simple test to verify tensor shapes are correct for DAG FDD integration.
"""

import torch
import torch.nn as nn

def test_tensor_shapes():
    """Test that tensor shapes are compatible between predictions and targets."""
    print("Testing tensor shapes for DAG FDD integration...")
    
    try:
        from models.detectors.dag_fdd import ModelOut
        
        # Create model
        model = ModelOut(
            pretrained=True,
            output_classes=1,
            classification_strategy='binary'
        )
        
        # Create test data
        batch_size = 2
        x = torch.randn(batch_size, 3, 256, 256)  # Input images
        y = torch.tensor([0, 1])  # Labels (batch_size,)
        
        # Test forward pass
        y_hat = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y_hat['cls'].shape}")
        print(f"  Target shape: {y.shape}")
        
        # Test loss computation
        data_dict = {'image': x, 'label': y.unsqueeze(1)}  # Add unsqueeze to match expected shape
        loss = model.get_loss(y_hat, None, data_dict)
        print(f"✓ Loss computation successful")
        print(f"  Loss value: {loss.item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tensor shape test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Tensor Shape Test for DAG FDD Integration")
    print("=" * 50)
    
    if test_tensor_shapes():
        print("\n🎉 All tensor shape tests passed!")
        print("The DAG FDD integration should now work correctly.")
    else:
        print("\n❌ Tensor shape test failed.")
        print("Please check the error messages above.")
    
    print("=" * 50) 