import torch
import importlib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Test model initialization with CUDA
try:
    print("Step 1: Importing model module")
    model_name = "swintransformdf"  # One of the models mentioned in your test script
    ActiveModel = importlib.import_module(f'models.detectors.{model_name}').ModelOut
    
    print(f"Step 2: Creating {model_name} model")
    model = ActiveModel(
        pretrained=True,
        finetune=True,
        output_classes=1,
        classification_strategy='binary'
    )
    
    print("Step 3: Moving model to CUDA")
    model.model.cuda()
    
    print("Step 4: Creating a test tensor and moving to CUDA")
    # Create a sample input tensor (batch of 2, 3 channels, 255x255)
    sample_input = torch.randn(2, 3, 255, 255)
    sample_input_cuda = sample_input.cuda()
    
    print("Step 5: Running model inference")
    with torch.no_grad():
        output = model.model(sample_input_cuda)
    
    print(f"Step 6: Success! Output shape: {output.shape}")
    
except Exception as e:
    print(f"Error during model execution: {str(e)}")
    import traceback
    traceback.print_exc()
