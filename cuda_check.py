import torch
import os

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())
print("Number of devices:", torch.cuda.device_count())

# Environment variables
print("\nRelevant environment variables:")
for var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
    print(f"{var}: {os.environ.get(var, 'Not set')}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
else:
    print("\nTrying to diagnose the issue:")
    try:
        # Try to manually initialize CUDA
        device = torch.device("cuda:0")
        x = torch.tensor([1.0, 2.0], device=device)
        print("Manual CUDA initialization worked!")
    except Exception as e:
        print(f"Error when manually initializing CUDA: {str(e)}")
