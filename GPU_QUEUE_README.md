# GPU Queue System for HyperGraph Test UI

## Overview

The HyperGraph Test UI now includes a GPU-based queueing system that manages test run execution based on GPU availability. This system ensures efficient resource utilization by automatically queuing runs when no GPUs are available and starting them when GPUs become free.

## Features

### 🚀 Automatic GPU Management
- **GPU Detection**: Automatically detects available GPUs using GPUtil or PyTorch
- **Memory Monitoring**: Tracks GPU memory usage and availability
- **Resource Allocation**: Allocates specific GPUs to test runs with memory restrictions

### 📋 Smart Queue Management
- **Priority Queue**: Supports priority-based run scheduling
- **Automatic Processing**: Automatically starts queued runs when GPUs become available
- **Queue Status**: Real-time queue status monitoring and display

### 🎯 Run Status Tracking
- **Queued**: Runs waiting for GPU availability
- **Running**: Active runs with allocated GPUs
- **Completed**: Successfully finished runs
- **Failed**: Runs that encountered errors
- **Stopped**: Manually stopped runs

## Architecture

### Core Components

1. **GPUQueueManager** (`web_ui/gpu_queue_manager.py`)
   - Main queue management system
   - GPU allocation and monitoring
   - Run scheduling and execution
   - Background thread management

2. **Updated Web UI** (`web_ui/app.py`)
   - Integrated with GPU queue manager
   - New API endpoints for GPU status
   - Queue-aware run management

3. **Enhanced Templates**
   - Real-time GPU status display
   - Queue information in dashboard
   - Dynamic status updates

### Key Methods

#### GPUQueueManager
- `queue_run(config_name, config, priority=0)`: Add run to queue
- `get_gpu_info()`: Get detailed GPU information
- `get_available_gpus(min_memory_gb)`: Find available GPUs
- `estimate_gpu_memory_requirement(config)`: Estimate memory needs
- `get_queue_status()`: Get current queue status

#### API Endpoints
- `GET /api/gpu/status`: Get GPU and queue status
- `GET /api/gpu/queue`: Get queue status only
- `POST /api/test-runs`: Queue a new test run
- `GET /api/test-runs`: List all runs with status

## Usage

### Starting a Test Run

1. **Create Configuration**: Use the web UI to create a test configuration
2. **Queue Run**: Click "Start Run" - the system will automatically queue the run
3. **Monitor Status**: Check the runs page to see queue position and GPU allocation
4. **Automatic Execution**: Runs start automatically when GPUs become available

### Monitoring Queue Status

#### Dashboard
- **Active Runs**: Shows currently running tests with GPU allocation
- **Queued Runs**: Shows runs waiting for GPU availability
- **GPU Status**: Real-time GPU availability and memory usage

#### Runs Page
- **Queue Panel**: Detailed GPU and queue information
- **Status Badges**: Clear status indicators (Running, Queued, Completed, etc.)
- **GPU Allocation**: Shows which GPU each running test is using

### Priority System

Runs can be assigned priorities (higher numbers = higher priority):
- **Priority 0**: Normal priority (default)
- **Priority 1**: High priority
- **Priority 2**: Very high priority

Higher priority runs are processed first when GPUs become available.

## GPU Memory Estimation

The system automatically estimates GPU memory requirements based on:

- **Batch Size**: Larger batches require more memory
- **Model Architecture**: Transformer models require more memory
- **Number of Epochs**: Longer training requires more memory
- **Base Requirement**: Minimum 2GB for basic operations

### Memory Requirements by Model Type
- **ResNet/ResNeSt**: +1GB
- **Swin/Transformer**: +2GB
- **Large Batch (>64)**: +2GB
- **Many Epochs (>50)**: +1GB

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Restrict GPU access (set automatically by queue manager)
- `PYTHONHASHSEED`: For reproducibility (handled by test script)

### Queue Settings
- `min_gpu_memory_gb`: Minimum GPU memory required (default: 2.0GB)
- `gpu_check_interval`: GPU monitoring frequency (default: 5 seconds)
- `queue_check_interval`: Queue processing frequency (default: 2 seconds)

## Testing

Run the test script to verify GPU queue functionality:

```bash
python test_gpu_queue.py
```

This will test:
- GPU detection and information retrieval
- Queue functionality and run management
- GPU allocation and run execution
- Run stopping and cleanup

## Troubleshooting

### Common Issues

1. **No GPUs Detected**
   - Check if GPUtil is installed: `pip install GPUtil`
   - Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU drivers and CUDA installation

2. **Runs Stuck in Queue**
   - Check GPU memory availability
   - Verify minimum memory requirements
   - Check for GPU allocation conflicts

3. **Queue Not Processing**
   - Check background threads are running
   - Verify queue check interval settings
   - Check for exceptions in logs

### Debug Information

The system provides detailed logging:
- GPU detection and allocation
- Queue processing decisions
- Run execution status
- Error handling and recovery

### Manual Override

If needed, you can manually:
- Stop running tests via the web UI
- Clear the queue by restarting the server
- Adjust GPU memory requirements in the configuration

## Future Enhancements

### Planned Features
- **Multi-GPU Support**: Run single tests across multiple GPUs
- **Resource Reservation**: Reserve GPUs for specific users/teams
- **Advanced Scheduling**: Time-based scheduling and resource limits
- **Load Balancing**: Distribute load across multiple machines
- **Notification System**: Email/Slack notifications for queue status

### Performance Optimizations
- **Memory Optimization**: Better memory estimation algorithms
- **Queue Optimization**: Improved priority handling and scheduling
- **Monitoring**: Enhanced GPU monitoring and alerting

## API Reference

### GPU Status Response
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 3080",
      "memory_total_gb": 10.0,
      "memory_free_gb": 8.5,
      "memory_utilization": 15.0,
      "status": "available",
      "allocated_to": null
    }
  ],
  "queue_status": {
    "queue_length": 2,
    "active_runs": ["run_20241201_120000_abc123"],
    "available_gpus": [0, 1],
    "queued_runs": [
      {
        "run_id": "run_20241201_120100_def456",
        "config_name": "test_config",
        "priority": 1,
        "estimated_gpu_memory": 3.0
      }
    ]
  }
}
```

### Run Status Values
- `queued`: Waiting for GPU availability
- `running`: Currently executing on GPU
- `completed`: Successfully finished
- `failed`: Encountered an error
- `stopped`: Manually stopped

## Contributing

When contributing to the GPU queue system:

1. **Test GPU Detection**: Ensure compatibility with different GPU setups
2. **Memory Estimation**: Improve accuracy of memory requirement estimation
3. **Error Handling**: Add robust error handling for GPU failures
4. **Documentation**: Update this README with new features

## Support

For issues with the GPU queue system:
1. Check the troubleshooting section above
2. Review the application logs
3. Run the test script to verify functionality
4. Check GPU availability and memory usage 