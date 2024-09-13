import wandb
import time

def upload_sweeps(sweep_list, batch_tag):
    uploaded_sweeps = []
    for sweep_config in sweep_list:
        # Add the batch tag and status tags to the sweep configuration
        if 'tags' not in sweep_config:
            sweep_config['tags'] = []
        sweep_config['tags'].extend([batch_tag, 'unclaimed', 'unfinished'])
        
        # Ensure gpu_usage is specified
        if 'gpu_usage' not in sweep_config:
            raise ValueError("Each sweep must specify 'gpu_usage' in GB")
        
        # Create a new sweep
        sweep_id = wandb.sweep(sweep_config)
        
        # Add the sweep_id to the list of uploaded sweeps
        uploaded_sweeps.append(sweep_id)
        
        print(f"Uploaded sweep: {sweep_id} with batch tag: {batch_tag}")
    
    return uploaded_sweeps

def check_batch_completion(project, batch_tag):
    api = wandb.Api()
    sweeps = api.sweeps(f"{project}")
    
    all_completed = True
    for sweep in sweeps:
        if batch_tag in sweep.tags and 'unfinished' in sweep.tags:
            all_completed = False
            break
    
    return all_completed

def run_completion_code(project, batch_tag):
    while not check_batch_completion(project, batch_tag):
        print("Waiting for all sweeps to complete...")
        time.sleep(300)  # Check every 5 minutes
    
    print("All sweeps in the batch have completed!")
    # Add your completion code here
    # For example:
    # send_notification()
    # generate_report()
    # start_next_batch()

# Example usage
sweep_list = [
    {
        'method': 'bayes',
        'metric': {'name': 'accuracy', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.1},
            'batch_size': {'values': [16, 32, 64, 128]},
        },
        'gpu_usage': 4  # GPU usage in GB
    },
    # Add more sweep configurations as needed
]

batch_tag = "batch_1"
project = "your_project_name"

uploaded_sweep_ids = upload_sweeps(sweep_list, batch_tag)

# Start the completion check in a separate thread or process
import threading
threading.Thread(target=run_completion_code, args=(project, batch_tag)).start()
