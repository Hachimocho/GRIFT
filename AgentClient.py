import wandb
import GPUtil
import time
import random
import logging
import threading
from wandb.errors import CommError, Error as WandbError
from utils.WandbArtifactUtils import load_tag_runless
from utils.import_utils import import_classes_from_directory, load_class_from_globals, import_and_load
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
HEARTBEAT_INTERVAL = 600  # 10 minutes in seconds

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def retry_with_backoff(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (CommError, WandbError) as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Max retries reached. Function {func.__name__} failed: {str(e)}")
                    raise
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait} seconds...")
                time.sleep(wait)
    return wrapper

@retry_with_backoff
def find_unclaimed_sweeps(project, batch_tag):
    api = wandb.Api()
    sweeps = api.sweeps(f"{project}")
    
    unclaimed_sweeps = []
    for sweep in sweeps:
        if batch_tag in sweep.tags and 'unclaimed' in sweep.tags:
            unclaimed_sweeps.append(sweep)
    
    return unclaimed_sweeps

def check_gpu_availability(required_vram):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.memoryFree >= required_vram * 1024:  # Convert GB to MB
            return True
    return False

@retry_with_backoff
def try_claim_sweep(sweep, client_id):
    api = wandb.Api()
    lock_name = f"lock_{sweep.id}"
    
    try:
        artifact = wandb.Artifact(lock_name, type="lock")
        artifact.add_string("owner", client_id)
        artifact.add_string("timestamp", str(time.time()))
        api.log_artifact(artifact, project=sweep.project)
        
        sweep.tags.remove('unclaimed')
        sweep.tags.append('claimed')
        sweep.update()
        return True
    except CommError:
        return False

@retry_with_backoff
def release_sweep(sweep):
    api = wandb.Api()
    lock_name = f"lock_{sweep.id}"
    
    try:
        api.artifact(f"{sweep.project}/{lock_name}:latest").delete()
    except CommError:
        logger.warning(f"Failed to delete lock artifact for sweep {sweep.id}")
    
    sweep.tags.remove('claimed')
    sweep.tags.append('unclaimed')
    sweep.update()

@retry_with_backoff
def mark_sweep_finished(sweep):
    api = wandb.Api()
    sweep.tags.remove('unfinished')
    sweep.tags.append('finished')
    sweep.update()
    
    lock_name = f"lock_{sweep.id}"
    try:
        api.artifact(f"{sweep.project}/{lock_name}:latest").delete()
    except CommError:
        logger.warning(f"Failed to delete lock artifact for finished sweep {sweep.id}")

def client_main(project, batch_tag):
    client_id = f"client_{random.randint(1000, 9999)}"
    
    while True:
        try:
            unclaimed_sweeps = find_unclaimed_sweeps(project, batch_tag)
            
            for sweep in unclaimed_sweeps:
                required_vram = sweep.config.get('gpu_usage', 0)
                
                if check_gpu_availability(required_vram):
                    if try_claim_sweep(sweep, client_id):
                        logger.info(f"Successfully claimed sweep {sweep.id}")
                        run_sweep(sweep, client_id)
                    else:
                        logger.info(f"Failed to claim sweep {sweep.id}, it may have been claimed by another client")
                else:
                    logger.info(f"Not enough GPU memory for sweep {sweep.id}. Required: {required_vram}GB")
            
            time.sleep(60)  # Wait for 1 minute before checking for new sweeps
        except Exception as e:
            logger.error(f"Unexpected error in client_main: {str(e)}")
            time.sleep(60)  # Wait before retrying

def your_training_function(sweep_id, pid):
    try:
        data, dataloader, dataset, graph, manager, model, trainer, traversal, test_traversal = _load_config()
        wandb.agent(
            sweep_id=sweep_id, project=pid, count=wandb.config["optimizations_per_sweep"], function=trainer.run
        )
        print("Finished running sweep!")
    except RuntimeError as e:
        if "out of memory" in str(e):
            
        else:
            raise e

def _load_config():
    """
    Load configuration parameters for the agent training process.
    
    Load order: 
    1. The data, node, and edge classes are gathered
    2. The datasets are loaded using the data, node, and edge classes
    3. The dataloader is loaded using the datasets
    4. The graph is loaded using the dataloader
    5. The manager is loaded using the graph
    6. The models are loaded
    7. The traversals are loaded using the manager
    8. The trainer is loaded using the manager, traversals, and models
    
    Returns:
        trainer (Trainer): The loaded trainer object.
    """
    data, node, edge, dataloader, datasets, graph, manager, model, trainer, traversal = wandb.config["parameters"]["modules"]["value"]
    data_params = wandb.config["parameters"][data]
    node_params = wandb.config["parameters"][node]
    edge_params = wandb.config["parameters"][edge]
    loaded_datasets = []
    for dataset in datasets:
        loaded_dataset = import_and_load("datasets", dataset, **(wandb.config["parameters"][dataset].join({"data": [data, data_params], "node": [node, node_params]})))
        loaded_datasets.append(loaded_dataset)
    dataloader = import_and_load("dataloaders", dataloader, **(wandb.config["parameters"][dataloader].join({"datasets": loaded_datasets, "edge_class": [edge, edge_params]})))
    graph = dataloader.load()
    manager = import_and_load("managers", manager, **(wandb.config["parameters"][manager].join({"graph": graph})))
    models = []
    for _ in range(wandb.config["num_models"]):
        model = import_and_load("models", model, **wandb.config["parameters"][model])
        models.append(model)
    train_traversal = import_and_load("traversals", traversal, **(wandb.config["parameters"][traversal]).join({"graph": manager.graph}))
    test_traversal = import_and_load("traversals", next(iter(wandb.config["parameters"]["test_traversal"]["value"].keys())), next(iter(**(wandb.config["parameters"]["test_traversal"]["value"]).join({"graph": manager.graph}))))
    trainer = import_and_load("trainers", trainer, **(wandb.config["parameters"][trainer]).join({"manager": manager, "train_traversal": train_traversal, "test_traversal": test_traversal, "models": models}))
    return trainer

@retry_with_backoff
def update_lock_timestamp(sweep, client_id):
    api = wandb.Api()
    lock_name = f"lock_{sweep.id}"
    
    try:
        artifact = api.artifact(f"{sweep.project}/{lock_name}:latest")
        artifact.metadata['timestamp'] = str(time.time())
        artifact.save()
        logger.info(f"Updated timestamp for sweep {sweep.id}")
    except CommError:
        logger.warning(f"Failed to update timestamp for sweep {sweep.id}")

def heartbeat(sweep, client_id, stop_event):
    while not stop_event.is_set():
        update_lock_timestamp(sweep, client_id)
        time.sleep(HEARTBEAT_INTERVAL)

def run_sweep(sweep, client_id):
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat, args=(sweep, client_id, stop_event))
    heartbeat_thread.start()

    try:
        # Your sweep running code here
        wandb.agent(sweep.id, function=your_training_function)
        
        mark_sweep_finished(sweep)
    except Exception as e:
        logger.error(f"Error running sweep {sweep.id}: {str(e)}")
    finally:
        stop_event.set()
        heartbeat_thread.join()
        release_sweep(sweep)

if __name__ == "__main__":
    # Example usage
    config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
    project = config["project_name"]
    batch_tag = load_tag_runless(project)
    client_main(project, batch_tag)
