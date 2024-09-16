import wandb
import time
import logging
from wandb.errors import CommError, Error as WandbError
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
LOCK_TIMEOUT = 1800  # 30 minutes in seconds (should be greater than HEARTBEAT_INTERVAL)

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
def get_all_sweeps(project):
    api = wandb.Api()
    return api.sweeps(project)

@retry_with_backoff
def release_stale_lock(project, sweep):
    api = wandb.Api()
    lock_name = f"lock_{sweep.id}"
    
    try:
        api.artifact(f"{project}/{lock_name}:latest").delete()
        logger.info(f"Deleted stale lock for sweep {sweep.id}")
    except CommError:
        logger.warning(f"Failed to delete stale lock artifact for sweep {sweep.id}")
    
    sweep.tags.remove('claimed')
    sweep.tags.append('unclaimed')
    sweep.update()
    logger.info(f"Released stale lock for sweep {sweep.id}")

def cleanup_main(project):
    while True:
        try:
            cleanup_stale_locks(project)
            time.sleep(300)  # Run cleanup every 5 minutes
        except Exception as e:
            logger.error(f"Unexpected error in cleanup_main: {str(e)}")
            time.sleep(60)  # Wait before retrying

@retry_with_backoff
def get_lock_artifact(project, sweep_id):
    api = wandb.Api()
    lock_name = f"lock_{sweep_id}"
    try:
        return api.artifact(f"{project}/{lock_name}:latest")
    except CommError:
        return None

def cleanup_stale_locks(project):
    sweeps = get_all_sweeps(project)
    current_time = time.time()
    
    for sweep in sweeps:
        if 'claimed' in sweep.tags:
            lock_artifact = get_lock_artifact(project, sweep.id)
            if lock_artifact:
                lock_time = float(lock_artifact.metadata.get('timestamp', 0))
                if current_time - lock_time > LOCK_TIMEOUT:
                    logger.info(f"Found potentially stale lock for sweep {sweep.id}")
                    # Check if the sweep is actually running
                    sweep_status = sweep.state
                    if sweep_status != 'running':
                        logger.info(f"Sweep {sweep.id} is not running. Releasing stale lock.")
                        release_stale_lock(project, sweep)
                    else:
                        logger.info(f"Sweep {sweep.id} is still running. Keeping the lock.")
            else:
                logger.warning(f"Sweep {sweep.id} is marked as claimed but has no lock artifact")
                release_stale_lock(project, sweep)

if __name__ == "__main__":
    config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
    project = config["project_name"]
    cleanup_main(project)
