import logging
import os
from datetime import datetime
# Load the config file using the config_loader module
from utils.config_loader import load_config


cfg = load_config()
log_directory = cfg['log_directory']

def setup_logger(log_directory = log_directory, resume=False): 
    """
    Set up a logger that writes logs to a .txt file in the logs directory.

    Args:
        checkpoint_dir (str): Directory where the checkpoint is saved.
        resume (bool): Whether training is resuming from an existing checkpoint.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(log_directory), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Determine log file name based on whether we are resuming training or not
    if resume:
        # Resume from the same log file if resuming from a checkpoint
        log_file_name = os.path.basename(log_directory).split('.')[0] + '.txt'
    else:
        # Create a new log file with timestamp if starting a new checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f'log_{timestamp}.txt'
    
    log_file_path = os.path.join(logs_dir, log_file_name)

    # Configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler to write logs to the file
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    
    # Console handler to output logs to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to the logger
    if not logger.hasHandlers():  # To prevent adding handlers multiple times
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger