import logging
import sys
from pathlib import Path

def setup_logger():
    """
    Configure and return a global logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger('llkms')
    logger.setLevel(logging.INFO)
    
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    file_handler = logging.FileHandler(log_dir / 'llkms.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
