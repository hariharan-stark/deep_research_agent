import logging
import sys
import os

def setup_logger(name: str = "deep_research_agent", level: int = logging.INFO):
    """
    Sets up a centralized logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Force reconfiguration to ensure handlers are present
    # if logger.hasHandlers():
    #    return logger
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "agent.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
