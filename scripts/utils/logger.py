import logging
from rich.logging import RichHandler
from pathlib import Path

def get_logger(name: str, log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler(rich_tracebacks=True))
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True)
        logger.addHandler(logging.FileHandler(log_file))
    return logger