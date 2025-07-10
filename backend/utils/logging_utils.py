import logging
import os

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional: file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
