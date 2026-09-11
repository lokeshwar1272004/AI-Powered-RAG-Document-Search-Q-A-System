import logging
import os
from src.config import settings


def setup_logger(name:str)->logging.Logger:

    logger=logging.getLogger(name)

    # Avoid adding duplicate handlers if this is called more than once
    # for the same module (e.g. on module reload).
    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level)

    formatter=logging.Formatter(
        fmt="%(asctime)s| %(levelname)-8s | %(name)s |  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",

    )

     # Console handler — what you see while developing
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — permanent record on disk
    os.makedirs(os.path.dirname(settings.log_file),exist_ok=True)
    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
