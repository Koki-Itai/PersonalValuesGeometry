import logging
import os

import yaml


def setup_logger(
    name: str, log_level: str = "DEBUG", log_file: str | None = None
) -> logging.Logger:
    with open("logging.yaml") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    logger = logging.getLogger(name)

    if log_level:
        logger.setLevel(log_level.upper())

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
