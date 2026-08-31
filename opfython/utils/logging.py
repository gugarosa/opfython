"""Logging helpers."""

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "opfython.log"


def get_console_handler() -> StreamHandler:
    """Return a formatted stdout handler."""

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Return a formatted daily rotating file handler."""

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Return the package-configured logger."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())
    logger.propagate = False
    return logger
