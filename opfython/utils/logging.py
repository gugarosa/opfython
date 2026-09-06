# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Logging helpers."""

import logging
import sys
from logging import Logger, StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "opfython.log"


def get_console_handler() -> StreamHandler:
    """Return a new formatted stdout handler.

    Returns:
        StreamHandler: Handler owned by the caller and bound to the current stdout.

    """

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Return a new formatted daily rotating file handler.

    Returns:
        TimedRotatingFileHandler: Caller-owned handler opening LOG_FILE on first emission and rotating at midnight.

    """

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Return a logger, applying package defaults only when it is unconfigured.

    Existing handlers, inherited handlers, levels, filters, and propagation settings remain unchanged.
    A fresh logger receives stdout and delayed midnight-rotating file handlers at DEBUG level without propagation.

    Args:
        logger_name: Name registered with the standard logging module.

    Returns:
        Logger: Shared logger retaining application configuration or its first-use package defaults.

    """

    logger = logging.getLogger(logger_name)

    if not (
        logger.hasHandlers()
        or logger.level != logging.NOTSET
        or logger.filters
        or logger.disabled
        or not logger.propagate
    ):
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
        logger.propagate = False

    return logger
