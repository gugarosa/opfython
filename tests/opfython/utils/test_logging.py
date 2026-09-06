# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import io
import logging as std_logging
import sys
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import Mock

import pytest

from opfython.utils import logging


@pytest.fixture
def isolated_logger(request, monkeypatch, tmp_path):
    logger = std_logging.getLogger(f"{__name__}.{request.node.name}")
    original_parent = logger.parent
    logger.parent = None
    monkeypatch.setattr(logging, "LOG_FILE", str(tmp_path / "opfython.log"))
    try:
        yield logger
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.parent = original_parent


def test_logging_helpers(tmp_path, monkeypatch):
    path = tmp_path / "helpers.log"
    monkeypatch.setattr(logging, "LOG_FILE", str(path))
    console = logging.get_console_handler()
    file_handler = logging.get_timed_file_handler()

    try:
        assert console.formatter is logging.FORMATTER
        assert console.stream is sys.stdout
        assert file_handler.formatter is logging.FORMATTER
        assert file_handler.delay is True
        assert file_handler.when == "MIDNIGHT"
        assert file_handler.stream is None
        assert not path.exists()
    finally:
        console.close()
        file_handler.close()


def test_get_logger_applies_fresh_defaults(isolated_logger, tmp_path):
    logger = logging.get_logger(isolated_logger.name)

    assert logger is isolated_logger
    assert logger.level == std_logging.DEBUG
    assert logger.propagate is False
    assert logger.hasHandlers()
    assert len(logger.handlers) == 2
    console, file_handler = logger.handlers
    assert type(console) is std_logging.StreamHandler
    assert console.stream is sys.stdout
    assert console.formatter is logging.FORMATTER
    assert isinstance(file_handler, TimedRotatingFileHandler)
    assert file_handler.baseFilename == str(tmp_path / "opfython.log")
    assert file_handler.formatter is logging.FORMATTER
    assert file_handler.when == "MIDNIGHT"
    assert file_handler.delay is True
    assert file_handler.stream is None
    assert not (tmp_path / "opfython.log").exists()


def test_get_logger_reuses_handlers_and_resources(isolated_logger, monkeypatch, tmp_path, capsys):
    logger = logging.get_logger(isolated_logger.name)
    handlers = logger.handlers[:]
    monkeypatch.setattr(logging, "get_console_handler", Mock(side_effect=RuntimeError("Unexpected console handler")))
    monkeypatch.setattr(logging, "get_timed_file_handler", Mock(side_effect=RuntimeError("Unexpected file handler")))

    for _ in range(3):
        assert logging.get_logger(logger.name) is logger

    assert logger.handlers == handlers
    assert handlers[1].stream is None
    assert not (tmp_path / "opfython.log").exists()

    logger.info("single-record")
    stream = handlers[1].stream
    assert logging.get_logger(logger.name) is logger
    assert handlers[1].stream is stream
    assert not stream.closed
    assert capsys.readouterr().out.count("single-record") == 1
    assert (tmp_path / "opfython.log").read_text().count("single-record") == 1

    handlers[1].close()
    assert stream.closed
    assert handlers[1].stream is None


def test_get_logger_preserves_application_handlers(isolated_logger, tmp_path):
    output = io.StringIO()
    handler = std_logging.StreamHandler(output)
    formatter = std_logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(std_logging.ERROR)
    isolated_logger.addHandler(handler)
    isolated_logger.setLevel(std_logging.ERROR)
    application_filter = std_logging.Filter()
    isolated_logger.addFilter(application_filter)

    logger = logging.get_logger(isolated_logger.name)

    assert logger is isolated_logger
    assert logger.handlers == [handler]
    assert logger.level == std_logging.ERROR
    assert logger.filters == [application_filter]
    assert logger.propagate is True
    assert handler.formatter is formatter
    assert handler.level == std_logging.ERROR
    logger.error("application-record")
    assert output.getvalue() == "application-record\n"
    assert not output.closed
    assert not (tmp_path / "opfython.log").exists()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("level", std_logging.INFO), ("propagate", False), ("disabled", True), ("filters", [std_logging.Filter()])],
)
def test_get_logger_preserves_configuration_without_handlers(isolated_logger, attribute, value, tmp_path):
    setattr(isolated_logger, attribute, value)

    assert logging.get_logger(isolated_logger.name) is isolated_logger

    assert getattr(isolated_logger, attribute) == value
    assert not isolated_logger.handlers
    assert not (tmp_path / "opfython.log").exists()


def test_get_logger_preserves_inherited_application_handlers(isolated_logger, tmp_path):
    output = io.StringIO()
    parent = std_logging.Logger(f"{isolated_logger.name}.parent", level=std_logging.WARNING)
    handler = std_logging.StreamHandler(output)
    parent.addHandler(handler)
    isolated_logger.parent = parent

    try:
        logger = logging.get_logger(isolated_logger.name)

        assert logger is isolated_logger
        assert not logger.handlers
        assert logger.level == std_logging.NOTSET
        assert logger.propagate is True
        assert parent.handlers == [handler]
        logger.warning("inherited-record")
        assert output.getvalue() == "inherited-record\n"
        assert not output.closed
        assert not (tmp_path / "opfython.log").exists()
    finally:
        parent.removeHandler(handler)
        handler.close()
