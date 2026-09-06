# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import Mock

import pytest

from opfython.utils import exception


@pytest.mark.parametrize(
    "error",
    [
        exception.Error("Error", "error"),
        exception.ArgumentError("error"),
        exception.BuildError("error"),
        exception.SizeError("error"),
        exception.TypeError("error"),
        exception.ValueError("error"),
    ],
)
def test_package_exceptions_are_raiseable(error):
    with pytest.raises(exception.Error):
        raise error

    assert str(error) == "error"


@pytest.mark.parametrize(
    "error_type",
    [exception.ArgumentError, exception.BuildError, exception.SizeError, exception.TypeError, exception.ValueError],
)
@pytest.mark.parametrize("message", ["unchanged message", "`x` must be valid.", "punctuation...", "line\nbreak", ""])
def test_package_exceptions_preserve_caller_messages(error_type, message):
    error = error_type(message)

    assert error.args == (message,)
    assert str(error) == message


def test_base_error_logs_a_formatted_diagnostic_without_changing_arguments(monkeypatch):
    log = Mock()
    monkeypatch.setattr(exception.logger, "error", log)
    message = "Caller punctuation..."

    error = exception.Error("ApplicationError", message)

    assert error.args == (message,)
    assert str(error) == message
    format_string, *arguments = log.call_args.args
    diagnostic = format_string % tuple(arguments)
    assert "`cls=ApplicationError`" in diagnostic
    assert diagnostic.endswith(".")
    assert not diagnostic.endswith("..")
