# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Package exception types."""

from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class Error(Exception):
    """Base package error."""

    def __init__(self, cls: str, msg: str) -> None:
        """Initialize an error without changing the supplied message.

        Args:
            cls: Error name used in the construction-time diagnostic.
            msg: Caller-supplied message retained in the exception arguments.

        """

        super().__init__(msg)
        logger.error("`cls=%s` was constructed: %s.", cls, str(msg).rstrip("."))


class ArgumentError(Error):
    """Wrong argument error."""

    def __init__(self, error: str) -> None:
        """Initialize an argument error.

        Args:
            error: Caller-supplied exception message.

        """

        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Unbuilt object error."""

    def __init__(self, error: str) -> None:
        """Initialize a build error.

        Args:
            error: Caller-supplied exception message.

        """

        super().__init__("BuildError", error)


class SizeError(Error):
    """Invalid size error."""

    def __init__(self, error: str) -> None:
        """Initialize a size error.

        Args:
            error: Caller-supplied exception message.

        """

        super().__init__("SizeError", error)


class TypeError(Error):
    """Invalid type error."""

    def __init__(self, error: str) -> None:
        """Initialize a type error.

        Args:
            error: Caller-supplied exception message.

        """

        super().__init__("TypeError", error)


class ValueError(Error):
    """Invalid value error."""

    def __init__(self, error: str) -> None:
        """Initialize a value error.

        Args:
            error: Caller-supplied exception message.

        """

        super().__init__("ValueError", error)
