"""Package exception types."""

from opfython.utils import logging

logger = logging.get_logger(__name__)


class Error(Exception):
    """Base package error."""

    def __init__(self, cls: str, msg: str) -> None:
        super().__init__(msg)
        logger.error("%s: %s.", cls, msg)


class ArgumentError(Error):
    """Wrong argument error."""

    def __init__(self, error: str) -> None:
        super().__init__("ArgumentError", error)


class BuildError(Error):
    """Unbuilt object error."""

    def __init__(self, error: str) -> None:
        super().__init__("BuildError", error)


class SizeError(Error):
    """Invalid size error."""

    def __init__(self, error: str) -> None:
        super().__init__("SizeError", error)


class TypeError(Error):
    """Invalid type error."""

    def __init__(self, error: str) -> None:
        super().__init__("TypeError", error)


class ValueError(Error):
    """Invalid value error."""

    def __init__(self, error: str) -> None:
        super().__init__("ValueError", error)
