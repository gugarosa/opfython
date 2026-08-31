from opfython.utils import logging


def test_logging_helpers():
    console = logging.get_console_handler()
    file_handler = logging.get_timed_file_handler()
    logger = logging.get_logger(__name__)

    assert console.formatter is logging.FORMATTER
    assert file_handler.formatter is logging.FORMATTER
    assert logger.name == __name__
    assert logger.hasHandlers()

    console.close()
    file_handler.close()
