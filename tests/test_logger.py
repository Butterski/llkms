import logging

from llkms.utils.logger import setup_logger


def test_setup_logger():
    logger = setup_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.hasHandlers()
