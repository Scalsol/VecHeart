# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging

from reimu.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='vecheart', log_file=log_file, log_level=log_level)

    return logger
