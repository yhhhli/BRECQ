import os
import logging
import torch
import linklink as link


_logger = None
_logger_fh = None
_logger_names = []


def create_logger(log_file, level=logging.INFO):
    global _logger, _logger_fh
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger_fh = fh
    else:
        _logger.removeHandler(_logger_fh)
        _logger.setLevel(level)

    return _logger


def get_logger(name, level=logging.INFO):
    global _logger_names
    logger = logging.getLogger(name)
    if name in _logger_names:
        return logger

    _logger_names.append(name)
    if link.get_rank() > 0:
        logger.addFilter(RankFilter())

    return logger


class RankFilter(logging.Filter):
    def filter(self, record):
        return False