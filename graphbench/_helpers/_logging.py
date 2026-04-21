import logging


_DEFAULT_FORMAT = "[%(levelname)s] %(message)s"


def get_logger(name, level=logging.INFO, fmt=_DEFAULT_FORMAT):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
