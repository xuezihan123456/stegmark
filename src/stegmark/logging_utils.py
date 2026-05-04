from __future__ import annotations

import logging

logger = logging.getLogger("stegmark")
logger.addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO) -> None:
    if any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        logger.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
