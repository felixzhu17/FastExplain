from loguru import logger as loguru_logger
import sys


class Logging:
    def __init__(self):
        self.logging_level = "INFO"
        self.enable_logging = True
        self._init_logger()

    @property
    def logger(self):
        self._init_logger()
        return self._logger

    def _init_logger(self):
        logger = loguru_logger
        logger.remove()
        if self.enable_logging:
            logger.add(
                sys.stdout,
                level=self.logging_level,
                colorize=True,
                format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level}</level> |  <level>{message}</level>",
            )
        self._logger = logger
