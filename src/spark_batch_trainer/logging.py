"""Logging configuration for the package.

Provides a standardized logger configuration with both file rotation
and console output, ensuring consistent logging across the application.
"""

from logging import getLogger, Formatter, INFO, StreamHandler
from logging.handlers import RotatingFileHandler
from os import makedirs, path
from typing import Optional


def configure_logger(name: str, log_path: Optional[str] = None, level: int = INFO):
    """Configure a logger with console output and optional file rotation.

    Args:
        name: Name of the logger (used as log file name as well).
        log_path: Optional directory for rotating log files. By default the
            library logs only to the console and never writes inside the
            installed package.
        level: Logging level (e.g., ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL``). Defaults to ``INFO``.

    Returns:
        logging.Logger: Configured logger instance.

    Note:
        Logs are always written to the console. File logging is opt-in
        through ``log_path``; rotating files keep up to 5 MB per file with
        five backups. Calling this twice for the same ``name`` does not add
        duplicate handlers.

    Example:
        >>> from spark_batch_trainer.logging import configure_logger
        >>> logger = configure_logger("training")
        >>> logger.info("Training process started.")
        2025-09-17 12:00:00 - training - INFO - Training process started.
    """
    if len(path.split(name)) > 1:
        name = path.split(name)[-1]

    logger = getLogger(name)

    if not logger.handlers:
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if log_path is not None:
            makedirs(log_path, exist_ok=True)
            log_file = path.join(log_path, f"{name}.log")
            handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Console handler
        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger.setLevel(level)

    return logger
