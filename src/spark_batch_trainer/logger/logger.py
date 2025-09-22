"""Logger module for the package.

Provides a standardized logger configuration with both file rotation
and console output, ensuring consistent logging across the application.
"""

from logging import getLogger, Formatter, INFO, StreamHandler
from logging.handlers import RotatingFileHandler
from os import makedirs, path

ROOT_DIR_LOGS = path.join(path.dirname(__file__), "logs")

if not path.exists(ROOT_DIR_LOGS):
    makedirs(ROOT_DIR_LOGS)


def setup_logger(name: str, log_path: str = ROOT_DIR_LOGS, level: int = INFO):
    """
    Setup a logger with file rotation and console output.

    Parameters
    ----------
    name : str
        Name of the logger (used as log file name as well).
    log_path : str, optional
        Directory path where log files will be stored.
        Defaults to the ``logs/`` folder relative to this module.
    level : int, optional
        Logging level (e.g., ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
        Default is ``INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    - Logs are written both to console (stdout) and to a rotating file.
    - Rotating file handler keeps logs up to 5 MB per file with up to 5 backups.
    - Prevents duplicate handlers from being added if called multiple times.

    Examples
    --------
    >>> from batchtrainingbooster.logger.logger import setup_logger
    >>> logger = setup_logger("training")
    >>> logger.info("Training process started.")
    2025-09-17 12:00:00 - training - INFO - Training process started.
    """
    if len(path.split(name)) > 1:
        name = path.split(name)[-1]

    logger = getLogger(name)

    if not logger.handlers:
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create the log folder
        makedirs(log_path, exist_ok=True)
        log_file = path.join(log_path, f"{name}.log")

        # File handler with rotation
        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
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