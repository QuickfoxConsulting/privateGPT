import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_tracebacks: bool = True,
    rich_markup: bool = True,
) -> None:
    """
    Set up logging configuration with rich formatting.
    
    Args:
        log_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file
        rich_tracebacks: Whether to use rich traceback formatting
        rich_markup: Whether to enable rich markup in log messages
    """
    # Configure rich handler with default theme to avoid compatibility issues
    rich_handler = RichHandler(
        show_time=True,
        show_path=True,
        rich_tracebacks=rich_tracebacks,
        markup=rich_markup,
        tracebacks_show_locals=True,
        tracebacks_extra_lines=3,
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add rich handler
    root_logger.addHandler(rich_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        root_logger.addHandler(file_handler)
    
    # Set up specific loggers
    loggers = {
        "uvicorn": "INFO",
        "uvicorn.access": "WARNING",
        "uvicorn.error": "INFO",
        "fastapi": "INFO",
        "private_gpt": log_level,
    }
    
    for logger_name, level in loggers.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Log startup message
    logging.info(f"Logging configured with level: {log_level}")
    if log_file:
        logging.info(f"Log file: {log_file}")