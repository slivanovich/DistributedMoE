import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union


class LoggerAdapter(logging.LoggerAdapter):
    # Custom logger adapter that adds context information to log records.

    def process(self, msg, kwargs):
        # Add context information to the log message.
        if self.extra:
            context_parts = []
            for key, value in self.extra.items():
                context_parts.append(f"{key}={value}")
            if context_parts:
                context_str = " | ".join(context_parts)
                msg = f"[{context_str}] {msg}"
        return msg, kwargs


class CustomFormatter(logging.Formatter):
    # Custom formatter with color support and structured output.

    # Color codes for different log levels.
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan.
        "INFO": "\033[32m",  # Green.
        "WARNING": "\033[33m",  # Yellow.
        "ERROR": "\033[31m",  # Red.
        "CRITICAL": "\033[35m",  # Magenta.
        "RESET": "\033[0m",  # Reset.
    }

    def __init__(self, use_colors: bool = True, include_thread: bool = True):
        self.use_colors = use_colors and sys.stderr.isatty()
        self.include_thread = include_thread

        # Base format string.
        fmt_parts = ["%(asctime)s", "%(name)s", "%(levelname)s"]

        if self.include_thread:
            fmt_parts.append("%(threadName)s")

        fmt_parts.append("%(message)s")

        fmt_string = " - ".join(fmt_parts)
        super().__init__(fmt_string, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Add color if enabled.
        if self.use_colors:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB.
    backup_count: int = 5,
    use_colors: bool = True,
) -> None:
    # Create log directory if needed.
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for log files.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging.
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"()": CustomFormatter, "use_colors": use_colors, "include_thread": True},
            "file": {"()": CustomFormatter, "use_colors": False, "include_thread": True},
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {},
        "loggers": {
            # Root logger — no handlers, only level. Third-party logs won't appear unless explicitly configured.
            "": {"level": "WARNING", "handlers": []},
            # Top-level project loggers — all handlers attached here. Child loggers propagate up to them.
            "fault_tolerance": {"level": log_level, "handlers": [], "propagate": False},
            "dist_moe": {"level": log_level, "handlers": [], "propagate": False},
            # Child loggers — no handlers, only level. They propagate to dist_moe.
            "dist_moe.tte": {"level": log_level, "handlers": [], "propagate": True},
            "dist_moe.moe": {"level": log_level, "handlers": [], "propagate": True},
            "dist_moe.utils": {"level": log_level, "handlers": [], "propagate": True},
        },
    }

    handlers = []

    # Console handler.
    if enable_console_logging:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "console",
            "stream": "ext://sys.stderr",
        }
        handlers.append("console")

    # File handlers.
    if enable_file_logging:
        # Main log file.
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "file",
            "filename": str(log_path / f"dist_moe_{timestamp}.log"),
            "maxBytes": max_file_size,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        handlers.append("file")

        # Error log file.
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(log_path / f"dist_moe_errors_{timestamp}.log"),
            "maxBytes": max_file_size,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        handlers.append("error_file")

        # Debug log file (only if debug level is enabled).
        if log_level == "DEBUG":
            config["handlers"]["debug_file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": str(log_path / f"dist_moe_debug_{timestamp}.log"),
                "maxBytes": max_file_size,
                "backupCount": backup_count,
                "encoding": "utf-8",
            }
            handlers.append("debug_file")

    # Assign handlers to the top-level project loggers. Child loggers propagate to them.
    config["loggers"]["dist_moe"]["handlers"] = handlers
    config["loggers"]["fault_tolerance"]["handlers"] = handlers

    # Apply configuration.
    logging.config.dictConfig(config)

    # Log the configuration.
    logger = logging.getLogger("dist_moe.logging")
    logger.info(f"Logging configured - Level: {log_level}, Handlers: {handlers}")
    logger.info(f"Log directory: {log_path}")


def get_logger(name: str, context: Optional[Dict[str, str]] = None) -> Union[logging.Logger, LoggerAdapter]:
    # Get a logger with optional context information.
    logger = logging.getLogger(name)

    if context:
        return LoggerAdapter(logger, context)

    return logger


def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    # Log function entry with parameters.
    if kwargs:
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"Entering {func_name}({params})")
    else:
        logger.debug(f"Entering {func_name}()")


def log_function_exit(logger: logging.Logger, func_name: str, result=None):
    # Log function exit with optional result.
    if result is not None:
        logger.debug(f"Exiting {func_name}() -> {result}")
    else:
        logger.debug(f"Exiting {func_name}()")


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **metrics):
    # Log performance metrics.
    metric_parts = [f"duration={duration_ms:.2f}ms"]
    for key, value in metrics.items():
        metric_parts.append(f"{key}={value}")

    metrics_str = ", ".join(metric_parts)
    logger.info(f"Performance [{operation}]: {metrics_str}")


# Environment-based configuration.
def configure_from_environment():
    # Configure logging based on environment variables.
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = os.getenv("LOG_DIR")
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
    enable_console_logging = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
    use_colors = os.getenv("USE_COLORS", "true").lower() == "true"

    setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging,
        use_colors=use_colors,
    )


# Initialize logging on module import if not already configured.
if not logging.getLogger().handlers:
    configure_from_environment()
