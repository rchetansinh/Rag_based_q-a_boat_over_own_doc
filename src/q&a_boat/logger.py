import logging
import os
from logging.handlers import RotatingFileHandler


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI colour codes to console output.

    Each log level is assigned a distinct colour so severity is immediately
    visible when reading terminal output:

    - DEBUG    → Cyan
    - INFO     → Green
    - WARNING  → Yellow
    - ERROR    → Red
    - CRITICAL → Red background

    The colour is applied only to the level name; the rest of the log line
    (timestamp, file, line number, message) is rendered in the terminal's
    default colour.  ANSI codes have no effect in non-TTY outputs (log files,
    CI pipelines) where plain text is preferred anyway.
    """

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Apply the colour for the record's level and delegate to a standard
        ``logging.Formatter`` for the rest of the formatting.

        Parameters
        ----------
        record : logging.LogRecord
            The log record emitted by a ``logger.info()`` / ``logger.error()``
            etc. call.

        Returns
        -------
        str
            The fully formatted, colour-annotated log line.
        """
        log_color = self.COLORS.get(record.levelno, self.RESET)
        format_str = (
            f"[%(asctime)s] {log_color}%(levelname)-8s\033[0m "
            f"[%(filename)s:%(lineno)d] -> %(message)s"
        )
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a pre-configured logger instance.

    Each module in the pipeline calls this with its own ``__name__`` so log
    lines include the originating file name and line number, making it easy
    to trace errors back to their source.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module (e.g. ``"ingestion.loader"``).
        Python's logging system uses this as the logger's identity; loggers
        with the same name share the same instance, so calling this function
        twice with the same name is safe — the second call returns the already-
        configured logger without adding duplicate handlers.

    Handlers configured
    -------------------
    Console handler (INFO and above)
        Coloured output via ``ColorFormatter``.  Useful for watching ingestion
        progress and query logs in the terminal during development.

    File handler (DEBUG and above)
        Plain-text, rotating log written to
        ``src/q&a_boat/logs/app.log``.
        Rotation settings: max 5 MB per file, 5 historical files kept.
        The file captures DEBUG messages that are suppressed on the console,
        providing a full audit trail without cluttering the terminal.

    Returns
    -------
    logging.Logger
        Configured logger ready for use with ``.info()``, ``.warning()``,
        ``.error()``, etc.
    """
    logger = logging.getLogger(name)

    # Guard against duplicate handlers if the logger is already initialised.
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Console handler — coloured, INFO+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # File handler — plain text, DEBUG+, rotating
    log_dir = r"/Users/prathod/Projects/Rag_based_q-a_boat_over_own_doc/src/q&a_boat/logs"
    os.makedirs(log_dir, exist_ok=True)

    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
