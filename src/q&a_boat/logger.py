import logging
import os
from logging.handlers import RotatingFileHandler


# ANSI Escape Codes for colored console output
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red Background
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        format_str = f"[%(asctime)s] {log_color}%(levelname)-8s\033[0m [%(filename)s:%(lineno)d] -> %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a pre-configured logger instance."""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # 1. Console Handler (Colored Output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.INFO
    )  # Change to DEBUG if you want verbose local logs
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # 2. File Handler (Rotating log file, Standard Text Output)
    log_dir = r"C:\Users\Pragneshsinh.Rathod\Downloads\All_Projects\Git_Repo_Python\Rag_based_Q&A_boat_over_own_doc\Rag_based_Q&A_boat_over_own_doc\logs"
    os.makedirs(log_dir, exist_ok=True)

    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Keeps up to 5 historical log files, max 5MB each
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything in the file
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
