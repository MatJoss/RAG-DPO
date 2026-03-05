"""
Structured Logger — JSON structured logging for the RAG-DPO application.

Replaces plain text logging with JSON lines for machine-parseable logs.
All modules that use `logging.getLogger(__name__)` automatically benefit.

Usage:
    from src.utils.structured_logger import setup_structured_logging
    setup_structured_logging()  # Call once at app startup
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


from src.utils.paths import LOGS_DIR

DEFAULT_LOG_DIR = LOGS_DIR
DEFAULT_LOG_FILE = "app.jsonl"
MAX_LOG_SIZE_MB = 20  # Rotation après 20 MB


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields (structured data from logger.info("msg", extra={...}))
        for key in ("event", "question", "total_time", "n_sources", "n_cited",
                     "model", "error", "component", "action", "detail",
                     "retrieval_time", "generation_time", "duration"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)

        return json.dumps(entry, ensure_ascii=False)


class RotatingJSONLHandler(logging.FileHandler):
    """File handler with simple size-based rotation for JSONL files."""

    def __init__(self, filepath: Path, max_size_mb: int = MAX_LOG_SIZE_MB):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_size_mb * 1024 * 1024
        super().__init__(str(filepath), mode='a', encoding='utf-8')

    def emit(self, record: logging.LogRecord):
        # Check rotation before writing
        try:
            if os.path.exists(self.baseFilename):
                size = os.path.getsize(self.baseFilename)
                if size > self.max_bytes:
                    self._rotate()
        except OSError:
            pass
        super().emit(record)

    def _rotate(self):
        """Rotate: current → .1, delete old .1."""
        self.close()
        rotated = self.baseFilename + '.1'
        try:
            if os.path.exists(rotated):
                os.unlink(rotated)
            os.rename(self.baseFilename, rotated)
        except OSError:
            pass
        self.stream = self._open()


def setup_structured_logging(
    log_dir: Optional[Path] = None,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = logging.INFO,
    console: bool = True,
    console_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
):
    """Configure structured JSON logging for the entire application.

    Call this ONCE at application startup. All existing loggers will
    output JSON to the log file, and optionally human-readable to console.

    Args:
        log_dir: Directory for log files. Defaults to project_root/logs/
        log_file: Name of the JSON log file
        level: Minimum log level
        console: Whether to also log to console (human-readable)
        console_format: Format string for console output
    """
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates (Streamlit reruns)
    root_logger.handlers.clear()

    # JSON file handler — structured, machine-parseable
    json_handler = RotatingJSONLHandler(log_dir / log_file)
    json_handler.setFormatter(JSONFormatter())
    json_handler.setLevel(level)
    root_logger.addHandler(json_handler)

    # Console handler — human-readable (optional)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(console_format))
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Silence noisy libraries
    for noisy in ("httpx", "ollama", "chromadb", "sentence_transformers",
                   "urllib3", "httpcore", "hpack", "markdown_it"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Structured logging initialized",
        extra={"event": "logging_init", "component": "structured_logger"}
    )
