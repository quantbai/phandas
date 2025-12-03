"""Structured logging utilities for phandas.

Provides consistent logging across all modules with optional structured output
for log aggregation systems (ELK, Splunk, etc.).

Examples
--------
>>> from phandas.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Fetching data", symbols=['BTC', 'ETH'], source='binance')
"""

import logging
import json
import sys
from typing import Any, Dict, Optional
from functools import wraps
import time


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured log output.
    
    Formats log records as JSON for machine parsing while maintaining
    human-readable console output when not in JSON mode.
    
    Parameters
    ----------
    json_mode : bool, default False
        If True, output logs as JSON. If False, output human-readable format.
    
    Attributes
    ----------
    json_mode : bool
        Current output mode
    """
    
    def __init__(self, json_mode: bool = False):
        super().__init__()
        self.json_mode = json_mode
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record.
        
        Parameters
        ----------
        record : logging.LogRecord
            Log record to format
        
        Returns
        -------
        str
            Formatted log string (JSON or human-readable)
        """
        if self.json_mode:
            return self._format_json(record)
        return self._format_human(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON string."""
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'context') and record.context:
            log_data['context'] = record.context
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)
    
    def _format_human(self, record: logging.LogRecord) -> str:
        """Format as human-readable string."""
        base = f"{self.formatTime(record)} [{record.levelname:8}] {record.name}: {record.getMessage()}"
        
        if hasattr(record, 'context') and record.context:
            ctx_str = ' '.join(f"{k}={v}" for k, v in record.context.items())
            base = f"{base} | {ctx_str}"
        
        return base


class StructuredLogger(logging.Logger):
    """Logger with structured context support.
    
    Extends standard Logger to accept keyword arguments as structured context
    that can be output as JSON or appended to human-readable logs.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : int, default logging.NOTSET
        Logging level
    
    Examples
    --------
    >>> logger = StructuredLogger('phandas.data')
    >>> logger.info("Fetching data", symbols=['BTC'], timeframe='1d')
    """
    
    def _log_with_context(self, log_level: int, msg: str, args: tuple, 
                          exc_info: Any = None, extra: Optional[Dict] = None,
                          stack_info: bool = False, stacklevel: int = 1,
                          **context: Any) -> None:
        """Internal method to log with context."""
        if extra is None:
            extra = {}
        extra['context'] = context
        super()._log(log_level, msg, args, exc_info, extra, stack_info, stacklevel + 1)
    
    def debug(self, msg: str, *args, **context) -> None:
        """Log debug message with optional context."""
        if self.isEnabledFor(logging.DEBUG):
            self._log_with_context(logging.DEBUG, msg, args, **context)
    
    def info(self, msg: str, *args, **context) -> None:
        """Log info message with optional context."""
        if self.isEnabledFor(logging.INFO):
            self._log_with_context(logging.INFO, msg, args, **context)
    
    def warning(self, msg: str, *args, **context) -> None:
        """Log warning message with optional context."""
        if self.isEnabledFor(logging.WARNING):
            self._log_with_context(logging.WARNING, msg, args, **context)
    
    def error(self, msg: str, *args, **context) -> None:
        """Log error message with optional context."""
        if self.isEnabledFor(logging.ERROR):
            self._log_with_context(logging.ERROR, msg, args, **context)
    
    def critical(self, msg: str, *args, **context) -> None:
        """Log critical message with optional context."""
        if self.isEnabledFor(logging.CRITICAL):
            self._log_with_context(logging.CRITICAL, msg, args, **context)


def get_logger(name: str, json_mode: bool = False) -> StructuredLogger:
    """Get or create a structured logger.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    json_mode : bool, default False
        If True, output logs as JSON
    
    Returns
    -------
    StructuredLogger
        Configured logger instance
    
    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing", count=100, status='ok')
    """
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter(json_mode=json_mode))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def configure_logging(level: int = logging.INFO, json_mode: bool = False) -> None:
    """Configure global logging settings for phandas.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    json_mode : bool, default False
        If True, output all logs as JSON
    
    Examples
    --------
    >>> from phandas.logging import configure_logging
    >>> configure_logging(level=logging.DEBUG, json_mode=True)
    """
    logging.setLoggerClass(StructuredLogger)
    
    root_logger = logging.getLogger('phandas')
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter(json_mode=json_mode))
    root_logger.addHandler(handler)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time.
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. If None, uses function's module logger.
    
    Returns
    -------
    callable
        Decorated function
    
    Examples
    --------
    >>> @log_execution_time()
    ... def slow_function():
    ...     time.sleep(1)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _logger.debug(f"{func.__name__} completed", 
                             elapsed_ms=round(elapsed * 1000, 2))
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                _logger.error(f"{func.__name__} failed",
                             elapsed_ms=round(elapsed * 1000, 2),
                             error=str(e))
                raise
        return wrapper
    return decorator

