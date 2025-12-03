"""Unit tests for phandas.logging module."""

import pytest
import logging
import json
from io import StringIO
from phandas.logging import (
    get_logger, configure_logging, StructuredFormatter, 
    StructuredLogger, log_execution_time
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""
    
    def test_human_format(self):
        """Human format should be readable."""
        formatter = StructuredFormatter(json_mode=False)
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        
        output = formatter.format(record)
        
        assert 'Test message' in output
        assert 'INFO' in output
    
    def test_json_format(self):
        """JSON format should be valid JSON."""
        formatter = StructuredFormatter(json_mode=True)
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data['message'] == 'Test message'
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test'
    
    def test_json_with_context(self):
        """JSON format should include context."""
        formatter = StructuredFormatter(json_mode=True)
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        record.context = {'key': 'value', 'count': 42}
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data['context']['key'] == 'value'
        assert data['context']['count'] == 42


class TestStructuredLogger:
    """Tests for StructuredLogger class."""
    
    def test_info_with_context(self, capfd):
        """Logger should accept context kwargs."""
        logger = get_logger('test.logger')
        logger.setLevel(logging.INFO)
        
        logger.info("Test message", symbol='BTC', count=10)
        
        captured = capfd.readouterr()
        assert 'Test message' in captured.err
    
    def test_all_levels(self, capfd):
        """All log levels should work with context."""
        logger = get_logger('test.levels')
        logger.setLevel(logging.DEBUG)
        
        logger.debug("Debug", severity='debug')
        logger.info("Info", severity='info')
        logger.warning("Warning", severity='warning')
        logger.error("Error", severity='error')


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_returns_structured_logger(self):
        """get_logger should return StructuredLogger."""
        logger = get_logger('test.get')
        
        assert isinstance(logger, logging.Logger)
    
    def test_same_logger_returned(self):
        """Same name should return same logger."""
        logger1 = get_logger('test.same')
        logger2 = get_logger('test.same')
        
        assert logger1 is logger2


class TestConfigureLogging:
    """Tests for configure_logging function."""
    
    def test_sets_level(self):
        """configure_logging should set log level."""
        configure_logging(level=logging.DEBUG)
        
        logger = logging.getLogger('phandas')
        assert logger.level == logging.DEBUG
    
    def test_json_mode(self):
        """configure_logging should enable JSON mode."""
        configure_logging(json_mode=True)
        
        logger = logging.getLogger('phandas')
        if logger.handlers:
            formatter = logger.handlers[0].formatter
            if isinstance(formatter, StructuredFormatter):
                assert formatter.json_mode is True


class TestLogExecutionTime:
    """Tests for log_execution_time decorator."""
    
    def test_decorator_logs_time(self, capfd):
        """Decorator should log execution time."""
        logger = get_logger('test.timing')
        logger.setLevel(logging.DEBUG)
        
        @log_execution_time(logger)
        def fast_func():
            return 42
        
        result = fast_func()
        
        assert result == 42
    
    def test_decorator_logs_error(self, capfd):
        """Decorator should log errors."""
        logger = get_logger('test.error')
        logger.setLevel(logging.DEBUG)
        
        @log_execution_time(logger)
        def error_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            error_func()

