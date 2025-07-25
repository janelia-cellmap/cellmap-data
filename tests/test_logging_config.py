"""Tests for centralized logging configuration."""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from cellmap_data.utils.logging_config import (
    CellMapLogger,
    get_logger,
    configure_logging,
    set_log_level,
    LoggingMixin,
    log_method_entry,
    log_method_exit,
    log_performance,
)


class TestCellMapLogger:
    """Test the CellMapLogger class."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        CellMapLogger.configure_logging()

        assert CellMapLogger._configured is True

        # Check that root logger is configured
        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_configure_logging_with_file(self):
        """Test logging configuration with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            CellMapLogger.configure_logging(filename=tmp_path)

            root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
            assert len(root_logger.handlers) == 2  # Console + file

            # Verify file handler
            file_handlers = [
                h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_configure_logging_no_console(self):
        """Test logging configuration without console output."""
        CellMapLogger.configure_logging(console=False)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        console_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) == 0

    def test_configure_logging_custom_level(self):
        """Test logging configuration with custom level."""
        CellMapLogger.configure_logging(level=logging.DEBUG)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.DEBUG

    def test_get_logger_component_names(self):
        """Test getting loggers by component name."""
        CellMapLogger.configure_logging()

        # Test component name mapping
        dataset_logger = CellMapLogger.get_logger("dataset")
        assert dataset_logger.name == "cellmap_data.dataset"

        image_logger = CellMapLogger.get_logger("image")
        assert image_logger.name == "cellmap_data.image"

    def test_get_logger_full_names(self):
        """Test getting loggers by full name."""
        CellMapLogger.configure_logging()

        # Test full name usage
        full_name_logger = CellMapLogger.get_logger("cellmap_data.custom.module")
        assert full_name_logger.name == "cellmap_data.custom.module"

    def test_get_logger_auto_configure(self):
        """Test that get_logger auto-configures logging."""
        # Should auto-configure when not already configured
        logger = CellMapLogger.get_logger("dataset")

        assert CellMapLogger._configured is True
        assert logger.name == "cellmap_data.dataset"

    def test_set_level(self):
        """Test setting log level for all loggers."""
        CellMapLogger.configure_logging()
        logger1 = CellMapLogger.get_logger("dataset")
        logger2 = CellMapLogger.get_logger("image")

        CellMapLogger.set_level(logging.DEBUG)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.DEBUG

    def test_logger_caching(self):
        """Test that loggers are cached correctly."""
        CellMapLogger.configure_logging()

        logger1 = CellMapLogger.get_logger("dataset")
        logger2 = CellMapLogger.get_logger("dataset")

        # Should return the same instance
        assert logger1 is logger2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_get_logger_function(self):
        """Test the get_logger convenience function."""
        logger = get_logger("dataset")
        assert logger.name == "cellmap_data.dataset"
        assert CellMapLogger._configured is True

    def test_configure_logging_function(self):
        """Test the configure_logging convenience function."""
        configure_logging(level=logging.DEBUG)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.DEBUG
        assert CellMapLogger._configured is True

    def test_set_log_level_function(self):
        """Test the set_log_level convenience function."""
        configure_logging()
        set_log_level(logging.WARNING)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.WARNING


class TestLoggingMixin:
    """Test the LoggingMixin class."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_logging_mixin(self):
        """Test LoggingMixin functionality."""

        class TestClass(LoggingMixin):
            pass

        # Mock the module name for testing
        TestClass.__module__ = "cellmap_data.test.module"
        TestClass.__init_subclass__()

        instance = TestClass()
        logger = instance.logger

        assert isinstance(logger, logging.Logger)
        assert (
            logger.name == "cellmap_data.test.module.TestClass"
        )  # Updated to match actual behavior


class TestContextualLogging:
    """Test contextual logging utilities."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_log_method_entry(self):
        """Test log_method_entry utility."""
        logger = get_logger("test")

        with patch.object(logger, "debug") as mock_debug:
            log_method_entry(logger, "test_method", param1="value1", param2="value2")
            mock_debug.assert_called_once_with(
                "Entering test_method with parameters: param1=value1, param2=value2"
            )

    def test_log_method_entry_no_params(self):
        """Test log_method_entry utility without parameters."""
        logger = get_logger("test")

        with patch.object(logger, "debug") as mock_debug:
            log_method_entry(logger, "test_method")
            mock_debug.assert_called_once_with("Entering test_method")

    def test_log_method_exit(self):
        """Test log_method_exit utility."""
        logger = get_logger("test")

        with patch.object(logger, "debug") as mock_debug:
            log_method_exit(logger, "test_method", result="success")
            mock_debug.assert_called_once_with(
                "Exiting test_method with result: success"
            )

    def test_log_method_exit_no_result(self):
        """Test log_method_exit utility without result."""
        logger = get_logger("test")

        with patch.object(logger, "debug") as mock_debug:
            log_method_exit(logger, "test_method")
            mock_debug.assert_called_once_with("Exiting test_method")

    def test_log_performance(self):
        """Test log_performance utility."""
        logger = get_logger("test")

        with patch.object(logger, "info") as mock_info:
            log_performance(logger, "data_loading", 1.234)
            mock_info.assert_called_once_with(
                "Performance: data_loading completed in 1.234s"
            )


class TestLoggingIntegration:
    """Test logging integration with existing modules."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_module_logger_creation(self):
        """Test that modules can create loggers correctly."""
        from cellmap_data.utils.logging_config import get_logger

        # Test that different modules get appropriate loggers
        dataset_logger = get_logger("dataset")
        image_logger = get_logger("image")

        assert dataset_logger.name == "cellmap_data.dataset"
        assert image_logger.name == "cellmap_data.image"
        assert dataset_logger is not image_logger

    def test_logging_level_propagation(self):
        """Test that logging level changes propagate correctly."""
        configure_logging(level=logging.INFO)

        dataset_logger = get_logger("dataset")
        image_logger = get_logger("image")

        # Change level
        set_log_level(logging.DEBUG)

        # Root logger should have new level
        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        assert root_logger.level == logging.DEBUG

    @patch("sys.stdout.write")
    def test_actual_logging_output(self, mock_write):
        """Test that logging actually produces output."""
        configure_logging(level=logging.INFO, console=True)
        logger = get_logger("test")

        logger.info("Test message")

        # Should have written to stdout (console handler)
        assert mock_write.called

    def test_no_duplicate_configuration(self):
        """Test that multiple configuration calls don't create duplicate handlers."""
        configure_logging()
        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        initial_handler_count = len(root_logger.handlers)

        # Configure again - should not add more handlers
        configure_logging()
        assert len(root_logger.handlers) == initial_handler_count

    def test_reset_configuration(self):
        """Test resetting configuration."""
        configure_logging()
        logger = get_logger("test")

        assert CellMapLogger._configured is True
        assert len(CellMapLogger._loggers) > 0

        CellMapLogger.reset_configuration()

        assert CellMapLogger._configured is False
        assert len(CellMapLogger._loggers) == 0


class TestLoggingConfiguration:
    """Test various logging configuration scenarios."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        CellMapLogger.reset_configuration()

    def teardown_method(self):
        """Clean up after each test."""
        CellMapLogger.reset_configuration()

    def test_component_logger_names(self):
        """Test that all component logger names are correctly mapped."""
        configure_logging()

        expected_components = [
            "dataset",
            "dataset_writer",
            "datasplit",
            "image",
            "image_writer",
            "multidataset",
            "dataloader",
            "transforms",
            "utils",
            "validation",
        ]

        for component in expected_components:
            logger = get_logger(component)
            expected_name = f"cellmap_data.{component}"
            assert logger.name == expected_name

    def test_custom_format_string(self):
        """Test custom format string configuration."""
        custom_format = "%(name)s - %(levelname)s - %(message)s"
        configure_logging(format_string=custom_format)

        root_logger = logging.getLogger(CellMapLogger.LIBRARY_NAME)
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # Check that the format string is applied using public API
        record = logging.LogRecord(
            name="cellmap_data.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        expected = f"{record.name} - {logging.getLevelName(record.levelno)} - {record.getMessage()}"
        assert formatted == expected
