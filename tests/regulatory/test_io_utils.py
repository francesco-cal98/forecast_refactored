"""Tests for regulatory.ingest.io_utils module.

Tests I/O utilities: logging, directory creation, and Parquet I/O.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.regulatory.ingest.io_utils import (
    ensure_dirs,
    get_logger,
    safe_to_parquet,
)


class TestGetLogger:
    """Tests for logger creation and configuration."""

    def test_get_logger_creates_logger(self):
        """Test get_logger creates a configured logger."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_default_level(self):
        """Test logger has INFO level by default."""
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO

    def test_get_logger_custom_level_string(self):
        """Test logger accepts custom level as string."""
        logger = get_logger("test_debug", level="DEBUG")
        assert logger.level == logging.DEBUG

        logger = get_logger("test_warning", level="WARNING")
        assert logger.level == logging.WARNING

        logger = get_logger("test_error", level="ERROR")
        assert logger.level == logging.ERROR

    def test_get_logger_custom_level_int(self):
        """Test logger accepts custom level as integer."""
        logger = get_logger("test_int_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_get_logger_case_insensitive_level(self):
        """Test logger level string is case-insensitive."""
        logger = get_logger("test_case", level="debug")
        assert logger.level == logging.DEBUG

        logger = get_logger("test_case2", level="WaRnInG")
        assert logger.level == logging.WARNING

    def test_get_logger_invalid_level_defaults_to_info(self):
        """Test logger defaults to INFO for invalid level strings."""
        logger = get_logger("test_invalid", level="INVALID_LEVEL")
        assert logger.level == logging.INFO

    def test_get_logger_has_handler(self):
        """Test logger has at least one handler configured."""
        logger = get_logger("test_handler")
        assert len(logger.handlers) > 0

    def test_get_logger_handler_has_formatter(self):
        """Test logger handler has a formatter."""
        logger = get_logger("test_formatter")
        handler = logger.handlers[0]
        assert handler.formatter is not None

    def test_get_logger_does_not_duplicate_handlers(self):
        """Test calling get_logger multiple times doesn't add duplicate handlers."""
        logger1 = get_logger("test_no_dup")
        initial_handlers = len(logger1.handlers)

        logger2 = get_logger("test_no_dup")
        assert len(logger2.handlers) == initial_handlers


class TestEnsureDirs:
    """Tests for directory creation utility."""

    def test_ensure_dirs_creates_directory(self, tmp_path):
        """Test ensure_dirs creates a new directory."""
        new_dir = tmp_path / "test_dir"
        assert not new_dir.exists()

        ensure_dirs(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_dirs_creates_parent_for_file(self, tmp_path):
        """Test ensure_dirs creates parent directory for file paths."""
        file_path = tmp_path / "subdir" / "file.txt"
        parent_dir = file_path.parent

        assert not parent_dir.exists()
        ensure_dirs(file_path)
        assert parent_dir.exists()
        assert parent_dir.is_dir()

    def test_ensure_dirs_creates_nested_directories(self, tmp_path):
        """Test ensure_dirs creates nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        ensure_dirs(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_ensure_dirs_handles_existing_directory(self, tmp_path):
        """Test ensure_dirs handles already existing directories."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        # Should not raise error
        ensure_dirs(existing_dir)
        assert existing_dir.exists()

    def test_ensure_dirs_handles_string_paths(self, tmp_path):
        """Test ensure_dirs works with string paths."""
        new_dir = str(tmp_path / "string_dir")
        ensure_dirs(new_dir)
        assert Path(new_dir).exists()

    def test_ensure_dirs_handles_path_objects(self, tmp_path):
        """Test ensure_dirs works with Path objects."""
        new_dir = tmp_path / "path_dir"
        ensure_dirs(new_dir)
        assert new_dir.exists()

    def test_ensure_dirs_handles_multiple_paths(self, tmp_path):
        """Test ensure_dirs handles multiple paths at once."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2" / "subdir"
        file_path = tmp_path / "dir3" / "file.csv"

        ensure_dirs(dir1, dir2, file_path)

        assert dir1.exists()
        assert dir2.exists()
        assert file_path.parent.exists()

    def test_ensure_dirs_handles_none(self, tmp_path):
        """Test ensure_dirs gracefully handles None in path list."""
        dir1 = tmp_path / "dir1"
        # Should not raise error
        ensure_dirs(dir1, None, None)
        assert dir1.exists()

    def test_ensure_dirs_distinguishes_file_vs_directory(self, tmp_path):
        """Test ensure_dirs distinguishes between file and directory paths."""
        # File path (has suffix)
        file_path = tmp_path / "subdir" / "file.parquet"
        ensure_dirs(file_path)
        assert file_path.parent.exists()
        assert not file_path.exists()  # File itself not created

        # Directory path (no suffix)
        dir_path = tmp_path / "another_dir"
        ensure_dirs(dir_path)
        assert dir_path.exists()
        assert dir_path.is_dir()


class TestSafeToParquet:
    """Tests for safe Parquet writing utility."""

    def test_safe_to_parquet_writes_dataframe(self, tmp_path):
        """Test safe_to_parquet writes DataFrame to file."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        output_path = tmp_path / "test.parquet"

        safe_to_parquet(df, output_path)

        assert output_path.exists()
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_safe_to_parquet_creates_parent_directories(self, tmp_path):
        """Test safe_to_parquet creates parent directories if needed."""
        output_path = tmp_path / "level1" / "level2" / "test.parquet"
        df = pd.DataFrame({"col": [1, 2]})

        assert not output_path.parent.exists()
        safe_to_parquet(df, output_path)

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_safe_to_parquet_handles_empty_dataframe(self, tmp_path):
        """Test safe_to_parquet handles empty DataFrame (schema-only)."""
        df = pd.DataFrame(columns=["work_uri", "celex", "title"])
        output_path = tmp_path / "empty.parquet"

        safe_to_parquet(df, output_path)

        assert output_path.exists()
        loaded_df = pd.read_parquet(output_path)
        assert loaded_df.empty
        assert list(loaded_df.columns) == ["work_uri", "celex", "title"]

    def test_safe_to_parquet_accepts_string_path(self, tmp_path):
        """Test safe_to_parquet accepts string paths."""
        df = pd.DataFrame({"x": [1, 2]})
        output_path = str(tmp_path / "string_path.parquet")

        safe_to_parquet(df, output_path)
        assert Path(output_path).exists()

    def test_safe_to_parquet_accepts_path_object(self, tmp_path):
        """Test safe_to_parquet accepts Path objects."""
        df = pd.DataFrame({"x": [1, 2]})
        output_path = tmp_path / "path_object.parquet"

        safe_to_parquet(df, output_path)
        assert output_path.exists()

    def test_safe_to_parquet_overwrites_existing(self, tmp_path):
        """Test safe_to_parquet overwrites existing files."""
        output_path = tmp_path / "overwrite.parquet"

        # Write first DataFrame
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        safe_to_parquet(df1, output_path)

        # Write second DataFrame
        df2 = pd.DataFrame({"b": [4, 5]})
        safe_to_parquet(df2, output_path)

        # Should contain df2, not df1
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df2, loaded_df)

    def test_safe_to_parquet_preserves_dtypes(self, tmp_path):
        """Test safe_to_parquet preserves DataFrame dtypes."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        output_path = tmp_path / "dtypes.parquet"

        safe_to_parquet(df, output_path)
        loaded_df = pd.read_parquet(output_path)

        assert loaded_df["int_col"].dtype == df["int_col"].dtype
        assert loaded_df["float_col"].dtype == df["float_col"].dtype
        assert loaded_df["str_col"].dtype == df["str_col"].dtype
        assert loaded_df["bool_col"].dtype == df["bool_col"].dtype

    def test_safe_to_parquet_no_index(self, tmp_path):
        """Test safe_to_parquet writes without index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        output_path = tmp_path / "no_index.parquet"

        safe_to_parquet(df, output_path)
        loaded_df = pd.read_parquet(output_path)

        # Index should not be preserved
        assert loaded_df.index.tolist() == [0, 1, 2]
