"""Configuration management for byutils."""

import os
from pathlib import Path


# Storage path constants (importable from other projects)
AUTODELETE_DIR = Path.home() / "nobackup" / "autodelete"
ARCHIVE_DIR = Path.home() / "nobackup" / "archive"


class Config:
    """Configuration for byutils HuggingFace cache management."""

    # Default cache directory
    DEFAULT_CACHE_DIR = AUTODELETE_DIR / "hf_cache"

    @staticmethod
    def get_cache_dir() -> Path:
        """
        Get cache directory from environment or default.

        Checks HF_HOME environment variable first, otherwise uses default.

        Returns:
            Path to cache directory
        """
        env_cache = os.getenv("HF_HOME")
        if env_cache:
            return Path(env_cache)
        return Config.DEFAULT_CACHE_DIR

    @staticmethod
    def get_models_cache() -> Path:
        """
        Get models cache directory.

        Returns:
            Path to models cache subdirectory
        """
        return Config.get_cache_dir() / "models"

    @staticmethod
    def get_datasets_cache() -> Path:
        """
        Get datasets cache directory.

        Returns:
            Path to datasets cache subdirectory
        """
        return Config.get_cache_dir() / "datasets"

    @staticmethod
    def setup_environment() -> None:
        """
        Set up HuggingFace environment variables.

        This ensures HuggingFace libraries use our configured cache directories.
        """
        cache_dir = Config.get_cache_dir()
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(Config.get_models_cache())
        os.environ["HF_DATASETS_CACHE"] = str(Config.get_datasets_cache())
