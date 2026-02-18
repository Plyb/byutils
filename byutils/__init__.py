"""
byutils: Cache-aware HuggingFace loading utilities for BYU ORC supercomputer.

Provides drop-in replacements for HuggingFace loaders with intelligent cache
management and automatic internet detection.

Example usage:
    >>> from byutils import load_model, load_tokenizer, load_dataset
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> # Automatically handles cache and internet detection
    >>> model = load_model(
    ...     "meta-llama/Llama-3.1-8B-Instruct",
    ...     model_class=AutoModelForCausalLM,
    ...     device_map="auto"
    ... )
    >>> tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
    >>> dataset = load_dataset("lmsys/lmsys-chat-1m")

Storage paths for use in other projects:
    >>> from byutils import AUTODELETE_DIR, ARCHIVE_DIR
    >>> data_dir = AUTODELETE_DIR / "my_project" / "data"
    >>> results_dir = ARCHIVE_DIR / "my_project" / "results"
"""

__version__ = "0.1.0"

from ._config import Config, AUTODELETE_DIR, ARCHIVE_DIR
from .loaders import (
    load_model,
    load_tokenizer,
    load_dataset,
    load_pipeline,
    prefetch_model,
    prefetch_dataset,
)
from .exceptions import (
    BYUtilsError,
    CacheMissError,
    InternetRequiredError,
)

# Set up environment on import
Config.setup_environment()

__all__ = [
    # Main loading functions
    "load_model",
    "load_tokenizer",
    "load_dataset",
    "load_pipeline",
    # Prefetch functions
    "prefetch_model",
    "prefetch_dataset",
    # Configuration and storage paths
    "Config",
    "AUTODELETE_DIR",
    "ARCHIVE_DIR",
    # Exceptions
    "BYUtilsError",
    "CacheMissError",
    "InternetRequiredError",
]
