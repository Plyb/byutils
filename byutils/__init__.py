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
from ._connectivity import is_login_node

# Set up the HF environment BEFORE importing loaders: loaders import huggingface_hub, which
# freezes its cache path (HF_HUB_CACHE) and related settings into module constants at import
# time -- so HF_HOME must be exported first or it silently has no effect.
Config.setup_environment()

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
    # Connectivity
    "is_login_node",
    # Exceptions
    "BYUtilsError",
    "CacheMissError",
    "InternetRequiredError",
]
