"""HuggingFace loading functions with cache awareness."""

from typing import Optional, Any, Union
from pathlib import Path
import os

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline,
        Pipeline,
    )
    from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict, load_from_disk
    from huggingface_hub import snapshot_download
except ImportError as e:
    raise ImportError(
        f"Required dependencies not found: {e}\n"
        "Please install with: pip install transformers datasets huggingface-hub"
    ) from e

from ._config import Config
from ._connectivity import should_allow_download
from .exceptions import CacheMissError


def _get_cached_model_path(
    model_id: str,
    cache_dir: Optional[str] = None,
    allow_download: Optional[bool] = None
) -> str:
    """
    Internal function to resolve model to cache path.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Override cache directory
        allow_download: Override download permission

    Returns:
        Absolute path to cached model

    Raises:
        CacheMissError: If model not in cache and downloads disabled
    """
    cache = cache_dir or str(Config.get_models_cache())
    allow_dl = should_allow_download(force_download=allow_download)

    try:
        return snapshot_download(
            repo_id=model_id,
            cache_dir=cache,
            local_files_only=not allow_dl
        )
    except Exception as e:
        if not allow_dl:
            raise CacheMissError(model_id, cache) from e
        raise


def load_model(
    model_id: str,
    model_class: Optional[type] = None,
    cache_dir: Optional[str] = None,
    allow_download: Optional[bool] = None,
    **kwargs
) -> Any:
    """
    Load a HuggingFace model with cache awareness.

    Automatically detects whether to download or use cache based on
    execution environment (login vs compute node).

    Args:
        model_id: HuggingFace model identifier (e.g., "gpt2")
        model_class: Model class (defaults to AutoModel)
        cache_dir: Override cache directory
        allow_download: Explicitly allow/prevent downloads
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded model instance

    Raises:
        CacheMissError: If model not cached and downloads disabled

    Example:
        >>> from byutils import load_model
        >>> from transformers import AutoModelForCausalLM
        >>> model = load_model(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     model_class=AutoModelForCausalLM,
        ...     device_map="auto"
        ... )
    """
    if model_class is None:
        model_class = AutoModel

    model_path = _get_cached_model_path(model_id, cache_dir, allow_download)

    # Force local_files_only to avoid metadata checks
    kwargs.setdefault("local_files_only", True)

    return model_class.from_pretrained(model_path, **kwargs)


def load_tokenizer(
    model_id: str,
    cache_dir: Optional[str] = None,
    allow_download: Optional[bool] = None,
    **kwargs
) -> AutoTokenizer:
    """
    Load a HuggingFace tokenizer with cache awareness.

    Automatically detects whether to download or use cache based on
    execution environment (login vs compute node).

    Args:
        model_id: HuggingFace model identifier (e.g., "gpt2")
        cache_dir: Override cache directory
        allow_download: Explicitly allow/prevent downloads
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded tokenizer instance

    Raises:
        CacheMissError: If tokenizer not cached and downloads disabled

    Example:
        >>> from byutils import load_tokenizer
        >>> tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
    """
    model_path = _get_cached_model_path(model_id, cache_dir, allow_download)

    # Force local_files_only
    kwargs.setdefault("local_files_only", True)

    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_dataset(
    path: str,
    name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_download: Optional[bool] = None,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Load a HuggingFace dataset with cache awareness.

    Automatically detects whether to download or use cache based on
    execution environment (login vs compute node).

    Args:
        path: Dataset path or identifier (e.g., "lmsys/lmsys-chat-1m")
        name: Dataset configuration name
        cache_dir: Override cache directory
        allow_download: Explicitly allow/prevent downloads
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Loaded dataset instance

    Raises:
        CacheMissError: If dataset not cached and downloads disabled

    Example:
        >>> from byutils import load_dataset
        >>> dataset = load_dataset("lmsys/lmsys-chat-1m")
    """
    cache = cache_dir or str(Config.get_datasets_cache())
    allow_dl = should_allow_download(force_download=allow_download)

    # Construct the dataset cache path
    dataset_cache_path = Path(cache) / path / (name if name else "")
    dataset_cache_path = dataset_cache_path.resolve()

    # If offline mode and cache exists, load directly from disk to avoid network requests
    if not allow_dl and os.path.exists(dataset_cache_path):
        try:
            return load_from_disk(str(dataset_cache_path))
        except Exception as e:
            # Fall back to hf_load_dataset if load_from_disk fails
            pass

    # Set cache directory
    kwargs.setdefault("cache_dir", Path(cache))

    # Handle offline mode
    if not allow_dl:
        kwargs.setdefault("download_mode", "reuse_cache_if_exists")

    try:
        ds = hf_load_dataset(path, name, **kwargs)

        ds.save_to_disk(dataset_cache_path)

        return ds
    except Exception as e:
        if not allow_dl:
            resource_id = f"{path}/{name}" if name else path
            raise CacheMissError(resource_id, cache) from e
        raise


def load_pipeline(
    task: str,
    model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_download: Optional[bool] = None,
    **kwargs
) -> Pipeline:
    """
    Create a HuggingFace pipeline with cache awareness.

    Automatically detects whether to download or use cache based on
    execution environment (login vs compute node).

    Args:
        task: Pipeline task (e.g., "text-generation", "translation")
        model: Model identifier (optional)
        cache_dir: Override cache directory
        allow_download: Explicitly allow/prevent downloads
        **kwargs: Additional arguments passed to pipeline

    Returns:
        Pipeline instance

    Raises:
        CacheMissError: If model not cached and downloads disabled

    Example:
        >>> from byutils import load_pipeline
        >>> pipe = load_pipeline(
        ...     "text-generation",
        ...     model="meta-llama/Llama-3.1-8B-Instruct",
        ...     device_map="auto"
        ... )
    """
    if model is not None:
        model_path = _get_cached_model_path(model, cache_dir, allow_download)
        kwargs.setdefault("model", model_path)

    # Force local_files_only
    kwargs.setdefault("local_files_only", True)

    return pipeline(task, **kwargs)


def prefetch_model(
    model_id: str,
    cache_dir: Optional[str] = None
) -> str:
    """
    Pre-download a model to cache (run on login node).

    Use this function on a login node before submitting compute jobs
    to ensure the model is available in cache.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Override cache directory

    Returns:
        Path to cached model

    Example:
        >>> from byutils import prefetch_model
        >>> # Run on login node before submitting job
        >>> prefetch_model("meta-llama/Llama-3.1-8B-Instruct")
    """
    return _get_cached_model_path(model_id, cache_dir, allow_download=True)


def prefetch_dataset(
    path: str,
    name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Pre-download a dataset to cache (run on login node).

    Use this function on a login node before submitting compute jobs
    to ensure the dataset is available in cache.

    Args:
        path: Dataset path or identifier
        name: Dataset configuration name
        cache_dir: Override cache directory
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Loaded dataset

    Example:
        >>> from byutils import prefetch_dataset
        >>> # Run on login node before submitting job
        >>> prefetch_dataset("lmsys/lmsys-chat-1m")
    """
    return load_dataset(path, name, cache_dir, allow_download=True, **kwargs)
