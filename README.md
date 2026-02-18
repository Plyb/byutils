# BYUtils

Cache-aware HuggingFace loading utilities for BYU's Office of Research Computing (ORC) supercomputer.

## Overview

BYUtils provides drop-in replacements for HuggingFace loaders that automatically handle the supercomputer's network restrictions. Compute nodes lack internet access, while login nodes have full connectivity. This library automatically detects your environment and uses cache appropriately.

## Features

- **Automatic detection** of login vs compute nodes
- **Cache-aware loading** for models, tokenizers, datasets, and pipelines
- **Clear error messages** when resources need to be prefetched
- **Storage path constants** for consistent file organization across projects
- **Zero configuration** required for standard usage

## Installation

```bash
pip install git+https://github.com/Plyb/byutils.git
```

## Quick Start

### Basic Usage

```python
from byutils import load_model, load_tokenizer
from transformers import AutoModelForCausalLM

# Automatically handles cache and internet detection
model = load_model(
    "meta-llama/Llama-3.1-8B-Instruct",
    model_class=AutoModelForCausalLM,
    device_map="auto"
)
tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
```

### Prefetch Workflow

The recommended workflow for compute jobs:

```python
# 1. On login node (before submitting job):
from byutils import prefetch_model, prefetch_dataset

prefetch_model("meta-llama/Llama-3.1-8B-Instruct")
prefetch_dataset("lmsys/lmsys-chat-1m")

# 2. In SLURM job script (compute node):
from byutils import load_model, load_dataset

# These will use cache without internet access
model = load_model("meta-llama/Llama-3.1-8B-Instruct")
dataset = load_dataset("lmsys/lmsys-chat-1m")
```

## API Reference

### Loading Functions

#### `load_model(model_id, model_class=None, cache_dir=None, allow_download=None, **kwargs)`

Load a HuggingFace model with cache awareness.

**Parameters:**
- `model_id` (str): HuggingFace model identifier (e.g., "gpt2")
- `model_class` (type, optional): Model class (defaults to AutoModel)
- `cache_dir` (str, optional): Override cache directory
- `allow_download` (bool, optional): Explicitly allow/prevent downloads
- `**kwargs`: Additional arguments passed to `from_pretrained`

**Returns:** Loaded model instance

**Example:**
```python
from byutils import load_model
from transformers import AutoModelForCausalLM

model = load_model(
    "gpt2",
    model_class=AutoModelForCausalLM,
    device_map="auto"
)
```

#### `load_tokenizer(model_id, cache_dir=None, allow_download=None, **kwargs)`

Load a HuggingFace tokenizer with cache awareness.

**Parameters:**
- `model_id` (str): HuggingFace model identifier
- `cache_dir` (str, optional): Override cache directory
- `allow_download` (bool, optional): Explicitly allow/prevent downloads
- `**kwargs`: Additional arguments passed to `from_pretrained`

**Returns:** Loaded tokenizer instance

#### `load_dataset(path, name=None, cache_dir=None, allow_download=None, **kwargs)`

Load a HuggingFace dataset with cache awareness.

**Parameters:**
- `path` (str): Dataset path or identifier
- `name` (str, optional): Dataset configuration name
- `cache_dir` (str, optional): Override cache directory
- `allow_download` (bool, optional): Explicitly allow/prevent downloads
- `**kwargs`: Additional arguments passed to `load_dataset`

**Returns:** Loaded dataset instance

#### `load_pipeline(task, model=None, cache_dir=None, allow_download=None, **kwargs)`

Create a HuggingFace pipeline with cache awareness.

**Parameters:**
- `task` (str): Pipeline task (e.g., "text-generation")
- `model` (str, optional): Model identifier
- `cache_dir` (str, optional): Override cache directory
- `allow_download` (bool, optional): Explicitly allow/prevent downloads
- `**kwargs`: Additional arguments passed to `pipeline`

**Returns:** Pipeline instance

### Prefetch Functions

#### `prefetch_model(model_id, cache_dir=None)`

Pre-download a model to cache. Run this on a login node before submitting compute jobs.

#### `prefetch_dataset(path, name=None, cache_dir=None, **kwargs)`

Pre-download a dataset to cache. Run this on a login node before submitting compute jobs.

### Storage Path Constants

Use these constants for consistent file organization across projects:

```python
from byutils import AUTODELETE_DIR, ARCHIVE_DIR

# AUTODELETE_DIR = Path.home() / "nobackup" / "autodelete"
# ARCHIVE_DIR = Path.home() / "nobackup" / "archive"

data_dir = AUTODELETE_DIR / "my_project" / "data"
results_dir = ARCHIVE_DIR / "my_project" / "results"
```

### Configuration

#### Default Cache Location

- Default: `~/nobackup/autodelete/hf_cache`
- Models: `~/nobackup/autodelete/hf_cache/models`
- Datasets: `~/nobackup/autodelete/hf_cache/datasets`

#### Override Cache Location

Set the `HF_HOME` environment variable before importing:

```python
import os
os.environ["HF_HOME"] = "/custom/cache/path"

from byutils import load_model
```

Or in your shell:

```bash
export HF_HOME=/custom/cache/path
python my_script.py
```

### Exceptions

#### `BYUtilsError`

Base exception for all byutils errors.

#### `CacheMissError`

Raised when attempting to load a resource that isn't cached while offline. Includes helpful instructions for prefetching.

#### `InternetRequiredError`

Raised when internet is required but unavailable. Suggests retrying on a login node.

## Usage Patterns

### Pattern 1: Simple Auto-Detection (Recommended)

Best for interactive development and simple scripts:

```python
from byutils import load_model, load_tokenizer

# Automatically detects login vs compute node
model = load_model("gpt2")
tokenizer = load_tokenizer("gpt2")
```

### Pattern 2: Explicit Control

When you need fine-grained control:

```python
from byutils import load_model

# Force download (useful on login node)
model = load_model("gpt2", allow_download=True)

# Force cache-only (useful for testing compute node behavior)
model = load_model("gpt2", allow_download=False)
```

### Pattern 3: SLURM Job Workflow

For production compute jobs:

```bash
# 1. Create prefetch script (prefetch.py)
cat > prefetch.py << 'EOF'
from byutils import prefetch_model, prefetch_dataset

prefetch_model("gpt2")
prefetch_dataset("squad")
EOF

# 2. Run prefetch on login node
python prefetch.py

# 3. Submit job that uses cached resources
sbatch job.sh
```

```python
# job.sh runs:
from byutils import load_model, load_dataset

model = load_model("gpt2")  # Uses cache
dataset = load_dataset("squad")  # Uses cache
```

### Pattern 4: Migration from Existing Code

**Before:**
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

**After:**
```python
from byutils import load_model, load_tokenizer

model = load_model("gpt2")
tokenizer = load_tokenizer("gpt2")
```

## How It Works

### Internet Detection

BYUtils uses a three-tier detection strategy:

1. **Hostname check** (primary): Login nodes contain "login" in hostname
2. **Explicit override**: User-provided `allow_download` parameter
3. **Connectivity test** (fallback): HTTP request to HuggingFace

### Cache Management

- All resources stored in `~/nobackup/autodelete/hf_cache` by default
- Subdirectories for models and datasets
- Automatically sets HuggingFace environment variables
- Files in autodelete are automatically deleted after 12 weeks

### Loading Behavior

**On login nodes:**
- Downloads resources if not cached
- Uses cache if available
- Full internet access

**On compute nodes:**
- Uses cache exclusively
- No internet requests
- Raises `CacheMissError` if resource not cached

## Troubleshooting

### CacheMissError

If you see:
```
CacheMissError: Cache miss: 'model-name' not found in /home/<netid>/nobackup/autodelete/hf_cache/models
```

**Solution:** Run prefetch on a login node:
```python
from byutils import prefetch_model
prefetch_model("model-name")
```

### Import Errors

If you see dependency errors, install requirements:
```bash
pip install transformers datasets huggingface-hub
```

### Cache Full

If your autodelete directory is full, clean old models:
```bash
# Check cache size
du -sh ~/nobackup/autodelete/hf_cache

# Remove old cache (be careful!)
rm -rf ~/nobackup/autodelete/hf_cache/old_models
```

## Development

### Running Tests

```bash
# Simple test
python -c "from byutils import load_model; print(load_model('gpt2'))"

# Test prefetch
python -c "from byutils import prefetch_model; prefetch_model('gpt2')"
```

## License

Internal use for BYU ORC projects.

## Related Resources

- [BYU ORC Documentation](https://rc.byu.edu)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
