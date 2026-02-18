"""Test script for byutils library."""

import sys

from byutils import (
    load_model,
    load_tokenizer,
    Config,
    AUTODELETE_DIR,
    ARCHIVE_DIR,
)
from byutils._connectivity import is_login_node
from transformers import AutoModelForCausalLM

print("=" * 60)
print("BYUtils Test Script")
print("=" * 60)

# Test configuration
print(f"\nConfiguration:")
print(f"  Cache directory: {Config.get_cache_dir()}")
print(f"  Models cache: {Config.get_models_cache()}")
print(f"  Datasets cache: {Config.get_datasets_cache()}")
print(f"  On login node: {is_login_node()}")

# Test storage paths
print(f"\nStorage Paths:")
print(f"  AUTODELETE_DIR: {AUTODELETE_DIR}")
print(f"  ARCHIVE_DIR: {ARCHIVE_DIR}")

# Test with gpt2 (small model)
print(f"\nTesting with gpt2 model...")
print("  Loading tokenizer...")
tokenizer = load_tokenizer("gpt2")
print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

print("  Loading model...")
model = load_model("gpt2", model_class=AutoModelForCausalLM)
print(f"  ✓ Model loaded (type: {type(model).__name__})")

# Test inference
print("\nTesting inference...")
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
result = tokenizer.decode(outputs[0])
print(f"  Input: 'Hello, world!'")
print(f"  Output: '{result}'")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
