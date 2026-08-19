"""Minimal prompt-embedding adapter runtime."""

from .adapter_factory import ADAPTER_TYPES, build_adapter_from_config
from .model import SafeEmbeddingAdapter

__all__ = [
    "ADAPTER_TYPES",
    "SafeEmbeddingAdapter",
    "build_adapter_from_config",
]
