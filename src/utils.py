"""Utility functions for rknn-converter."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_binary(path: Union[str, Path], dtype: np.dtype = np.float32) -> np.ndarray:
    """Load binary file as numpy array."""
    return np.fromfile(path, dtype=dtype)


def save_binary(data: np.ndarray, path: Union[str, Path]) -> None:
    """Save numpy array to binary file."""
    path = Path(path)
    ensure_dir(path.parent)
    data.tofile(path)


def compute_file_hash(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def intersperse(lst: List[Any], item: Any) -> List[Any]:
    """Intersperse an item between list elements."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def split_sentences(text: str, language: str) -> List[str]:
    """Split text into sentences based on language."""
    if language.lower() in ["en", "english"]:
        pattern = r"(?<=[.!?])\s+"
    elif language.lower() in ["zh", "chinese"]:
        pattern = r"(?<=[。！？；;])"
    else:
        pattern = r"(?<=[.!?。！？；;])\s*"

    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def validate_rknn_installation() -> bool:
    """Check if RKNN-Toolkit2 is installed."""
    try:
        import rknn.api
        return True
    except ImportError:
        return False


def validate_onnx_version() -> bool:
    """Check if ONNX version is compatible."""
    try:
        import onnx
        from packaging.version import Version

        return Version(onnx.__version__) == Version("1.18.0")
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cpu": True,
        "cuda": False,
        "mps": False,
        "rknn": False,
    }

    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["mps"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    info["rknn"] = validate_rknn_installation()

    return info
