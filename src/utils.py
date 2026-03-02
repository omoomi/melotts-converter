"""Utility functions for melotts-converter.

This module provides common utility functions for file I/O, data processing,
formatting, and validation used throughout the codebase.

Example:
    >>> from utils import ensure_dir, load_json, save_json
    >>> dir_path = ensure_dir('models/en')
    >>> data = {'key': 'value'}
    >>> save_json(data, dir_path / 'config.json')
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path as string or Path object.

    Returns:
        Path object for the ensured directory.

    Example:
        >>> from utils import ensure_dir
        >>> dir_path = ensure_dir('models/en/voices')
        >>> dir_path.exists()
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON data from a file.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Example:
        >>> data = load_json('models/en/config.json')
        >>> data['sample_rate']
        44100
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to a JSON file.

    Args:
        data: Dictionary to be saved as JSON.
        path: Output file path. Parent directories will be created if needed.

    Example:
        >>> save_json({'name': 'test', 'value': 123}, 'output/config.json')
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_binary(path: Union[str, Path], dtype: np.dtype = np.float32) -> np.ndarray:
    """Load binary data as numpy array.

    Args:
        path: Path to binary file.
        dtype: NumPy data type for the array (default: float32).

    Returns:
        NumPy array containing the loaded data.

    Example:
        >>> embedding = load_binary('models/en/voices/en-us.bin')
        >>> embedding.shape
        (256,)
    """
    return np.fromfile(path, dtype=dtype)


def save_binary(data: np.ndarray, path: Union[str, Path]) -> None:
    """Save numpy array to binary file.

    Args:
        data: NumPy array to save.
        path: Output file path. Parent directories will be created if needed.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(256).astype(np.float32)
        >>> save_binary(data, 'models/en/voices/test.bin')
    """
    path = Path(path)
    ensure_dir(path.parent)
    data.tofile(path)


def compute_file_hash(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file.

    Args:
        path: Path to file to hash.
        algorithm: Hash algorithm to use (default: 'sha256').
            Supported: 'md5', 'sha1', 'sha256', 'sha512', etc.

    Returns:
        Hexadecimal string representation of the file hash.

    Example:
        >>> hash_value = compute_file_hash('models/en/checkpoint.pth')
        >>> len(hash_value)
        64
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def intersperse(lst: List[Any], item: Any) -> List[Any]:
    """Intersperse an item between list elements.

    Creates a new list where the item is placed between each element of
    the original list, and also at the beginning and end.

    Args:
        lst: Input list to intersperse.
        item: Item to insert between elements.

    Returns:
        New list with item interspersed.

    Example:
        >>> intersperse([1, 2, 3], 0)
        [0, 1, 0, 2, 0, 3, 0]
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def split_sentences(text: str, language: str) -> List[str]:
    """Split text into sentences based on language-specific punctuation.

    Args:
        text: Input text to split.
        language: Language code ('en', 'zh', etc.).

    Returns:
        List of non-empty sentence strings.

    Example:
        >>> split_sentences("Hello world. How are you?", "en")
        ['Hello world.', 'How are you?']
        >>> split_sentences("你好。你好吗？", "zh")
        ['你好。', '你好吗？']
    """
    if language.lower() in ["en", "english"]:
        pattern = r"(?<=[.!?])\s+"
    elif language.lower() in ["zh", "chinese"]:
        pattern = r"(?<=[。！？；;])"
    else:
        pattern = r"(?<=[.!?。！？；;])\s*"

    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Strips leading/trailing whitespace and normalizes internal whitespace
    to single spaces.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text string.

    Example:
        >>> clean_text("  Hello    world  ")
        'Hello world'
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string.

    Args:
        seconds: Time value in seconds.

    Returns:
        Formatted time string (e.g., '150.5ms', '2.34s', '1m 30.5s').

    Example:
        >>> format_time(0.5)
        '500.0ms'
        >>> format_time(90.5)
        '1m 30.5s'
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable string.

    Args:
        bytes_size: Size in bytes.

    Returns:
        Formatted size string (e.g., '1.5KB', '2.3MB', '1.2GB').

    Example:
        >>> format_size(1536)
        '1.5KB'
        >>> format_size(2500000000)
        '2.3GB'
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def validate_rknn_installation() -> bool:
    """Check if RKNN-Toolkit2 is properly installed.

    Returns:
        True if RKNN-Toolkit2 can be imported, False otherwise.

    Example:
        >>> validate_rknn_installation()
        True
    """
    try:
        import rknn.api
        return True
    except ImportError:
        return False


def validate_onnx_version() -> bool:
    """Check if installed ONNX version is compatible.

    Returns:
        True if ONNX version is exactly 1.18.0, False otherwise.

    Example:
        >>> validate_onnx_version()
        True
    """
    try:
        import onnx
        from packaging.version import Version

        return Version(onnx.__version__) == Version("1.18.0")
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.

    Checks for CPU, CUDA (NVIDIA), MPS (Apple Silicon), and RKNN (Rockchip)
    availability.

    Returns:
        Dictionary mapping device names to availability status.

    Example:
        >>> info = get_device_info()
        >>> info['cpu']
        True
        >>> info['cuda']
        False
    """
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
