"""
Reproducibility Utilities
=========================

Utilities for ensuring reproducible training runs.

These are the ONLY reproducibility utilities needed.
All other training logic is in engine.py.

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Union

import numpy as np

# SSOT import for hash utilities
from src.utils.hash_utils import compute_file_hash as _compute_file_hash_ssot

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy random generator
    - PyTorch (if available)

    Args:
        seed: Random seed to use
    """
    # Python random
    random.seed(seed)
    logger.debug(f"Set random.seed({seed})")

    # NumPy
    np.random.seed(seed)
    logger.debug(f"Set numpy.random.seed({seed})")

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        logger.debug(f"Set torch.manual_seed({seed})")

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("Set CUDA deterministic mode")

    except ImportError:
        logger.debug("PyTorch not available, skipping torch seed")

    logger.info(f"Reproducible seeds set: {seed}")


def compute_file_hash(path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of a file.

    SSOT: Delegates to src.utils.hash_utils

    Args:
        path: Path to file
        chunk_size: Chunk size for reading

    Returns:
        SHA256 hexdigest

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    return _compute_file_hash_ssot(path, chunk_size=chunk_size).full_hash


def compute_json_hash(path: Union[str, Path]) -> str:
    """
    Compute hash of JSON file (canonical form).

    Args:
        path: Path to JSON file

    Returns:
        SHA256 hexdigest
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "set_reproducible_seeds",
    "compute_file_hash",
    "compute_json_hash",
]
