"""
create synthetic data
"""

# load packages
import os
import numpy as np


def create_gaussian_synthetic(n: int, p: int) -> np.ndarray:
    """create a standard gaussian synthetic datatset"""
    return np.random.normal(0, 1, (n, p))
