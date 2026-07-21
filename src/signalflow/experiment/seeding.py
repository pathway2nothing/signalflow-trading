"""Deterministic seeding for research runs."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed os hash, random, and numpy; returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
