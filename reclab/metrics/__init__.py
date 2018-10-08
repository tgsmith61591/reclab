"""
The ``reclab.metrics`` submodule provides several different rankings metrics
that are widely used for benchmarking the efficacy of a recommender algorithm.
"""

# Import here:
from .ranking import *

__all__ = [s for s in dir() if not s.startswith("_")]
