"""
The ``reclab.collab`` sub-module defines a number of collaborative filtering
recommender algorithms, including popular matrix factorization techniques and
nearest neighbor methods.
"""

# Import here:
from .als import *
from .approximate import *
from .base import *
from .neighbors import *

__all__ = [s for s in dir() if not s.startswith("_")]
