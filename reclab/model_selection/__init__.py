"""
The ``reclab.model_selection`` submodule provides many utilities for cross-
validating your recommender models, splitting your data into train/test splits
and performing grid searches.
"""

from ._split import check_cv
from ._split import train_test_split
from ._split import KFold
from ._search import RecommenderGridSearchCV
from ._search import RandomizedRecommenderSearchCV

__all__ = [s for s in dir() if not s.startswith("_")]
