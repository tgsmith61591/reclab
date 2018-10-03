"""
__init__ script for model_selection. In this file, you should include import
statements from files within the submodule. I.e., if your module resembles
the following:

  my_submodule/
    |_ a.py
    |_ b.py

Your __init__.py would include:
from .a import *
from .b import *
"""

from ._split import *

__all__ = [s for s in dir() if not s.startswith("_")]
