# -*- coding: utf-8 -*-
"""
The ``reclab.datasets`` submodule provides several different ratings datasets
used in various examples and tests across the package. If you would like to
prototype a model, this is a good place to find easy-to-access data.
"""

from .base import *

__all__ = [s for s in dir() if not s.startswith("_")]
