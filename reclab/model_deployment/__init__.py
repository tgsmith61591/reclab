"""
The ``reclab.model_deployment`` submodule provides tools for serving your
selected model and producing recommendations all within a wrapper, including:

* Automatically encoding/inverse-encoding user and product IDs
* Producing recommendations from JSON or array (depending on your service
  architecture)
"""

from .wrapper import *

__all__ = [s for s in dir() if not s.startswith("_")]
