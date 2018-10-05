# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os

__all__ = [
    'safe_mkdirs'
]


def safe_mkdirs(directory):
    """Make an entire directory tree.

    Creates an entire directory tree (even if its parents don't exist)
    without failing if it already exists. Also avoids the race condition
    of checking first.

    Parameters
    ----------
    directory : str
        The absolute path of the directory to create.
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != 17:
            raise
