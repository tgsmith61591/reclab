# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
from os.path import expanduser

import numpy as np

# The directory in which we'll store NMS & Annoy model indices for pickling
RECLAB_CACHE = os.environ.get('RECLAB_CACHE',
                              expanduser('~/.reclab-cache'))


def set_blas_singlethread():
    """Set BLAS internal threading to be single-threaded.

    Checks to see if using OpenBlas/Intel MKL, and set the appropriate num
    threads to 1 (causes severe perf issues when training - can be 10x slower)
    """
    if np.__config__.get_info('openblas_info') and os.environ.get(
            'OPENBLAS_NUM_THREADS') != '1':
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if np.__config__.get_info('blas_mkl_info') and os.environ.get(
            'MKL_NUM_THREADS') != '1':
        os.environ["MKL_NUM_THREADS"] = "1"
