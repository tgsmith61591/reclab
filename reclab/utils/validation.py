# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils import validation as skval
from sklearn.externals import six

from scipy import sparse
import numpy as np

__all__ = [
    'check_consistent_length',
    'is_iterable',
    'to_sparse_csr'
]

ITYPE = np.int32
DTYPE = np.float64  # implicit asks for doubles, not float32s...


def check_consistent_length(u, i, r):
    """Ensure users, items, and ratings are all of the same dimension.

    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        A numpy array of the users.

    i : array-like, shape=(n_samples,)
        A numpy array of the items.

    r : array-like, shape=(n_samples,)
        A numpy array of the ratings.
    """
    skval.check_consistent_length(u, i, r)
    return np.asarray(u), np.asarray(i), np.asarray(r, dtype=DTYPE)


def _make_sparse_csr(data, rows, cols, dtype=DTYPE):
    # check lengths
    check_consistent_length(data, rows, cols)
    data, rows, cols = (np.asarray(x) for x in (data, rows, cols))

    shape = (np.unique(rows).shape[0], np.unique(cols).shape[0])
    return sparse.csr_matrix((data, (rows, cols)),
                             shape=shape, dtype=dtype)


def is_iterable(x):
    """Determine whether an element is iterable.

    This function determines whether an element is iterable by checking
    for the ``__iter__`` attribute. Since Python 3.x adds the ``__iter__``
    attribute to strings, we also have to make sure the input is not a
    string or unicode type.

    Parameters
    ----------
    x : object
        The object or primitive to test whether
        or not is an iterable.
    """
    if isinstance(x, six.string_types):
        return False
    return hasattr(x, '__iter__')


def to_sparse_csr(u, i, r, axis=0, dtype=DTYPE):
    """Create a sparse ratings matrix.

    Create a sparse ratings matrix with users and items as rows and columns,
    and ratings as the values.

    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        The user vector. Positioned along the row axis if ``axis=0``,
        otherwise positioned along the column axis.

    i : array-like, shape=(n_samples,)
        The item vector. Positioned along the column axis if ``axis=0``,
        otherwise positioned along the row axis.

    r : array-like, shape=(n_samples,)
        The ratings vector.

    axis : int, optional (default=0)
        The axis along which to position the users. If 0, the users are
        along the rows (with items as columns). If 1, the users are columns
        with items as rows.

    dtype : type, optional (default=np.float32)
        The type of the values in the ratings matrix.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be an int in (0, 1)")

    rows = u if axis == 0 else i
    cols = i if axis == 0 else u
    return _make_sparse_csr(data=r, rows=rows, cols=cols, dtype=dtype)
