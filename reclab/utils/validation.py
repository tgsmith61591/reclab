# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.utils import validation as skval
from sklearn.externals import six

from scipy import sparse
import numpy as np

__all__ = [
    'check_consistent_length',
    'check_permitted_value',
    'check_sparse_array',
    'get_n_factors',
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


def check_permitted_value(permitted_dict, provided_key):
    """Get a key/value mapping from a dict of permitted value.

    If the key is not found in the dictionary, raise a KeyError with a more
    descriptive message of the error.

    Parameters
    ----------
    permitted_dict : dict
        A dictionary of valid key/value mappings

    provided_key : str, int
        The key provided by the user
    """
    try:
        return permitted_dict[provided_key]
    except KeyError:
        raise KeyError("%r is not a valid key. Must be one of %s"
                       % (provided_key, str(list(permitted_dict.keys()))))


def check_sparse_array(array, dtype="numeric", order=None, copy=False,
                       force_all_finite=True, ensure_2d=True, allow_nd=False,
                       ensure_min_samples=1, ensure_min_features=1,
                       warn_on_dtype=False, estimator=None):
    """Input validation on a sparse matrix.

    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.
    """
    if not sparse.issparse(array):
        raise ValueError("Expected sparse CSR, but got %s" % type(array))

    return skval.check_array(
        array=array, accept_sparse='csr', dtype=dtype, order=order, copy=copy,
        force_all_finite=force_all_finite, ensure_2d=ensure_2d,
        allow_nd=allow_nd, ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features, warn_on_dtype=warn_on_dtype,
        estimator=estimator)


def get_n_factors(n_dim, n_factors):
    """Get the number of factors to compute.

    Lots of recommendation algorithms compute a number of latent factors.
    This method computes the number of factors as either a ratio or returns
    the given integer.

    Parameters
    ----------
    n_dim : int
        The dimensions in the input space.

    n_factors : int or float
        The number of factors or the ratio of factors to compute. If a float,
        must be > 0.
    """
    if isinstance(n_factors, (int, float)):
        # If it's <= 0, raise
        if n_factors <= 0:
            raise ValueError("n_factors must be > 0")
        if isinstance(n_factors, int):
            return n_factors
        if isinstance(n_factors, float):
            return np.ceil(n_dim * n_factors).astype(int)
    raise TypeError("n_factors must be a float or an int, but got type=%s"
                    % type(n_factors))


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
