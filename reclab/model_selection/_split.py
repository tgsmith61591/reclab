# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.externals import six
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator

from ..utils.validation import check_sparse_array

import numbers

__all__ = [
    'check_cv',
    'train_test_split',
    'KFold'
]

MAX_SEED = 1e6


def check_cv(cv=3):
    """Validate the CV input.

    Input validation for cross-validation classes. Takes a CV value of either
    an integer, None or BaseCrossValidator and returns an appropriate CV
    method. For integers or None, default is KFold.

    Parameters
    ----------
    cv : int, None or BaseCrossValidator
        The CV class or number of folds.

        - None will default to 3-fold KFold
        - integer will default to ``integer``-fold KFold
        - BaseCrossValidator will pass through untouched

    Returns
    -------
    checked_cv : BaseCrossValidator
        The validated CV class
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=int(cv))
    if not hasattr(cv, "split") or isinstance(cv, six.string_types):
        raise ValueError("Expected integer or CV class, but got %r (type=%s)"
                         % (cv, type(cv)))
    return cv


def _validate_train_size(train_size):
    """Train size should be a float between 0 and 1."""
    assert isinstance(train_size, float) and (0. < train_size < 1.), \
        "train_size should be a float between 0 and 1"


def _sparsify_from_mask(X, mask):
    """Make a sparse array even sparser.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        Sparse matrix

    mask : scipy.sparse.csr_matrix, bool
        Boolean mask expression
    """
    S = X.copy()
    S.data[mask] = 0.
    S.eliminate_zeros()
    return S


def _split_between_values(X, permuted, low, high):
    """Mask a matrix between a low and a high value.

    Split a matrix around a mask such that values between low and high
    (high > values > low) are split into their own mask.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        The sparse matrix to split

    permuted : np.ndarray
        Permuted linspace of values between 0 and 1

    low : float
        Low value >= 0., < high

    high : float
        High value <= 1., > low
    """
    gt_low = permuted >= low
    lt_high = permuted < high
    mask = gt_low & lt_high

    # Get the sparsified matrices (mask gets zeroed, so use inverse)
    in_range = _sparsify_from_mask(X, ~mask)
    outside_range = _sparsify_from_mask(X, mask)
    return in_range, outside_range


def _get_train_mask_linspace(n, random_state, shuffle):
    # This is the train mask that we'll update over the course of this method.
    # Use a linspace so we're guaranteed an (almost) perfect number of samples,
    # whereas with random.rand, we can only approximate. +1 for linspace N
    # since it goes to 1 inclusive, and trim off the last one.
    values = np.linspace(0, 1, n + 1)[:-1]
    if shuffle:
        values = random_state.permutation(values)
    return values


def train_test_split(X, train_size=0.75, random_state=None):
    """Create a train/test split for sparse ratings matrices.

    Given vectors of users, items, and ratings, create a train/test split
    that masks users' ratings to the prescribed ratio of training samples.
    This only works on sparse matrices.

    See :ref:`train_test` for more information.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix
        The sparse ratings matrix with users along the row axis and items
        along the column axis. Entries represent ratings or other implicit
        ranking events (i.e., number of listens, etc.).

    train_size : float, optional (default=0.75)
        The ratio of the train set size. Should be a float between 0 and 1.

    random_state : RandomState, int or None, optional (default=None)
        The random state used to seed the train and test masks.

    Examples
    --------
    An example of a sparse matrix split that masks some ratings from the train
    set, and leaves them out for the test set.

    >>> from reclab.model_selection import train_test_split
    >>> from scipy import sparse
    >>> import numpy as np
    >>> u = [0, 1, 0, 2, 1, 3]
    >>> i = [1, 2, 2, 0, 3, 2]
    >>> r = [0.5, 1.0, 0.0, 1.0, 0.0, 1.]
    >>> X = sparse.csr_matrix((r, (u, i)), shape=(4, 4))
    >>> train, test = train_test_split(X, train_size=0.7, random_state=42)

    The output of the train/test split is two sparse matrices:

    >>> train.astype(np.float32)
    <4x4 sparse matrix of type '<class 'numpy.float32'>'
            with 3 stored elements in Compressed Sparse Row format>
    >>> test.astype(np.float32)
    <4x4 sparse matrix of type '<class 'numpy.float32'>'
            with 1 stored elements in Compressed Sparse Row format>

    When expanded, it's more clear what was stored and what was masked:

    >>> train.toarray().astype(np.float32)
    array([[0. , 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [1. , 0. , 0. , 0. ],
           [0. , 0. , 1. , 0. ]], dtype=float32)
    >>> test.toarray().astype(np.float32)
    array([[0., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)

    A more elaborate example:

    >>> from reclab.datasets import load_lastfm
    >>> lastfm = load_lastfm(as_sparse=True, cache=True)
    >>> train, test = train_test_split(lastfm.ratings, train_size=0.75,
    ...                                random_state=1)
    >>> train.astype(np.float32)
    <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 69655 stored elements in Compressed Sparse Row format>
    >>> test.astype(np.float32)
    <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 23179 stored elements in Compressed Sparse Row format>

    Notes
    -----
    Users (rows) and items (column) must be encoded (i.e., via LabelEncoder)
    prior to creating a sparse matrix of the elements. Since this is required
    prior to train/test splitting, it's recommended that you store the item
    and user encoders for later decoding.

    Returns
    -------
    r_train : scipy.sparse.csr_matrix
        The train set.

    r_test : scipy.sparse.csr_matrix
        The test set.
    """
    X = check_sparse_array(X)
    _validate_train_size(train_size)  # validate it's a float
    random_state = check_random_state(random_state)  # get the random state

    # Create a mask of random values in the shape of the sparse array that we
    # can use for easy masking
    nnz = X.nnz
    rand_vals = _get_train_mask_linspace(nnz, random_state, shuffle=True)

    # Get the mask from the random training values. For train/test split, we
    # can just use train_size as low
    test, train = _split_between_values(
        X, rand_vals, low=train_size, high=1.)

    # Anything over 0 constitutes our mask
    return train, test


# avoid pb w nose
train_test_split.__test__ = False


class BaseCrossValidator(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for all CV

    Parameters
    ----------
    n_splits : int, optional (default=3)
        The number of splits for the cross-validation procedure. Default
        is 3.

    random_state : RandomState, int or None, optional (default=None)
        The random state used to seed the train and test masks.
    """
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

        # Fail out for an illegal n_splits
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

    def get_n_splits(self):
        return self.n_splits

    @abstractmethod
    def split(self, X):
        """Split a dataset into ``n_splits`` folds.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            A sparse ratings matrix with users along the rows, and products
            or items along the column axis.

        Returns
        -------
        train : scipy.sparse.csr_matrix
            The training set

        test : scipy.sparse.csr_matrix
            The validation fold
        """


class KFold(BaseCrossValidator):
    """K-fold cross validation.

    Applies K-fold cross validation with no stratification to mask a subset
    of ratings events from a sparse ratings matrix.

    Parameters
    ----------
    n_splits : int, optional (default=3)
        The number of splits for the cross-validation procedure. Default
        is 3.

    random_state : RandomState, int or None, optional (default=None)
        The random state used to seed the train and test masks.
    
    shuffle : bool, optional (default=True)
        Whether to shuffle the train/test ratings events. Default is True.
        Using False is discouraged, as it is more likely to completely mask out
        users.
        
    Examples
    --------
    >>> from reclab.datasets import load_lastfm
    >>> lfm = load_lastfm(as_sparse=True, cache=True)
    >>> cv = KFold(n_splits=3, random_state=42, shuffle=True)
    >>> splits = list(cv.split(lfm.ratings))
    >>> splits  # doctest: +SKIP
    [(<1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 61889 stored elements in Compressed Sparse Row format>,
      <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 30945 stored elements in Compressed Sparse Row format>),
     (<1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 61889 stored elements in Compressed Sparse Row format>,
      <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 30945 stored elements in Compressed Sparse Row format>),
     (<1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 61890 stored elements in Compressed Sparse Row format>,
      <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
            with 30944 stored elements in Compressed Sparse Row format>)]
    """
    def __init__(self, n_splits=3, random_state=None, shuffle=True):
        super(KFold, self).__init__(
            n_splits=n_splits, random_state=random_state)

        self.shuffle = shuffle

    def split(self, X):
        """Split a dataset into ``n_splits`` folds.

        Splits a ratings matrix using K-fold cross validation with no
        stratification. Train and val folds each share the same dimensions
        (rows x cols) but mask elements out of the train set for collaborative
        filtering.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            A sparse ratings matrix with users along the rows, and products
            or items along the column axis.

        Returns
        -------
        train : scipy.sparse.csr_matrix
            The training set

        test : scipy.sparse.csr_matrix
            The validation fold
        """
        # Make sure it's a sparse array...
        X = check_sparse_array(X)

        # Use np.linspace to evenly partition the space between 0 and 1 into
        # k + 1 pieces so we can use them as "training_sizes"
        train_sizes = np.linspace(0, 1, self.n_splits + 1)

        # We use a series of "permuted values" to mask out the training/testing
        # folds.
        random_state = check_random_state(self.random_state)
        values = _get_train_mask_linspace(X.nnz, random_state,
                                          shuffle=self.shuffle)

        # Iterate the fold space bounds in a generator, returning train/test
        for lower, upper in zip(train_sizes[:-1], train_sizes[1:]):
            test, train = _split_between_values(X, values, lower, upper)
            yield train, test
