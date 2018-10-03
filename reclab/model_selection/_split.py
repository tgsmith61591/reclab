# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.externals import six
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator

from ..utils.validation import to_sparse_csr, check_consistent_length

import numbers

__all__ = [
    'BootstrapCV',
    'check_cv',
    'train_test_split'
]

MAX_SEED = 1e6


def check_cv(cv=3):
    """Input validation for cross-validation classes.

    Parameters
    ----------
    cv : int, None or BaseCrossValidator
        The CV class or number of folds.

        - None will default to 3-fold BootstrapCV
        - integer will default to ``integer``-fold BootstrapCV
        - BaseCrossValidator will pass through untouched

    Returns
    -------
    checked_cv : BaseCrossValidator
        The validated CV class
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        return BootstrapCV(n_splits=int(cv))
    if not hasattr(cv, "split") or isinstance(cv, six.string_types):
        raise ValueError("Expected integer or CV class, but got %r (type=%s)"
                         % (cv, type(cv)))
    return cv


def _validate_train_size(train_size):
    """Train size should be a float between 0 and 1."""
    assert isinstance(train_size, float) and (0. < train_size < 1.), \
        "train_size should be a float between 0 and 1"


def _get_stratified_tr_mask(u, i, train_size, random_state):
    _validate_train_size(train_size)  # validate it's a float
    random_state = check_random_state(random_state)
    n_events = u.shape[0]

    # this is our train mask that we'll update over the course of this method
    train_mask = random_state.rand(n_events) <= train_size  # type: np.ndarray

    # we have a random mask now. For each of users and items, determine which
    # are missing from the mask and randomly select one of each of their
    # ratings to force them into the mask
    for array in (u, i):
        # e.g.:
        # >>> array = np.array([1, 2, 3, 3, 1, 3, 2])
        # >>> train_mask = np.array([0, 1, 1, 1, 0, 0, 1]).astype(bool)
        # >>> unique, counts = np.unique(array, return_counts=True)
        # >>> unique, counts
        # (array([1, 2, 3]), array([2, 2, 3]))

        # then present:
        # >>> present
        # array([2, 3, 3, 2])
        present = array[train_mask]

        # and the test indices:
        # >>> test_vals
        # array([1, 1, 3])
        test_vals = array[~train_mask]

        # get the test indices that are NOT present (either
        # missing items or users)
        # >>> missing
        # array([1])
        missing = np.unique(test_vals[np.where(
            ~np.in1d(test_vals, present))[0]])

        # If there is nothing missing, we got perfectly lucky with our random
        # split and we'll just go with it...
        if missing.shape[0] == 0:
            continue

        # Otherwise, if we get to this point, we have to add in the missing
        # level to the mask to make sure at least one of each of those makes
        # it into the training data (so we don't lose a factor level for ALS)
        array_mask_missing = np.in1d(array, missing)

        # indices in "array" where we have a level that's currently missing
        # and that needs to be added into the mask
        where_missing = np.where(array_mask_missing)[0]  # e.g., array([0, 4])

        # I don't love having to loop here... but we'll iterate "where_missing"
        # to incrementally add in items or users until all are represented
        # in the training set to some degree
        added = set()
        for idx, val in zip(where_missing, array[where_missing]):
            # if we've already seen and added this one
            if val in added:  # O(1) lookup
                continue

            train_mask[idx] = True
            added.add(val)

    return train_mask


def _make_sparse_tr_te(users, items, ratings, train_mask):
    # now make the sparse matrices
    r_train = to_sparse_csr(u=users[train_mask], i=items[train_mask],
                            r=ratings[train_mask], axis=0)

    # TODO: anti mask?
    r_test = to_sparse_csr(u=users, i=items, r=ratings, axis=0)
    return r_train, r_test


def train_test_split(u, i, r, train_size=0.75, random_state=None):
    """Create a train/test split for sparse ratings.

    Given vectors of users, items, and ratings, create a train/test split
    that preserves at least one of each user and item in the training split
    to prevent inducing a cold-start situation.

    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        A numpy array of the users. This vector will be used to stratify the
        split to ensure that at least of each of the users will be included
        in the training split. Note that this diminishes the likelihood of a
        perfectly-sized split (i.e., ``len(train)`` may not exactly equal
        ``train_size * n_samples``).

    i : array-like, shape=(n_samples,)
        A numpy array of the items. This vector will be used to stratify the
        split to ensure that at least of each of the items will be included
        in the training split. Note that this diminishes the likelihood of a
        perfectly-sized split (i.e., ``len(train)`` may not exactly equal
        ``train_size * n_samples``).

    r : array-like, shape=(n_samples,)
        A numpy array of the ratings.

    train_size : float, optional (default=0.75)
        The ratio of the train set size. Should be a float between 0 and 1.

    random_state : RandomState, int or None, optional (default=None)
        The random state used to create the train mask.

    Examples
    --------
    An example of a sparse matrix split that masks some ratings from the train
    set, but not from the testing set:

    >>> from reclab.model_selection import train_test_split
    >>> u = [0, 1, 0, 2, 1, 3]
    >>> i = [1, 2, 2, 0, 3, 2]
    >>> r = [0.5, 1.0, 0.0, 1.0, 0.0, 1.]
    >>> train, test = train_test_split(u, i, r, train_size=0.5,
    ...                                random_state=42)
    >>> train.toarray()  # doctest: +SKIP
    array([[ 0. ,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ]], dtype=float32)
    >>> test.toarray()  # doctest: +SKIP
    array([[ 0. ,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ],
           [ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ]], dtype=float32)

    Here's a more robust example (with more ratings):

    >>> from sklearn.preprocessing import LabelEncoder
    >>> import numpy as np
    >>> rs = np.random.RandomState(42)
    >>> users = np.arange(100000)  # 100k users in DB
    >>> items = np.arange(30000)  # 30k items in DB
    >>> # Randomly select some for ratings:
    >>> items = rs.choice(items, users.shape[0])  # 100k rand item rtgs
    >>> users = rs.choice(users, users.shape[0])  # 100k rand user rtgs
    >>> # Label encode so they're positional indices:
    >>> users = LabelEncoder().fit_transform(users)
    >>> items = LabelEncoder().fit_transform(items)
    >>> ratings = rs.choice((0., 0.25, 0.5, 0.75, 1.), items.shape[0])
    >>> train, test = train_test_split(users, items, ratings, random_state=rs)
    >>> train
    <26353x28921 sparse matrix of type '<type 'numpy.float32'>'
        with 77770 stored elements in Compressed Sparse Row format>
    >>> test
    <26353x28921 sparse matrix of type '<type 'numpy.float32'>'
        with 99994 stored elements in Compressed Sparse Row format>

    Notes
    -----
    ``u``, ``i`` inputs should be encoded (i.e., via LabelEncoder) prior to
    splitting the data. This is due to the indexing behavior used within the
    function.

    Returns
    -------
    r_train : scipy.sparse.csr_matrix
        The train set.

    r_test : scipy.sparse.csr_matrix
        The test set.
    """
    # make sure all of them are numpy arrays and of the same length
    users, items, ratings = check_consistent_length(u, i, r)

    train_mask = _get_stratified_tr_mask(
        users, items, train_size=train_size,
        random_state=random_state)

    return _make_sparse_tr_te(users, items, ratings, train_mask=train_mask)


# avoid pb w nose
train_test_split.__test__ = False


class BaseCrossValidator(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for all CV.

    Iterations must define ``_iter_train_mask``
    """
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, X):
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            A sparse ratings matrix.

        Returns
        -------
        train : scipy.sparse.csr_matrix
            The training set

        test : scipy.sparse.csr_matrix
            The test set
        """
        ratings = X.data
        users, items = X.nonzero()

        # make sure all of them are numpy arrays and of the same length
        # users, items, ratings = check_consistent_length(u, i, r)
        for train_mask in self._iter_train_mask(users, items, ratings):
            # yield in a generator so we don't have to store in mem
            yield _make_sparse_tr_te(users, items, ratings,
                                     train_mask=train_mask)

    @abstractmethod
    def _iter_train_mask(self, u, i, r):
        """Compute the training mask here.

        Returns
        -------
        train_mask : np.ndarray
            The train mask
        """


class BootstrapCV(BaseCrossValidator):
    """Cross-validate with bootstrapping.

    The bootstrap CV class makes no guarantees about exclusivity between folds.
    This is simply a naive way to handle KFold cross-validation.

    Examples
    --------
    >>> from reclab.datasets import load_lastfm
    >>> from reclab.model_selection import BootstrapCV
    >>> X = load_lastfm(as_sparse=True)
    >>> cv = BootstrapCV(random_state=42, n_splits=3)  # doctest: +SKIP
    [(<1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 65790 stored elements in Compressed Sparse Row format>,
    <1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 92834 stored elements in Compressed Sparse Row format>),
    (<1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 65740 stored elements in Compressed Sparse Row format>,
    <1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 92834 stored elements in Compressed Sparse Row format>),
    (<1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 65964 stored elements in Compressed Sparse Row format>,
    <1892x17632 sparse matrix of type '<class 'numpy.float64'>'
    with 92834 stored elements in Compressed Sparse Row format>)]
    """
    def _iter_train_mask(self, u, i, r):
        """Compute the training mask here."""
        train_size = 1. - (1. / self.n_splits)
        # train_size = 1. - ((n_samples / self.n_splits) / n_samples)
        random_state = check_random_state(self.random_state)

        for split in range(self.n_splits):
            yield _get_stratified_tr_mask(
                u, i, train_size=train_size,
                random_state=random_state.randint(MAX_SEED))
