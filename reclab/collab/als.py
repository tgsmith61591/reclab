# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .base import BaseMatrixFactorization
from ..base import BaseRecommender
from ..utils.decorators import inherit_function_doc
from ..utils.validation import check_sparse_array, get_n_factors

from sklearn.utils.validation import check_is_fitted, check_random_state, \
    _is_arraylike

from implicit import als
from implicit import cuda

import numpy as np

__all__ = [
    'AlternatingLeastSquares'
]


class AlternatingLeastSquares(BaseMatrixFactorization):
    r"""Alternating Least Squares

    An implicit collaborative filtering algorithm via matrix factorization
    based off the algorithms described in [1] with performance optimizations
    described in [2].

    Parameters
    ----------
    factors : int or float, optional (default=100)
        The number of latent factors to compute. Generally, fewer factors will
        result in a much faster runtime, but also approximates the
        reconstruction matrix less accurately.

    regularization : float, optional (default=0.01)
        The regularization (L2) parameter to use. The higher this value, the
        more the algorithm will learn to generalize at the cost of training
        reconstruction accuracy.

    use_native : bool, optional (default=True)
        Use native extensions to speed up model fitting.

    use_cg : bool, optional (default=True)
        Use a faster Conjugate Gradient solver to calculate factors. This is
        the method proposed in the paper by Takacs, et. al, "Applications
        of the Conjugate Gradient Method for Implicit Feedback Collaborative
        Filtering"

    use_gpu : bool, optional
        Fit on the GPU if available, default is to run on GPU only if
        available.

    iterations : int, optional (default=15)
        The number of ALS iterations to use when fitting data. The default
        is 25. More iterations will yield a better fit, but will take longer.

    calculate_training_loss : bool, optional (default=False)
        Whether to log out the training loss at each iteration.

    num_threads : int, optional (default=0)
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.

    show_progress : bool, optional (default=True)
        Whether to show a progress bar while training.

    random_state : int, RandomState or None, optional (default=None)
        The random state to control random seeding of the item and user
        factor matrices.

    Notes
    -----
    The negative items are implicitly defined: this algorithm assumes that
    non-zero items in the item_users matrix means that the user liked the item.
    The negatives are left unset in this sparse matrix: the library will assume
    that means Piu = 0 and Ciu = 1 for all these items.

    Examples
    --------
    Fitting an ALS model:

    >>> from reclab.datasets import load_lastfm
    >>> from reclab.model_selection import train_test_split
    >>> lastfm = load_lastfm(cache=True)
    >>> train, test = train_test_split(u=lastfm.users, i=lastfm.products,
    ...                                r=lastfm.ratings, random_state=42)
    >>> model = AlternatingLeastSquares(random_state=1, show_progress=False)
    >>> model.fit(train)
    AlternatingLeastSquares(calculate_training_loss=False, factors=100,
                iterations=15, num_threads=0, random_state=1,
                regularization=0.01, show_progress=False, use_cg=True,
                use_gpu=False, use_native=True)

    Inference for a given user:

    >>> model.recommend_for_user(0, test, n=5)  # doctest: +SKIP
    array([ 149, 2504,  153,  221, 1064])

    References
    ----------
    .. [1] Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for
           Implicit Feedback Datasets" (http://yifanhu.net/PUB/cf.pdf)

    .. [2] G. Takacs, I. Pilaszy, D. Tikk, "Applications of the Conjugate
           Gradient Method for Implicit Feedback Collaborative Filtering"
    """
    def __init__(self, factors=100, regularization=0.01,
                 use_native=True, use_cg=True, iterations=15,
                 use_gpu=cuda.HAS_CUDA, calculate_training_loss=False,
                 num_threads=0, show_progress=True, random_state=None):

        self.factors = factors
        self.regularization = regularization
        self.use_native = use_native
        self.use_cg = use_cg
        self.use_gpu = use_gpu
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads
        self.show_progress = show_progress
        self.random_state = random_state

    def _make_estimator(self, X):
        # Get the number of factors
        factors = get_n_factors(X.shape[1], self.factors)
        n_users, n_items = X.shape

        # Implicit really only likes numpy float32 for some reason
        dtype = np.float32
        est = als.AlternatingLeastSquares(
            factors=factors, regularization=self.regularization,
            dtype=dtype, use_native=self.use_native, use_cg=self.use_cg,
            use_gpu=self.use_gpu, iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            num_threads=self.num_threads)

        # initialize the factor matrices
        random_state = check_random_state(self.random_state)
        return self._initialize_factors(
            estimator=est, n_users=n_users, n_items=n_items, factors=factors,
            dtype=dtype, random_state=random_state)

    @inherit_function_doc(BaseMatrixFactorization)
    def fit(self, X):
        # Now validate that X is a sparse array. Implicit forces float32, so
        # better to check it now...
        X = check_sparse_array(X, dtype=np.float32, copy=False,
                               force_all_finite=True)

        # Fit the estimator (Implicit likes it in item-user major order...)
        self.estimator_ = est = self._make_estimator(X)
        est.fit(X.T, show_progress=self.show_progress)

        return self

    @inherit_function_doc(BaseMatrixFactorization)
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False,
                           recalculate_user=False):
        # Make sure the model has been fitted!
        check_is_fitted(self, "estimator_")

        # Now validate the ratings matrix
        R = check_sparse_array(R, dtype=np.float32, copy=False,
                               force_all_finite=True)

        est = self.estimator_
        user = est._user_factor(userid, R, recalculate_user)
        item_factors = np.array(est.item_factors)  # this is a copy

        # The scores are just the product of the user array and the items
        # factors
        scores = item_factors.dot(user)  # np.ndarray, shape=(n_items,)
        itms = np.arange(scores.shape[0])
        keep_mask = np.ones(scores.shape[0], dtype=bool)

        # if we're filtering any items at all, do so now...
        if filter_items is not None:
            assert _is_arraylike(filter_items)
            filter_items = np.asarray(filter_items)
            keep_mask[filter_items] = False

        # remove rated indices if needed
        if filter_previously_rated:
            rated_items = R[userid, :].indices
            keep_mask[rated_items] = False

        # Now filter out where necessary...
        scores = scores[keep_mask]
        itms = itms[keep_mask]

        # get the argsort order (negative for desc)
        order = np.argsort(-scores)
        scores = scores[order]
        itms = itms[order]

        if return_scores:
            return itms[:n], scores[:n]
        return itms[:n]
