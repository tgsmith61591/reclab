# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .als import AlternatingLeastSquares
from ..utils.decorators import inherit_function_doc
from ..utils.validation import check_sparse_array, get_n_factors

import numpy as np
from sklearn.utils.validation import check_random_state

from implicit import approximate_als as aals
from implicit import cuda

from annoy import AnnoyIndex
import nmslib

__all__ = [
    'AnnoyAlternatingLeastSquares',
    'NMSAlternatingLeastSquares'
]


class _BaseApproximateALS(AlternatingLeastSquares):
    """Base class for approximated ALS variants"""


class AnnoyAlternatingLeastSquares(_BaseApproximateALS):
    pass


class NMSAlternatingLeastSquares(_BaseApproximateALS):
    """(Faster) Alternating Least Squares.

    Speeds up :class:`reclab.collab.AlternatingLeastSquares` by using
    `NMSLib <https://github.com/searchivarius/nmslib>`_ to create approximate
    nearest neighbors indices of the latent factors.

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
        Use a faster Conjugate Gradient solver to calculate factors

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

    method : str, optional (default='hnsw')
        The NMSLib method to use. Default is "hnsw" (Hierarchical Navigable
        Small World). For a list of exhaustive methods, see [1]

    index_params: dict, optional (default=None)
        Optional params to send to the createIndex call in NMSLib

    query_params: dict, optional (default=None)
        Optional query time params for the NMSLib ``setQueryTimeParams`` call

    approximate_similar_items : bool, optional (default=True)
        Whether or not to build an NMSLIB index for computing similar_items.
        Default (recommended) is True.

    approximate_recommend : bool, optional (default=True)
        Whether or not to build an NMSLIB index for the recommend call.
        Default (recommended) is True.

    random_state : int, RandomState or None, optional (default=None)
        The random state to control random seeding of the item and user
        factor matrices.

    show_progress : bool, optional (default=True)
        Whether to show a progress bar while training.

    Attributes
    ----------
    estimator_ : implicit Recommender
        After the model is fit, the ``estimator_`` attribute will hold the fit
        ALS model.

    References
    ----------
    .. [1] Valid NMSLIB method source code
           https://github.com/nmslib/nmslib/tree/master/similarity_search/include/method
    """
    def __init__(self, factors=100, regularization=0.01,
                 use_native=True, use_cg=True, iterations=15,
                 use_gpu=cuda.HAS_CUDA, calculate_training_loss=False,
                 num_threads=0, method='hnsw', index_params=None,
                 query_params=None, approximate_similar_items=True,
                 approximate_recommend=True, random_state=None,
                 show_progress=True):

        super(NMSAlternatingLeastSquares, self).__init__(
            factors=factors, regularization=regularization,
            use_native=use_native, use_cg=use_cg, iterations=iterations,
            use_gpu=use_gpu, calculate_training_loss=calculate_training_loss,
            num_threads=num_threads, random_state=random_state,
            show_progress=show_progress)

        self.method = method
        self.index_params = index_params
        self.query_params = query_params
        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

    def _make_estimator(self, X):
        # Get the number of factors
        factors = get_n_factors(X.shape[1], self.factors)
        n_users, n_items = X.shape

        # Implicit really only likes numpy float32 for some reason
        dtype = np.float32
        est = aals.NMSLibAlternatingLeastSquares(
            method=self.method, index_params=self.index_params,
            query_params=self.query_params,
            approximate_similar_items=self.approximate_similar_items,
            approximate_recommend=self.approximate_recommend,
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

    @inherit_function_doc(AlternatingLeastSquares)
    def fit(self, X):
        # Validate that X is a sparse array. Implicit forces float32, so
        # better to check it now...
        X = check_sparse_array(X, dtype=np.float32, copy=False,
                               force_all_finite=True)

        # Fit the estimator (Implicit likes it in item-user major order...)
        self.estimator_ = est = self._make_estimator(X)
        est.fit(X.T, show_progress=self.show_progress)

        return self

    @inherit_function_doc(AlternatingLeastSquares)
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False,
                           recalculate_user=False):
        # TODO:
        pass
