# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .als import AlternatingLeastSquares
from ..utils.decorators import inherit_function_doc
from ..utils.validation import check_sparse_array, get_n_factors
from ..utils.system import safe_mkdirs
from .._config import RECLAB_CACHE

import numpy as np
from sklearn.utils.validation import check_random_state, check_is_fitted
from sklearn.externals import six
from sklearn.base import clone
from abc import ABCMeta, abstractmethod

from implicit import approximate_als as aals
from implicit import cuda

from annoy import AnnoyIndex
import nmslib

from os.path import join, exists
import uuid
import copy
import shutil

__all__ = [
    'AnnoyAlternatingLeastSquares',
    'NMSAlternatingLeastSquares'
]


def _get_filter_items(filter_previously_rated, previously_liked, filter_items):
    """Determine which items to filter out."""
    if filter_previously_rated:
        if filter_items is not None:
            previously_liked = np.append(previously_liked, filter_items)
        return previously_liked  # type: np.ndarray
    else:
        if filter_items is not None:
            return np.asarray(filter_items)
        else:
            return np.array([], dtype=np.int)


def _do_filter(items, scores, filter_out):
    """Filter items out of the recommendations.

    Given a list of items to filter out, remove them from recommended items
    and scores.
    """
    # If the filter is empty, just return (don't mask)
    if filter_out.shape[0] == 0:
        return items, scores

    remove_mask = np.in1d(items, filter_out)
    return items[~remove_mask], scores[~remove_mask]


def _recommend_aals_annoy(est, userid, R, n, filter_items,
                          recalculate_user, filter_previously_rated,
                          return_scores, recommend_function,
                          scaling_function, *args, **kwargs):
    """Produce recommendations for Annoy and NMS ALS algorithms"""

    user = est._user_factor(userid, R, recalculate_user)

    # Calculate the top N items, only removing the liked items from the
    # results if specified
    filter_out = _get_filter_items(filter_previously_rated,
                                   # Don't use user, since it might have
                                   # been re-estimated:
                                   R[userid].indices,
                                   filter_items)

    # If N is None, we set it to the number of items. The item_factors attr
    # exists in all ALS models here
    if n is None:
        n = est.item_factors.shape[0]

    # The count the produce
    count = n + len(filter_out)

    # See [1] in docstring for why we do this...
    query = np.append(user, 0)  # (is this the fastest way?)
    ids, dist = map(np.asarray,  # Need to be a Numpy array
                    recommend_function(query, count, *args, **kwargs))

    # Only compute the dist scaling if we care about them, since it's
    # expensive
    if return_scores:
        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product.
        scaling = est.max_norm * np.linalg.norm(query)
        dist = scaling_function(dist, scaling)  # sig: f(dist, scaling)

    # if we're filtering anything out...
    ids, dist = _do_filter(ids, dist, filter_out=filter_out)
    if return_scores:
        return ids[:n], dist[:n]
    return ids[:n]


class _BaseApproximateALS(six.with_metaclass(ABCMeta, AlternatingLeastSquares)):
    """Base class for approximated ALS variants"""
    def __init__(self, *args, **kwargs):
        # Pass the args for the constructor up
        super(_BaseApproximateALS, self).__init__(*args, **kwargs)

        # Assign a model key. This is used to load the model index from disk
        self._model_key = "%s-%s" % (self.__class__.__name__,
                                     str(uuid.uuid4()))

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

    def _preserialize_hook(self):
        """Pass-through unless overridden"""

    def __getstate__(self):
        """Pickle subhook for both Annoy and NMS algorithms"""
        # If it's not fit, this is super easy
        if not hasattr(self, "estimator_"):
            return self.__dict__

        # Otherwise we have to individually save elements of it...
        est = self.estimator_
        self._preserialize_hook()
        indices = {"similar_items_index": (est.similar_items_index,
                                           est.approximate_similar_items),
                   "recommend_index": (est.recommend_index,
                                       est.approximate_recommend)}

        # Remove the indices from the object for pickling
        est.similar_items_index = None
        est.recommend_index = None

        # Since the signatures of the __init__ functions should play nice with
        # sklearn, and since we've removed the un-picklables, we should be able
        # to copy this now.
        obj_dict = clone(self).__dict__

        # Make sure to bind the estimator to the object dictionary so it gets
        # pickled out.
        obj_dict['estimator_'] = copy.deepcopy(est)

        # The model key will get mangled, too, since the constructor gets
        # called again. Need to hang onto the proper one...
        obj_dict['_model_key'] = self._model_key

        # If the model key already exists in the cache, remove it now
        model_index_dir = join(RECLAB_CACHE, self._model_key)
        if exists(model_index_dir):
            shutil.rmtree(model_index_dir)
        safe_mkdirs(model_index_dir)

        # Save the indices to Disk. wrap this in try/finally so if something
        # breaks halfway through we don't blow up the disk space over time...
        try:
            for attr, (index, approx) in six.iteritems(indices):
                # Only do this if we're approximating the recommend operation,
                # otherwise the index is None anyways...
                if approx:
                    loc = join(model_index_dir, attr)

                    # Save index to cache. This is calling the abstract
                    # function that's of the signature f(index, loc) -> None
                    self._save_index(index, loc)

                    # Re-bind the object to the estimator
                    setattr(est, attr, index)

        # If we break down, remove the model index directory so as not to
        # blow up the filesystem!
        except Exception:
            shutil.rmtree(model_index_dir)
            raise

        return obj_dict

    def __setstate__(self, state):
        """Load an object from disk.

        The trick here is loading the similar items and recommend index objects
        from disk, as they don't play nice with Pickle or Joblib.
        """
        self.__dict__ = state

        # If the estimator_ attribute exists, we know we need to re-bind the
        # FloatIndex or AnnoyIndex attributes after we load them from disk
        if hasattr(self, "estimator_"):
            # Bind the index objects to the estimator to be able to make
            # recommendations
            est = self.estimator_
            for key, approx in (
                    ("similar_items_index", est.approximate_similar_items),
                    ("recommend_index", est.approximate_recommend)):

                # Only load it if we approximated, otherwise keep the
                # default as None
                if approx:
                    # Load the index and set it
                    location = join(RECLAB_CACHE, self._model_key)
                    setattr(est, key, self._load_index(location, key))

        return self

    @abstractmethod
    def _save_index(self, index, whereto):
        """Where to save the FloatIndex or AnnoyIndex objects."""

    @abstractmethod
    def _load_index(self, wherefrom, index_key):
        """Load an index from disk"""


class AnnoyAlternatingLeastSquares(_BaseApproximateALS):
    """Alternating Least Squares with nearest neighbor indexing.

    Improves :class:`reclab.collab.AlternatingLeastSquares` by using
    `Annoy <https://github.com/spotify/annoy>`_ to create an approximate
    nearest neighbors index of the latent factors. This dramatically reduces
    inference time.

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

    n_trees : int, optional (default=50)
        The number of trees to use when building the Annoy index. More trees
        gives higher precision when querying, but takes longer to index.

    search_k : int, optional (default=-1)
        Provides a way to search more trees at runtime, giving the ability to
        have more accurate results at the cost of taking more time.

    approximate_similar_items : bool, optional (default=True)
        Whether or not to build an Annoy index for computing similar_items.
        Default (recommended) is True.

    approximate_recommend : bool, optional (default=True)
        Whether or not to build an Annoy index for the recommend call.
        Default (recommended) is True.

    random_state : int, RandomState or None, optional (default=None)
        The random state to control random seeding of the item and user
        factor matrices.

    show_progress : bool, optional (default=True)
        Whether to show a progress bar while training.

    References
    ----------
    .. [1] https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/
    """
    def __init__(self, factors=100, regularization=0.01,
                 use_native=True, use_cg=True, iterations=15,
                 use_gpu=cuda.HAS_CUDA, calculate_training_loss=False,
                 num_threads=0, n_trees=50, search_k=-1,
                 approximate_similar_items=True, approximate_recommend=True,
                 random_state=None, show_progress=True):

        super(AnnoyAlternatingLeastSquares, self).__init__(
            factors=factors, regularization=regularization,
            use_native=use_native, use_cg=use_cg, iterations=iterations,
            use_gpu=use_gpu, calculate_training_loss=calculate_training_loss,
            num_threads=num_threads, random_state=random_state,
            show_progress=show_progress)

        self.n_trees = n_trees
        self.search_k = search_k
        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

    def _make_estimator(self, X):
        # Get the number of factors
        factors = get_n_factors(X.shape[1], self.factors)
        n_users, n_items = X.shape

        # Implicit really only likes numpy float32 for some reason
        dtype = np.float32
        est = aals.AnnoyAlternatingLeastSquares(
            n_trees=self.n_trees, search_k=self.search_k,
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

    def _preserialize_hook(self):
        """Used by Annoy ALS to get the AnnoyIndex dimension pre-save"""
        if hasattr(self, "estimator_"):
            est = self.estimator_

            # This is used for instantiating the new AnnoyIndex when
            # loading from disk
            if est.approximate_recommend:
                # It sucks we have to do this again... expensive, but we need
                # the dimension and the index doesn't store it directly.
                _, extra = aals.augment_inner_product_matrix(
                    est.item_factors)
                est.extra_ = extra.shape[1]

    @inherit_function_doc(AlternatingLeastSquares)
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False,
                           recalculate_user=False):
        # Make sure we're fitted
        check_is_fitted(self, "estimator_")

        # If not using approximate recommendations, just delegate to super
        if not self.estimator_.approximate_recommend:
            return self.estimator_.recommend(
                userid, R, N=n, filter_items=filter_items,
                recalculate_user=recalculate_user)

        # Otherwise, do things a bit differently from the implicit package
        est = self.estimator_

        # This is the scaling function we use to map the distances from
        # Euclidean back to Cosine:
        scaling_function = (lambda dist, scaling:
                            scaling * (1 - (np.array(dist) ** 2) / 2))

        return _recommend_aals_annoy(
            est=est, userid=userid, R=R, n=n,
            filter_items=filter_items, recalculate_user=recalculate_user,
            filter_previously_rated=filter_previously_rated,
            return_scores=return_scores,
            recommend_function=est.recommend_index.get_nns_by_vector,
            scaling_function=scaling_function,

            # Passed to the recommend function with query, count
            include_distances=True, search_k=est.search_k)

    def _save_index(self, index, whereto):
        """Where to save the FloatIndex or AnnoyIndex objects."""
        index.save(whereto)

    def _load_index(self, wherefrom, index_key):
        """Load an AnnoyIndex from disk"""
        est = self.estimator_

        # I can't think of anything more clever because I've been up for
        # hours and hours and hours, so this is the kludgiest solution:
        if index_key == "similar_items_index":
            n_index = est.item_factors.shape[1]
        # Otherwise, "recommend_index"
        else:
            # This assumes approximate_recommend, since it's the only way
            # it will ever get to this code
            n_index = est.extra_
        index = AnnoyIndex(n_index, "angular")
        index.load(join(wherefrom, index_key))
        return index


class NMSAlternatingLeastSquares(_BaseApproximateALS):
    """Alternating Least Squares with nearest neighbor indexing.

    Improves :class:`reclab.collab.AlternatingLeastSquares` by using
    `NMSLib <https://github.com/searchivarius/nmslib>`_ to create an
    approximate nearest neighbors index of the latent factors. This
    dramatically reduces inference time.

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
        Small World). For a list of exhaustive methods, see [2]

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

    References
    ----------
    .. [1] https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/
    .. [2] Valid NMSLIB method source code
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
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False,
                           recalculate_user=False):
        # Make sure we're fitted
        check_is_fitted(self, "estimator_")

        # If not using approximate recommendations, just delegate to super
        if not self.estimator_.approximate_recommend:
            return self.estimator_.recommend(
                userid, R, N=n, filter_items=filter_items,
                recalculate_user=recalculate_user)

        # Otherwise, do things a bit differently from the implicit package
        est = self.estimator_

        # This is the scaling function we use to map the distances from
        # Euclidean back to Cosine:
        scaling_function = (lambda dist, scaling: scaling * (1 - dist))

        return _recommend_aals_annoy(
            est=est, userid=userid, R=R, n=n,
            filter_items=filter_items, recalculate_user=recalculate_user,
            filter_previously_rated=filter_previously_rated,
            return_scores=return_scores,
            recommend_function=est.recommend_index.knnQuery,
            scaling_function=scaling_function)

    def _save_index(self, index, whereto):
        """Where to save the FloatIndex or AnnoyIndex objects."""
        index.saveIndex(whereto)

    def _load_index(self, wherefrom, index_key):
        """Load an NMS index from disk"""
        # NMSLib doesn't let us read this via a staticmethod so we need
        # to initialize a new index object (blech)
        # https://github.com/nmslib/nmslib/issues/350
        index = nmslib.init(method=self.method, space='cosinesimil')
        index.loadIndex(join(wherefrom, index_key))
        index.setQueryTimeParams(self.query_params)
        return index
