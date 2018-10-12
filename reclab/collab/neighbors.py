# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .base import BaseCollaborativeFiltering
from ..base import _recommend_items_and_maybe_scores, clone
from ..utils.decorators import inherit_function_doc
from ..utils.system import safe_mkdirs
from ..utils.validation import check_sparse_array, check_permitted_value
from .._config import RECLAB_CACHE

from os.path import join, exists
from scipy import sparse
import numpy as np
import shutil
import copy

from sklearn.utils.validation import check_is_fitted
from implicit import nearest_neighbours as nn
from implicit._nearest_neighbours import NearestNeighboursScorer

__all__ = [
    'ItemItemRecommender'
]

_estimators = {
    'kernel': nn.ItemItemRecommender,
    'cosine': nn.CosineRecommender,
    'tfidf': nn.TFIDFRecommender,
    'bm25': nn.BM25Recommender
}


class ItemItemRecommender(BaseCollaborativeFiltering):
    r"""Item-item collaborative filtering.

    Computes & recommends the nearest neighbors between items.
    Recommendations are produced by multiplying a user's likes (rated
    items) by the precomputed item similarity matrix.

    Parameters
    ----------
    metric : str or unicode, optional (default='kernel')
        kernel :
            Computes the item-pair similarities via the ratings matrix's
            self product: :math:`X^{T}X`
        cosine :
            Item-pair similarities are calculated via cosine similarity
            (equivalent to 'kernel' method on a normalized matrix).
        tfidf :
            Identical to the 'kernel' method applied to a normalized,
            TFIDF-weighted matrix.
        bm25 :
            Okapi BM25 (BM for "best matching") is a ranking function for
            search engines that ranks by relevancy, and is related to the
            "tfidf" method. See [1] and [2] for more information.

    k : int, optional (default=20)
        The number of nearest neighbors to store for each item. A higher 'k'
        value will cause the method to store a more dense similarity matrix,
        and will yield a higher bias-afflicted system, while a lower value of
        'k' will store a more sparse similarity matrix, but trends towards
        a higher variance system.

    k1 : float, optional (default=1.2)
        A free parameter used for BM25 similarity computation. K1 is typically
        chosen, in the absence of advanced optimization, as
        :math:`k_{i} \in [1.2, 2.0]`. If ``metric`` is not 'bm25', ``k1`` is
        ignored.

    b : float, optional (default=0.75)
        A free parameter used for BM25 similarity computation. B is commonly
        defaulted to 0.75. If ``metric`` is not 'bm25', ``B`` is ignored.

    show_progress : bool, optional (default=True)
        Whether to show a progress bar while training.

    Examples
    --------
    Fitting a item-item recommender with cosine similarity:

    >>> from reclab.datasets import load_lastfm
    >>> from reclab.model_selection import train_test_split
    >>> lastfm = load_lastfm(cache=True)
    >>> train, test = train_test_split(u=lastfm.users, i=lastfm.products,
    ...                                r=lastfm.ratings, random_state=42)
    >>> model = ItemItemRecommender(k=5, metric='cosine', show_progress=False)
    >>> model.fit(train)  # doctest: +NORMALIZE_WHITESPACE
    ItemItemRecommender(b=0.75, k=5, k1=1.2, metric='cosine',
                  show_progress=False)

    Inference for a given user:

    >>> model.recommend_for_user(0, test, n=5)  # doctest: +SKIP
    array([12673,  4229,  8762,  2536, 14711], dtype=int32)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Okapi_BM25
    .. [2] https://xapian.org/docs/bm25.html
    """
    def __init__(self, metric='kernel', k=20, k1=1.2, b=0.75,
                 show_progress=True):

        # Call to super constructor
        super(ItemItemRecommender, self).__init__()

        self.metric = metric
        self.k = k
        self.k1 = k1
        self.b = b
        self.show_progress = show_progress

    def _make_estimator(self):
        # Validate the metric
        metric = self.metric
        cls = check_permitted_value(permitted_dict=_estimators,
                                    provided_key=metric)

        # If it's BM25, we have several other options we pass
        if metric == 'bm25':
            return cls(K=self.k, K1=self.k1, B=self.b)
        # Otherwise, they all have the same signature
        return cls(K=self.k)

    @inherit_function_doc(BaseCollaborativeFiltering)
    def fit(self, X):
        # Validate that X is a sparse array. Implicit forces float32 for ALS,
        # but forces 64 for nearest neighbors (how annoying, right?)
        X = check_sparse_array(X, dtype=np.float64, copy=False,
                               force_all_finite=True)

        # Now fit it
        self.estimator_ = est = self._make_estimator()
        est.fit(X.T, show_progress=self.show_progress)

        return self

    @inherit_function_doc(BaseCollaborativeFiltering)
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False, **kwargs):
        # Make sure we're fitted...
        check_is_fitted(self, "estimator_")
        R = check_sparse_array(R, dtype=np.float64, copy=False,
                               force_all_finite=True)

        # If n is None, make it n_items
        est = self.estimator_  # type: nn.ItemItemRecommender
        if n is None:
            n = est.similarity.shape[1]

        # If we're filtering previously rated, we need to add this length to N
        # otherwise the implicit code will come in low...
        rated = set(R[userid].indices)
        N = n  # Keep the original N so we don't amend it for later filtering
        if filter_previously_rated:
            n += len(rated)

        # Get list of tuples:
        best = est.recommend(
            userid=userid, user_items=R, N=N,
            filter_already_liked_items=filter_previously_rated,
            filter_items=filter_items)

        # There is a bug in the implicit code that will cause previously
        # rated items to still be returned, but with a rating of zero. We need
        # to remove these... fortunately, the filter_items (should) have
        # already been removed by the implicit code.
        filter_out = set() if not filter_previously_rated else rated
        return _recommend_items_and_maybe_scores(
            best, return_scores=return_scores, filter_items=filter_out, n=n)

    def __getstate__(self):
        """Pickle sub-hook"""
        # If it's not fit, we just return this dictionary
        if not hasattr(self, "estimator_"):
            return self.__dict__

        # Otherwise we have to separately save the similarity matrix
        est = self.estimator_

        # Remove the estimator object to clone
        sim = est.similarity
        scorer = est.scorer
        est.similarity = None
        est.scorer = None

        # Since the signatures of the __init__ functions should play nice with
        # sklearn, and since we've removed the un-picklables, we should be able
        # to copy this now.
        obj_dict = clone(self, clone_model_key=True).__dict__

        # Make sure to bind the estimator to the object dictionary so it gets
        # pickled out.
        obj_dict['estimator_'] = copy.deepcopy(est)

        # Re-bind the scorer and re-attach the similarity to the estimator
        # for calling the save function later
        est.similarity = sim
        est.scorer = scorer

        # If the model key already exists in the cache, remove it now
        model_index_dir = join(RECLAB_CACHE, self._model_key)
        if exists(model_index_dir):
            shutil.rmtree(model_index_dir)
        safe_mkdirs(model_index_dir)

        # Save the indices to Disk. wrap this in try/finally so if something
        # breaks halfway through we don't blow up the disk space over time...
        try:
            loc = join(model_index_dir, "similarity")
            np.savez(loc, data=sim.data, indptr=sim.indptr,
                     indices=sim.indices, shape=sim.shape)

        # If we break down, remove the model index directory so as not to
        # blow up the filesystem!
        except Exception:
            shutil.rmtree(model_index_dir)
            raise

        return obj_dict

    def __setstate__(self, state):
        """Unpickle sub-hook"""
        self.__dict__ = state

        # If the estimator_ attribute exists, we know we need to re-bind the
        # similarity attribute, otherwise the estimator was not previously fit.
        if hasattr(self, "estimator_"):
            est = self.estimator_
            # Numpy forces .npz suffix
            location = join(RECLAB_CACHE, self._model_key, "similarity.npz")

            # Load the similarity matrix
            arr = np.load(location)
            est.similarity = sparse.csr_matrix(
                (arr['data'], arr['indices'], arr['indptr']),
                shape=arr['shape'])
            est.scorer = NearestNeighboursScorer(est.similarity)

        return self
