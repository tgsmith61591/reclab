# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .base import BaseCollaborativeFiltering
from ..base import _recommend_items_and_maybe_scores
from ..utils.decorators import inherit_function_doc
from ..utils.validation import check_sparse_array, check_permitted_value

import numpy as np
from sklearn.utils.validation import check_is_fitted

from implicit import nearest_neighbours as nn

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

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Okapi_BM25
    .. [2] https://xapian.org/docs/bm25.html
    """
    def __init__(self, metric='kernel', k=20, k1=1.2, b=0.75,
                 show_progress=True):

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
