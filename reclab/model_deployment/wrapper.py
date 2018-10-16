# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.preprocessing import LabelEncoder

__all__ = [
    'RecommenderDeployment'
]


class RecommenderDeployment():
    """A wrapper for deployed recommender models.

    Serves as a model wrapper for recommender models as they are deployed.
    Can map recommendations to different encodings (with a pre-fit sklearn
    LabelEncoder) as well as handle users who have not yet been included in
    the model.

    Parameters
    ----------
    estimator : BaseRecommender
        The pre-fit recommender model used to serve recommendations to users.

    item_encoder : sklearn.preprocessing.LabelEncoder, optional (default=None)
        A pre-fit encoder to map recommended items produced by the recommender
        back to encodings that they keys may better represent. This is useful
        if products have GUIDs that do not sequentially align with the indexing
        of a matrix (as is probable).

    user_encoder : sklearn.preprocessing.LabelEncoder, optional (default=None)
        A pre-fit encoder used to map user IDs to their positional index in the
        user factors or stored similarity matrices. If provided, the input user
        will be transformed to the indexing space to compute the
        recommendations.

    filter_previously_rated : bool, optional (default=True)
        Whether to filter items that have been previously rated by the
        user. True by default.

    filter_items : array-like or None, optional (default=None)
        Any items that should be filtered out of the recommend operation.

    return_scores : bool, optional (default=False)
        Whether to return the scores for each item for the user.

    user_missing_strategy : str or unicode, optional (default='warn')
        How to handle a request for recommendation for a user who does not
        exist in the system.

        error : Raises a KeyError with a detailed message. This is not the
            default behavior, and should be controlled for if this is run from
            an end-point, so as not to raise at the server level.

        warn : Warns with a detailed message and returns the ``n`` most popular
            items from
    """

    def __init__(self, estimator, item_encoder=None, user_encoder=None,
                 filter_previously_rated=True, filter_items=None,
                 return_scores=False, user_missing_strategy="warn"):

        self.estimator = estimator
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.filter_previously_rated = filter_previously_rated
        self.filter_items = filter_items
        self.return_scores = return_scores
        self.user_missing_strategy = user_missing_strategy

    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False, **kwargs):
        """Produce a recommendation for a user given his/her ratings history.

        Parameters
        ----------
        userid : int
            The positional index along the row axis of the user in the
            ratings matrix.

        R : scipy.sparse.csr_matrix
            The sparse ratings matrix of users (along the row axis) and items
            (along the column axis)

        n : int, optional (default=10)
            The number of items to recommend for the given user.

        filter_previously_rated : bool, optional (default=True)
            Whether to filter items that have been previously rated by the
            user. True by default.

        filter_items : array-like or None, optional (default=None)
            Any items that should be filtered out of the recommend operation.

        return_scores : bool, optional (default=False)
            Whether to return the scores for each item for the user.

        **kwargs : keyword args or dict
            Arguments specific to subclassed algorithms.
        """
