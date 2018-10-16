# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
import numpy as np
import numbers
import warnings

from ..base import BaseRecommender
from ..utils.validation import is_iterable

__all__ = [
    'RecommenderDeployment'
]


class RecommenderDeployment(BaseEstimator):
    r"""A wrapper for deployed recommender models.

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

    filter_items : array-like or None, optional (default=None)
        Any items that should be filtered out of the recommend operation. This
        is included in the constructor of the class since it may grow
        arbitrarily large. This allows us to prevent having to send a
        potentially large amount of data over REST repeatedly. If
        ``filter_items`` differs per user, it may be provided in ``**kwargs``
        in the ``recommend_for_user`` call.

    user_missing_strategy : str or unicode, optional (default='warn')
        How to handle a request for recommendation for a user who does not
        exist in the system.

        error : Raises a KeyError with a detailed message. This is not the
            default behavior, and should be controlled for if this is run from
            an end-point, so as not to raise at the server level.

        warn : Warns with a detailed message and returns an empty set of
            recommendations. If this is the strategy you use, you'll have to
            check for empty recommendations and default to a set of your
            choosing.
    """

    def __init__(self, estimator, item_encoder=None, user_encoder=None,
                 filter_items=None, user_missing_strategy="warn"):

        # Validate so the user doesn't deploy this and THEN have it validate...
        for encoder in (item_encoder, user_encoder):
            if encoder is not None and not isinstance(encoder, LabelEncoder):
                raise TypeError("Encoders must be either None or a scikit-"
                                "learn LabelEncoder, but got type=%s"
                                % type(encoder))

        # Validate the filter items
        if filter_items is not None:
            if not is_iterable(filter_items):
                raise TypeError("filter_items must be either None "
                                "or an iterable, but got type=%s"
                                % type(filter_items))
            # otherwise it IS an iterable and we want it to be a set
            filter_items = set(filter_items)

        # The user strategy has to be defined appropriately. Don't bother
        # creating a set here to check... lookup time might be O(1), but create
        # + lookup makes it less efficient. Besides, O(2) ain't too shabby.
        if user_missing_strategy not in ("warn", "error"):
            raise ValueError("user_missing_strategy must be one of 'warn' or "
                             "'error', but got %s" % user_missing_strategy)

        self.estimator = estimator
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.filter_items = filter_items
        self.user_missing_strategy = user_missing_strategy

    def recommend_for_user(self, userid, ratings, n=10,
                           filter_previously_rated=True, return_scores=False,
                           **kwargs):
        """Produce a recommendation for a user.

        Parameters
        ----------
        userid : int or string
            The positional index along the row axis of the user in the
            ratings matrix, or the user ID that will be transformed via the
            ``user_encoder`` (if present).

        ratings : dict or array-like
            The user's latest collection of rated items used to seed
            recommendations. Can either be a dictionary or array of ratings,
            and is subject to the following constraints:

            dict : If it's a dictionary and the ``item_encoder`` is not None,
                the keys will be encoded via the encoder's ``transform``
                function, and values (ratings) loaded into a numpy array of the
                encoded keys. If the encoder is None, the keys are assumed to
                represent positional indices into which they'll be placed in a
                numpy array.

            array-like : If the input of rated elements is array-like (list or
                numpy array), the dimensions are assumed to match those of the
                item factors or similarity matrix the algorithm uses
                internally. No encoding will be performed, since values in the
                array represent ratings, and indices represent item IDs.

        n : int, optional (default=10)
            The number of items to recommend for the given user.

        filter_previously_rated : bool, optional (default=True)
            Whether to filter items that have been previously rated by the
            user. True by default.

        return_scores : bool, optional (default=False)
            Whether to return the scores for each item for the user.

        **kwargs : keyword args or dict
            Arguments specific to subclassed algorithms.

        Returns
        -------
        recs : np.ndarray
            The recommended items. If ``self.item_encoder`` is not None, the
            items will be returned in their inverse-transformed states.

        scores (optional) : np.ndarray
            The scores that correspond the recommendations. Only included if
            ``return_scores`` is True.
        """
        # We'll allow filter_items to be sent in kwargs. It will override the
        # class filter_items
        filter_items = kwargs.pop("filter_items", self.filter_items)
        estimator = self.estimator  # type: BaseRecommender
        n_users = estimator.n_users()
        n_items = estimator.n_items()

        # Now encode the user, if needed
        ue = self.user_encoder
        original_userid = userid  # store for error reporting later
        if ue is not None:
            userid = ue.transform([userid])[0]
        # Make sure it's an int
        if not isinstance(userid, numbers.Integral):
            raise TypeError("userid must be an integer if it's not be "
                            "encoded! Got %s (type=%s)"
                            % (str(userid), type(userid)))
        # Cast to int in case it's a np.int64 or anything
        userid = int(userid)

        # Unpack the ratings into an array, encode if needed
        ie = self.item_encoder
        if isinstance(ratings, dict):
            keys, values = list(zip(*ratings.items()))

            # todo: What if the dict includes keys that aren't in the model?
            if ie is not None:
                keys = ie.transform(keys)

        # Otherwise it's array-like?
        else:
            ratings = np.asarray(ratings)
            if ratings.shape[0] != n_items:
                raise ValueError("Dim mismatch! Expected %i items but ratings "
                                 "vector contained %i"
                                 % (n_items, ratings.shape[0]))

            # get the keys (no item encoding here)
            keys = np.where(ratings != 0)[0]
            values = ratings[keys]

        # Create a sparse CSR matrix with the encoded userid as the index.
        # Will raise if the user ID is out of bounds
        try:
            R = sparse.csr_matrix((values, ([userid] * len(values), keys)),
                                  shape=(n_users, n_items))
        except ValueError as ex:
            # report the error
            msg = "User %s could not be found. Were they included in the " \
                  "model fit?" % (str(original_userid))
            if self.user_missing_strategy == "error":
                raise KeyError(msg, ex)
            else:  # Make sure the exception detail makes it into the warning
                warnings.warn(msg + " (%s)" % str(ex))
                recs = np.array([], dtype=np.int64)
                if return_scores:
                    scores = np.array([], dtype=np.float32)
                    return recs, scores
                return recs

        # Should be safe to get recommendations now
        recs = estimator.recommend_for_user(
            userid=userid, R=R, n=n, filter_items=filter_items,
            filter_previously_rated=filter_previously_rated,
            return_scores=return_scores, **kwargs)

        # Unpack if needed
        if return_scores:
            recs, scores = recs  # would be a tuple
        else:
            # Just so it's in the namespace... this is never used but it keeps
            # the IDE from complaining that 'scores' may not have been assigned
            # yet when returning.
            scores = None

        # Inverse transform the recommended items
        if ie is not None:
            recs = ie.inverse_transform(recs)

        if return_scores:
            return recs, scores
        return recs
