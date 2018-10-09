# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.base import BaseEstimator
from sklearn.externals import six

from abc import ABCMeta, abstractmethod
import itertools
import numpy as np

__all__ = [
    'BaseRecommender'
]


def _recommend_items_and_maybe_scores(best, return_scores, filter_items, n):
    """Internal method for recommender inference.

    Returns recommendations for items as well as maybe their corresponding
    scores if a user actually wants them.

    Parameters
    ----------
    best : iterable
        Iterable of tuples, where first index is the item and the second
        is the score.

    return_scores : bool
        Whether to get the scores back.

    filter_items : set
        Items to filter out

    n : int
        The number to return

    Examples
    --------
    If return_scores is False, we just get the filtered down item IDs back:

    >>> best = [(4, 0.98), (2, 0.91), (3, 0.76), (0, 0.52)]
    >>> _recommend_items_and_maybe_scores(best=best, return_scores=False,
    ...     filter_items={2}, n=5)
    array([4, 3, 0])

    If return_scores is true, we get a tuple back:

    >>> _recommend_items_and_maybe_scores(best=best, return_scores=True,
    ...     filter_items={2}, n=5)
    (array([4, 3, 0]), array([0.98, 0.76, 0.52]))
    """
    if return_scores:
        recs, scores = zip(
            *itertools.islice(
                (rec for rec in best
                 if rec[0] not in filter_items), n))
        return np.asarray(recs), np.asarray(scores)

    # Otherwise no need to unpack expression prior to returning
    return np.asarray(list(
        itertools.islice(
            (rec[0] for rec in best
             if rec[0] not in filter_items), n)))


class BaseRecommender(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for all recommenders.

    All recommenders must implement a ``fit`` method as well as the following
    inference methods:

        * ``recommend_for_user``
        * ``recommend_for_all_users``

    The BaseRecommender behaves similarly to the BaseEstimator in scikit-learn:
    all ``__init__`` args must be defined with defaults and without keyword
    arguments.

    Notes
    -----
    Some implicit estimators don't play nicely with serialization, but
    subclasses of BaseRecommender should. That means some subclasses will have
    ``__getstate__`` and ``__setstate__`` hacks.
    """

    @abstractmethod
    def fit(self, X):
        """Fit the recommender.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            A sparse matrix of ratings or content.
        """

    @abstractmethod
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

    def recommend_for_all_users(self, R, n=10, filter_previously_rated=True,
                                filter_items=None, return_scores=False,
                                **kwargs):
        """Produce recommendations for all users.

        Parameters
        ----------
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

        Notes
        -----
        For some recommendation algorithms, there may be faster ways to compute
        this. This naively loops each user, calling the
        :func:`recommend_for_user` function.
        """
        return (
            self.recommend_for_user(
                userid=i, R=R, n=n, filter_items=filter_items,
                filter_previously_rated=filter_previously_rated,
                return_scores=return_scores, **kwargs)
            for i in range(R.shape[0]))
