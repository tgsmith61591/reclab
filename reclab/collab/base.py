# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.externals import six
from abc import ABCMeta, abstractmethod

from ..base import BaseRecommender

__all__ = [
    'BaseCollaborativeFiltering'
]


class BaseCollaborativeFiltering(six.with_metaclass(ABCMeta, BaseRecommender)):
    """Base class for all collaborative filtering methods.

    Collaborative filtering is a family of recommender system algorithms that
    learn patterns based on the ratings history of users. The collaborative
    filtering algorithms implemented in reclab learn from ratings matrices
    only.
    """

    @abstractmethod
    def fit(self, X):
        """Fit the recommender on the ratings history of users.

        Trains the model, learning its parameters based only on the ratings
        history of a system's users.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            The sparse ratings matrix with users along the row axis and items
            along the column axis. Entries represent ratings or other implicit
            ranking events (i.e., number of listens, etc.)
        """

    @staticmethod
    def _initialize_factors(estimator, n_users, n_items, factors,
                            dtype, random_state):
        """Initialize the factor matrices.

        Implicit does not allow us to control seeding of the matrices, but
        it does allow us to provide our own matrices! This is the only way
        we can control reproducibility.
        """
        # XXX: we could also make the initialization strategy a tuning param?
        estimator.user_factors = \
            random_state.rand(n_users, factors).astype(dtype) * 0.01
        estimator.item_factors = \
            random_state.rand(n_items, factors).astype(dtype) * 0.01
        return estimator


class BaseMatrixFactorization(six.with_metaclass(ABCMeta,
                                                 BaseCollaborativeFiltering)):
    """Base class for all collaborative filtering methods with matrix
    factorization. Allows us to abstract out some of the method documentation.
    """
    @abstractmethod
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False,
                           recalculate_user=False):
        """Produce a recommendation for a user.

        Compute a user's recommendations as a product of his/her ratings
        history and the extracted latent factors for users and items.

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

        recalculate_user : bool, optional (default=False)
            Whether to recalculate the user factor.
        """
