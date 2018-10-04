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

    def _initialize_factors(self, estimator, n_users, n_items, factors,
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
