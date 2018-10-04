# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.externals import six
from abc import ABCMeta, abstractmethod

__all__ = [
    'RecommenderTestClass'
]


class RecommenderTestClass(six.with_metaclass(ABCMeta)):
    """An abstract base test class for algo test suites.
    All recommender algorithm test classes should inherit from this.
    """
    @abstractmethod
    def test_simple_fit(self):
        """Test a simple fit"""

    @abstractmethod
    def test_complex_fit(self):
        """Test a more complex fit"""

    @abstractmethod
    def test_recommend_single(self):
        """Test recommending for a single user."""

    @abstractmethod
    def test_recommend_all(self):
        """Test recommending for all users."""

    @abstractmethod
    def test_serialize(self):
        """Test serializing the algo."""
