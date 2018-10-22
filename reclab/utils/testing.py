# -*- coding: utf-8 -*-

from __future__ import absolute_import

from sklearn.externals import six, joblib
from abc import ABCMeta, abstractmethod

from numpy.testing import assert_array_equal
import numpy as np

import types
import shutil
import os

from .._config import RECLAB_CACHE, set_blas_singlethread

__all__ = [
    'RecommenderTestClass'
]

# The moment anything is imported from in the testing directory, set the
# BLAS threading to single thread, since we only import here when testing.
set_blas_singlethread()


class RecommenderTestClass(six.with_metaclass(ABCMeta)):
    """An abstract base test class for algo test suites.
    All recommender algorithm test classes should inherit from this.
    """
    @abstractmethod
    def test_simple_fit(self, *args, **kwargs):
        """Test a simple fit"""

    @abstractmethod
    def test_complex_fit(self, *args, **kwargs):
        """Test a more complex fit"""

    @abstractmethod
    def test_recommend_single(self, *args, **kwargs):
        """Test recommending for a single user."""

    @abstractmethod
    def test_recommend_all(self, *args, **kwargs):
        """Test recommending for all users."""

    @abstractmethod
    def test_serialize(self, *args, **kwargs):
        """Test serializing the algo."""

    @staticmethod
    def _single_recommend_assertions(clf, train_data, test_data):
        # Simple recommendation operation
        recs = clf.recommend_for_user(0, test_data, n=5,
                                      filter_previously_rated=False)
        assert len(recs) == 5

        # Create recommendations for everything, but filter out a single item
        n = train_data.shape[1]
        recs = clf.recommend_for_user(0, test_data, n=n,
                                      filter_previously_rated=False,
                                      filter_items=[1])

        # Show that '1' is not in the recommendations
        mask = np.in1d([1], recs)  # type: np.ndarray
        assert not mask.any()

        # Show we can also create recommendations with return_scores=True
        recs, scores = clf.recommend_for_user(
            0, test_data, n=5, return_scores=True)
        assert len(recs) == len(scores) == 5, (recs, scores)
        assert all(isinstance(arr, np.ndarray) for arr in (recs, scores))

    @staticmethod
    def _all_recommend_assertions(clf, test_data):
        n = test_data.shape[1]
        recs = clf.recommend_for_all_users(test_data, n=n,
                                           return_scores=True,
                                           filter_previously_rated=True)

        # Show that it's a generator
        assert isinstance(recs, types.GeneratorType)
        first_recs, first_scores = next(recs)
        assert len(first_recs) == len(first_scores)

        # show no rated items in the recs
        rated = test_data[0, :].indices
        mask = np.in1d(rated, first_recs)  # type: np.ndarray
        assert not mask.any()

    @staticmethod
    def _serialization_assertions(clf, train_data, test_data,
                                  tolerate_fail=False):
        pkl_location = "als.pkl"

        # Test persistence
        try:
            # Show we can serialize BEFORE it's fit
            joblib.dump(clf, pkl_location, compress=3)
            os.unlink(pkl_location)

            # NOW train
            clf.fit(train_data)

            # Get recommendations
            recs1 = clf.recommend_for_user(0, test_data, n=3,
                                           return_scores=False)

            # dump it, recommend again and show the internal state didn't
            # change while we were pickling it out
            joblib.dump(clf, pkl_location, compress=3)
            recs2 = clf.recommend_for_user(0, test_data, n=3,
                                           return_scores=False)

            # open it up and create more recommendations
            loaded = joblib.load(pkl_location)
            recs3 = loaded \
                .recommend_for_user(0, test_data, n=3,
                                    return_scores=False)

            # Now show they're all the same
            if not tolerate_fail:
                assert_array_equal(recs1, recs2,
                                   err_msg="%s != %s" % (str(recs1),
                                                         str(recs2)))
                assert_array_equal(recs1, recs3,
                                   err_msg="%s != %s" % (str(recs1),
                                                         str(recs3)))

        finally:
            os.unlink(pkl_location)

            # If the model has an index saved somewhere, remove it also
            if hasattr(clf, "_model_key"):
                index_cache = os.path.join(RECLAB_CACHE, clf._model_key)
                if os.path.exists(index_cache):
                    shutil.rmtree(index_cache)
