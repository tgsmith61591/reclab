# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.collab import AlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.utils.testing import RecommenderTestClass

from numpy.testing import assert_array_almost_equal, assert_allclose
import os

# set this to avoid the MKL BLAS warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load data and split into train/test
lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)


class TestAlternatingLeastSquares(RecommenderTestClass):
    def test_simple_fit(self):
        clf1 = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        clf2 = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        # Show that the _make_estimator will initialize the matrices in a
        # replicable fashion given the random seed
        # PRE-FIT:
        est1 = clf1._make_estimator(train)
        est2 = clf2._make_estimator(train)
        for attr in ('item_factors', 'user_factors'):
            assert_array_almost_equal(getattr(est1, attr),
                                      getattr(est2, attr))

        # Are they the same POST-fit? They SHOULD be... (note this is only
        # the case if use_cg is FALSE!!)
        clf1.fit(train)
        clf2.fit(train)

        # This SHOULD work--is something in implicit random?...
        # assert_allclose(clf1.estimator_.item_factors,
        #                 clf2.estimator_.item_factors, rtol=1e-5)
        # assert_allclose(clf1.estimator_.user_factors,
        #                 clf2.estimator_.user_factors, rtol=1e-5)

    def test_complex_fit(self):
        # Show we can fit a really complex model
        AlternatingLeastSquares(random_state=42, use_cg=True, iterations=15,
                                factors=150, regularization=0.01,
                                num_threads=1)

    def test_recommend_single(self):
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)
        clf.fit(train)

        # Make assertions on the recommendations
        self._single_recommend_assertions(clf, train, test)

    def test_recommend_all(self):
        # Recommend for ALL users
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1).fit(train)

        # Mask assertions
        self._all_recommend_assertions(clf, test)

    def test_serialize(self):
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        self._serialization_assertions(clf, train, test)
