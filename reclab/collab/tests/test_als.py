# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.collab import AlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.utils.testing import RecommenderTestClass

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, \
    assert_array_equal
from sklearn.externals import joblib

import os
import types

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
        AlternatingLeastSquares(random_state=42, use_cg=True, iterations=50,
                                factors=150, regularization=0.01)

    def test_recommend_single(self):
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)
        clf.fit(train)

        # Simple recommendation operation
        recs = clf.recommend_for_user(0, test, n=5,
                                      filter_previously_rated=False)
        assert len(recs) == 5

        # Create recommendations for everything, but filter out a single item
        n = train.shape[1]
        recs = clf.recommend_for_user(0, test, n=n,
                                      filter_previously_rated=False,
                                      filter_items=[1])

        # Show that '1' is not in the recommendations
        assert not np.in1d([1], recs).any()

    def test_recommend_all(self):
        # Recommend for ALL users
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5).fit(train)

        n = test.shape[1]
        recs = clf.recommend_for_all_users(test, n=n,
                                           return_scores=True,
                                           filter_previously_rated=True)

        # Show that it's a generator
        assert isinstance(recs, types.GeneratorType)
        first_recs, first_scores = next(recs)
        assert len(first_recs) == len(first_scores)

        # show no rated items in the recs
        rated = test[0, :].indices
        assert not np.in1d(rated, first_recs).any()

    def test_serialize(self):
        clf = AlternatingLeastSquares(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5)
        pkl_location = "als.pkl"

        # Test persistence
        try:
            # Show we can serialize BEFORE it's fit
            joblib.dump(clf, pkl_location, compress=3)
            os.unlink(pkl_location)

            # NOW train
            clf.fit(train)

            # Get recommendations
            recs1 = clf.recommend_for_user(0, test, n=3, return_scores=False)

            # dump it, recommend again and show the internal state didn't
            # change while we were pickling it out
            joblib.dump(clf, pkl_location, compress=3)
            recs2 = clf.recommend_for_user(0, test, n=3, return_scores=False)

            # open it up and create more recommendations
            recs3 = joblib.load(pkl_location)\
                          .recommend_for_user(0, test, n=3,
                                              return_scores=False)

            # Now show they're all the same
            assert_array_equal(recs1, recs2,
                               err_msg="%s != %s" % (str(recs1), str(recs2)))
            assert_array_equal(recs1, recs3,
                               err_msg="%s != %s" % (str(recs1), str(recs3)))

        finally:
            os.unlink(pkl_location)
