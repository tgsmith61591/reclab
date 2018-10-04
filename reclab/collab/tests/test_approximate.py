# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.utils.testing import RecommenderTestClass
from reclab.collab import NMSAlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os
import types

# set this to avoid the MKL BLAS warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load data and split into train/test
lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)


class TestNMSAlternatingLeastSquares(RecommenderTestClass):
    def test_simple_fit(self):
        clf1 = NMSAlternatingLeastSquares(
            approximate_recommend=True,
            approximate_similar_items=True,
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        clf2 = NMSAlternatingLeastSquares(
            approximate_recommend=True,
            approximate_similar_items=True,
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

        # Show we can fit
        clf1.fit(train)
        clf2.fit(train)

    def test_complex_fit(self):
        # Show we can fit a really complex model
        NMSAlternatingLeastSquares(
            approximate_similar_items=False,
            approximate_recommend=True, num_threads=1,
            random_state=42, use_cg=True, iterations=10,
            factors=150, regularization=0.01)

    def test_recommend_single(self):
        pass

    def test_recommend_all(self):
        pass

    def test_serialize(self):
        pass
