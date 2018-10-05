# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.utils.testing import RecommenderTestClass
from reclab.collab import NMSAlternatingLeastSquares, \
    AnnoyAlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split

from numpy.testing import assert_array_equal, assert_array_almost_equal

import os
import pytest

import numpy as np

# set this to avoid the MKL BLAS warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load data and split into train/test
lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)


class TestApproximateAlternatingLeastSquares(RecommenderTestClass):

    @pytest.mark.parametrize(
        'cls', [NMSAlternatingLeastSquares,
                AnnoyAlternatingLeastSquares])
    def test_simple_fit(self, cls):
        clf1 = cls(
            approximate_recommend=True,
            approximate_similar_items=True,
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        clf2 = cls(
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

    @pytest.mark.parametrize(
        'cls', [NMSAlternatingLeastSquares,
                AnnoyAlternatingLeastSquares])
    def test_complex_fit(self, cls):
        # Show we can fit a really complex model
        cls(approximate_similar_items=False,
            approximate_recommend=True, num_threads=1,
            random_state=42, use_cg=True, iterations=10,
            factors=150, regularization=0.01)

    @pytest.mark.parametrize(
        'cls', [NMSAlternatingLeastSquares,
                AnnoyAlternatingLeastSquares])
    def test_recommend_single(self, cls):
        clf = cls(
            approximate_recommend=True,
            approximate_similar_items=True,
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)
        clf.fit(train)

        # Make assertions on the recommendations
        self._single_recommend_assertions(clf, train, test)

    @pytest.mark.parametrize(
        'cls', [NMSAlternatingLeastSquares,
                AnnoyAlternatingLeastSquares])
    def test_recommend_all(self, cls):
        # Recommend for ALL users
        clf = cls(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)
        clf.fit(train)

        # Mask assertions
        self._all_recommend_assertions(clf, test)

    @pytest.mark.parametrize(
        'cls', [NMSAlternatingLeastSquares,
                AnnoyAlternatingLeastSquares])
    def test_serialize(self, cls):
        clf = cls(
            random_state=1, use_gpu=False, use_cg=True,
            iterations=5, num_threads=1)

        self._serialization_assertions(clf, train, test)
