# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.collab import ItemItemRecommender
from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.utils.testing import RecommenderTestClass

import os
import pytest

# set this to avoid the MKL BLAS warning
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load data and split into train/test
lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)

metrics = ['kernel', 'cosine', 'tfidf', 'bm25']


class TestNeighborsAlgos(RecommenderTestClass):
    @pytest.mark.parametrize("metric", metrics)
    def test_simple_fit(self, metric):
        clf = ItemItemRecommender(k=5, metric=metric)
        clf.fit(train)

        # Show the num items matches expected
        assert clf.estimator_.similarity.shape[1] == train.shape[1]

    @pytest.mark.parametrize("metric", metrics)
    def test_complex_fit(self, metric):
        # Show we can fit a really complex model
        clf = ItemItemRecommender(metric=metric, k=50)
        clf.fit(train)

    @pytest.mark.parametrize("metric", metrics)
    def test_recommend_single(self, metric):
        clf = ItemItemRecommender(k=5, metric=metric)
        clf.fit(train)

        # Make assertions on the recommendations
        self._single_recommend_assertions(clf, train, test)

    @pytest.mark.parametrize("metric", metrics)
    def test_recommend_all(self, metric):
        # Recommend for ALL users
        clf = ItemItemRecommender(k=5, metric=metric).fit(train)

        # Mask assertions
        self._all_recommend_assertions(clf, test)

    @pytest.mark.parametrize("metric", metrics)
    def test_serialize(self, metric):
        clf = ItemItemRecommender(metric=metric, k=5)
        self._serialization_assertions(clf, train, test)
