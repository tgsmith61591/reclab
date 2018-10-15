# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.model_selection import RandomizedRecommenderSearchCV, \
    RecommenderGridSearchCV, train_test_split, KFold
from reclab.model_selection._search import _CVWrapper
from reclab.collab import AlternatingLeastSquares, \
    NMSAlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab._config import RECLAB_CACHE, set_blas_singlethread

from sklearn.externals import joblib
from scipy.stats import randint, uniform

import os
import shutil
import warnings

# set this to avoid the MKL BLAS warning
set_blas_singlethread()

lastfm = load_lastfm(cache=True, as_sparse=True)
train, test = train_test_split(lastfm.ratings, random_state=42)


class TestRandomizedSearch:
    def _search_fit_assert(self, search, val=None):
        # Fit it
        search.fit(train, validation_set=val)

        # Show we can score it
        search.score(test)

        # Produce recommendations
        recs, scores = search.recommend_for_user(0, test, n=5,
                                                 return_scores=True)
        assert len(recs) == len(scores) == 5, (recs, scores)

        # Serialize it and show we can load and produce recommendations still
        pkl_loc = "search.pkl"
        try:
            joblib.dump(search, pkl_loc, compress=3)
            joblib.load(pkl_loc).recommend_for_user(
                0, test, n=5, return_scores=True)

        finally:
            os.unlink(pkl_loc)
            if os.path.exists(RECLAB_CACHE):
                shutil.rmtree(RECLAB_CACHE)

    def test_grid_cv_fit_recommend(self):
        # Create the estimator
        clf = NMSAlternatingLeastSquares(random_state=42, use_cg=True,
                                         iterations=5, factors=15)

        # These are the hyper parameters we'll use. Don't use many for
        # the grid search since it will fit every combination...
        hyper = {
            'factors': [5, 6]
        }

        # Make our cv
        cv = KFold(n_splits=2, random_state=1, shuffle=True)
        search = RecommenderGridSearchCV(
            estimator=clf, cv=cv, param_grid=hyper,
            n_jobs=1, verbose=1)

        self._search_fit_assert(search)

    def test_random_cv_fit_recommend(self):
        """Test a simple fit"""
        # Create the estimator
        clf = AlternatingLeastSquares(random_state=42, use_cg=True,
                                      iterations=5, factors=15)

        # These are the hyper parameters we'll use
        hyper = {
            'factors': randint(5, 6),
            'regularization': uniform(0.01, 0.05)
        }

        # Make our cv
        cv = KFold(n_splits=2, random_state=1, shuffle=True)
        search = RandomizedRecommenderSearchCV(
            estimator=clf, cv=cv, random_state=42,
            param_distributions=hyper, n_jobs=1,
            n_iter=2, recommend_params={"filter_previously_rated": True},
            verbose=1, scoring='ndcg')

        # While we're fitting, assert we get a warning about the
        # "filter_previously_rated" key in the fit params...
        with warnings.catch_warnings(record=True) as w:
            self._search_fit_assert(search)  # should warn in fit

            # Verify...
            assert len(w)
            assert any(["filter_previously_rated" in str(warn.message)
                        for warn in w])

    def test_random_val_fit(self):
        """Test a simple fit"""
        # Create the estimator
        clf = AlternatingLeastSquares(random_state=42, use_cg=True,
                                      iterations=5, factors=10)

        # These are the hyper parameters we'll use
        hyper = {
            'factors': randint(5, 6),
            'regularization': uniform(0.01, 0.05)
        }

        # Create search with no CV and use validation set instead
        search = RandomizedRecommenderSearchCV(
            estimator=clf, cv=None, random_state=42,
            param_distributions=hyper, n_jobs=1,
            n_iter=2, verbose=1)

        self._search_fit_assert(search, val=test)


def test_cv_wrapper():
    # Test that the CV wrapper produces exactly what we think it does...
    wrapper = _CVWrapper(cv=None, validation=test)
    split = wrapper.split(train)

    # The split should be a list of a single tuple
    assert isinstance(split, list), split
    assert len(split) == 1, split

    # The tuple element should be len 2
    tup = split[0]
    assert len(tup) == 2, tup
    assert tup[0] is train
    assert tup[1] is test
