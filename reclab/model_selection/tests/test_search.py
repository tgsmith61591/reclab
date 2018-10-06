# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.model_selection import RandomizedRecommenderSearchCV, \
    RecommenderGridSearchCV, train_test_split, BootstrapCV
from reclab.collab import AlternatingLeastSquares, \
    NMSAlternatingLeastSquares
from reclab.datasets import load_lastfm
from reclab._config import RECLAB_CACHE

from sklearn.externals import joblib
from scipy.stats import randint, uniform

import os
import shutil
import warnings


lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)


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
        clf = NMSAlternatingLeastSquares(random_state=42, use_cg=True)

        # These are the hyper parameters we'll use. Don't use many for
        # the grid search since it will fit every combination...
        hyper = {
            'factors': [5, 6]
        }

        # Make our cv
        cv = BootstrapCV(n_splits=3, random_state=1)
        search = RecommenderGridSearchCV(
            estimator=clf, cv=cv, param_grid=hyper,
            n_jobs=1)

        self._search_fit_assert(search)

    def test_random_cv_fit_recommend(self):
        """Test a simple fit"""
        # Create the estimator
        clf = AlternatingLeastSquares(random_state=42, use_cg=True)

        # These are the hyper parameters we'll use
        hyper = {
            'factors': randint(5, 10),
            'regularization': uniform(0.01, 0.05),
            'iterations': [5, 10, 15]
        }

        # Make our cv
        cv = BootstrapCV(n_splits=3, random_state=1)
        search = RandomizedRecommenderSearchCV(
            estimator=clf, cv=cv, random_state=42,
            param_distributions=hyper, n_jobs=1,
            n_iter=2, recommend_params={"filter_previously_rated": True})

        # While we're fitting, assert we get a warning about the
        # "filter_previously_rated" key in the fit params...
        with warnings.catch_warnings(record=True) as w:
            self._search_fit_assert(search)  # should warn in fit

            # Verify...
            assert len(w)
            assert any(["previously-rated" in str(warn.message)
                        for warn in w])

    def test_random_val_fit(self):
        """Test a simple fit"""
        # Create the estimator
        clf = AlternatingLeastSquares(random_state=42, use_cg=True)

        # These are the hyper parameters we'll use
        hyper = {
            'factors': randint(5, 10),
            'regularization': uniform(0.01, 0.05),
            'iterations': [5, 10]
        }

        # Create search with no CV and use validation set instead
        search = RandomizedRecommenderSearchCV(
            estimator=clf, cv=None, random_state=42,
            param_distributions=hyper, n_jobs=1,
            n_iter=2)

        self._search_fit_assert(search, val=test)
