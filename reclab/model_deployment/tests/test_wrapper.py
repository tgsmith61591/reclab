# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.model_deployment import RecommenderDeployment
from reclab.collab import AlternatingLeastSquares

from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from numpy.testing import assert_array_equal

import pytest
import warnings
import os

# Load data and split into train/test
lastfm = load_lastfm(cache=True, as_sparse=True)
train, test = train_test_split(lastfm.ratings, random_state=42)


class TestRecommenderDeployment(object):
    def test_simple_deployment(self):
        als = AlternatingLeastSquares(factors=10, use_cg=False, iterations=3)
        als.fit(train)
        recs1 = als.recommend_for_user(0, test)

        deployment = RecommenderDeployment(estimator=als)
        recs2 = deployment.recommend_for_user(0, test[0, :].toarray()[0])
        assert_array_equal(recs1, recs2)

    def test_encoded_deployment(self):
        users = ['adam', 'betty', 'betty', 'frank', 'frank']
        items = ["chili's", "chuy's", "chili's", "torchy's", "chuy's"]
        visits = [2, 4, 1, 8, 5]

        # Encode the labels
        user_le = LabelEncoder()
        item_le = LabelEncoder()
        users = user_le.fit_transform(users)
        items = item_le.fit_transform(items)

        # Make the matrix (don't bother splitting for this example)
        R = sparse.csr_matrix((visits, (users, items)), shape=(3, 3))
        als = AlternatingLeastSquares(factors=2, use_cg=False, iterations=5)
        als.fit(R)
        recs1 = als.recommend_for_user(0, R)

        # Test failing constructors first
        with pytest.raises(TypeError):
            RecommenderDeployment(
                estimator=als, item_encoder='bad_encoder',
                user_encoder=user_le,
                user_missing_strategy='error')
        with pytest.raises(TypeError):
            RecommenderDeployment(
                estimator=als, item_encoder=item_le,
                user_encoder='bad_encoder',
                user_missing_strategy='error')
        with pytest.raises(TypeError):
            RecommenderDeployment(
                estimator=als, item_encoder=item_le,
                user_encoder=user_le,
                filter_items='non-iterable',
                user_missing_strategy='error')
        with pytest.raises(ValueError):
            RecommenderDeployment(
                estimator=als, item_encoder=item_le,
                user_encoder=user_le,
                user_missing_strategy='bad-strategy')

        # "deploy" with both encoders
        deployment = RecommenderDeployment(
            estimator=als, item_encoder=item_le, user_encoder=user_le,
            user_missing_strategy='error')
        recs2 = deployment.recommend_for_user('adam', R[0, :].toarray()[0])

        # Show that the encoded recs are the same as before
        assert_array_equal(recs1, item_le.transform(recs2))

        # What if we pass a dict?
        recs3 = deployment.recommend_for_user('adam', {"chili's": 2})
        assert_array_equal(recs1, item_le.transform(recs3))

        # And if we want scores?
        recs4, scores = deployment.recommend_for_user(
            'adam', R[0, :].toarray()[0], return_scores=True)
        assert_array_equal(recs1, item_le.transform(recs4))
        assert scores.shape[0] == recs4.shape[0]

        # Test the persistence model
        pkl_location = "model.pkl"
        try:
            joblib.dump(deployment, pkl_location, compress=3)
            loaded = joblib.load(pkl_location)
            recs5 = loaded.recommend_for_user('adam', R[0, :].toarray()[0])
            assert_array_equal(recs1, item_le.transform(recs5))
        finally:
            os.unlink(pkl_location)

        # If we set the user_encoder to None, show we get the same
        # recommendations with a non-encoded user ID
        deployment.user_encoder = None
        recs_no_encode = deployment.recommend_for_user(0, R[0, :].toarray()[0])
        assert_array_equal(recs1, item_le.transform(recs_no_encode))

        # Oh, and now we fail with a TypeError if we pass a string since it
        # never gets transformed
        with pytest.raises(TypeError):
            deployment.recommend_for_user('adam', R[0, :].toarray()[0])

        # What if we give it a user that doesn't exist? Or a negative one?
        with pytest.raises(KeyError):
            deployment.recommend_for_user(9, R[0, :].toarray()[0])
        with pytest.raises(KeyError):
            deployment.recommend_for_user(-1, R[0, :].toarray()[0])

        # Show we fail with improper dims
        with pytest.raises(ValueError):
            deployment.recommend_for_user(0, [2.])

        # Now set the item encoder to none
        deployment.item_encoder = None
        recs_no_encode_anything = deployment.recommend_for_user(
            0, {0: 2})
        assert_array_equal(recs1, recs_no_encode_anything)

        # Set it to "warn" and try again
        deployment.user_missing_strategy = "warn"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # execute the fxn
            recs = deployment.recommend_for_user(9, R[0, :].toarray()[0])
            assert len(w)  # assert there's something there...
            assert recs.shape[0] == 0

            # do the same with return_scores
            recs, scores = deployment.recommend_for_user(
                9, R[0, :].toarray()[0], return_scores=True)
            assert recs.shape[0] == scores.shape[0] == 0
