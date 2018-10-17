# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.datasets import load_lastfm

from numpy.testing import assert_array_equal
import numpy as np
import pytest

from scipy import sparse


def unpack_bunch(bunch):
    return bunch.users, bunch.products, bunch.ratings


# Last FM tests
class TestLoadLastFM:
    def test_load_bunch(self):
        u, i, r = unpack_bunch(load_lastfm(cache=True))

        # Show the users have NOT been encoded
        for array in (u, i):
            unq = np.sort(np.unique(array))
            with pytest.raises(AssertionError):
                assert_array_equal(unq, np.arange(unq.shape[0]))

    def test_load_sparse(self):
        bunch = load_lastfm(cache=True, as_sparse=True)
        X = bunch.ratings
        assert sparse.issparse(X)

        # Assert on artists
        artists = bunch.artists
        assert isinstance(artists, np.ndarray)

        # Show that the next time we load this after caching, it's the same ref
        assert load_lastfm(cache=True, as_sparse=True) is bunch
