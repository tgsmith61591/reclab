# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.model_selection import train_test_split, check_cv, BootstrapCV
from reclab.utils import to_sparse_csr

from numpy.testing import assert_array_almost_equal
from scipy.sparse import issparse
import numpy as np

import pytest


def test_tr_te_split():
    u = [0, 1, 0, 2, 1, 3]
    i = [1, 2, 2, 0, 3, 2]
    r = [0.5, 1.0, 0.0, 1.0, 0.0, 1.]

    train, test = train_test_split(u, i, r, train_size=0.5,
                                   random_state=42)

    # one will be masked in the train array
    assert_array_almost_equal(
        train.toarray(),
        np.array([[0, 0.5, 0, 0],
                  [0, 0, 0, 0],  # masked
                  [1, 0, 0, 0],
                  [0, 0, 1, 0]]))

    assert_array_almost_equal(
        test.toarray(),
        np.array([[0, 0.5, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0]]))


def test_check_cv():
    cv = check_cv(None)
    assert isinstance(cv, BootstrapCV)
    assert cv.n_splits == 3

    cv = check_cv(5)
    assert isinstance(cv, BootstrapCV)
    assert cv.n_splits == 5

    cv = BootstrapCV(n_splits=3, random_state=42)
    cv2 = check_cv(cv)
    assert cv is cv2
    assert cv2.n_splits == 3
    assert cv2.random_state == 42


def test_bad_check_cv():
    with pytest.raises(ValueError) as v:
        check_cv("bad cv")


def test_bootstrap_cv():
    bunch = {
        'users': np.array([ 0,  0,  1,  1,  1,  2,  2,  2,  3,  4,  5,  6,
                            6,  7,  7,  8,  9, 10, 10, 11, 11, 12, 13, 13,
                            13, 14, 14, 14, 15, 16, 17, 17, 18, 19, 20, 21,
                            22, 22, 23, 24, 24, 25, 26, 26, 26, 27, 28, 29,
                            30, 30, 31, 31, 31, 32, 33, 34, 35, 36, 36, 36,
                            37, 37, 37, 37, 37, 37, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 45, 46, 47, 47, 48, 49, 50, 50, 51,
                            52, 52, 53, 54, 54, 55, 56, 56, 57, 58, 58, 58,
                            59, 59, 60, 60]),

        'products': np.array([5, 1, 8, 9, 2, 0, 7, 6, 6, 1, 3, 5, 1, 6, 7,
                              1, 0, 0, 6, 8, 6, 0, 4, 9, 9, 1, 7, 1, 3, 5,
                              9, 8, 7, 3, 7, 8, 7, 6, 2, 6, 2, 1, 4, 3, 8,
                              7, 4, 4, 3, 8, 0, 4, 7, 8, 4, 0, 2, 9, 6, 6,
                              7, 4, 3, 9, 5, 2, 1, 8, 2, 3, 8, 8, 7, 9, 6,
                              7, 1, 8, 8, 9, 1, 4, 3, 7, 2, 0, 5, 9, 4, 7,
                              3, 6, 4, 7, 2, 2, 9, 9, 5, 7]),

        'ratings': np.array([4., 1., 1., 5., 5., 3., 1., 1., 2., 3., 5., 4.,
                             2., 5., 3., 2., 3., 4., 4., 3., 2., 4., 5., 4.,
                             4., 4., 3., 5., 1., 3., 3., 1., 1., 1., 5., 2.,
                             2., 2., 3., 1., 2., 1., 3., 1., 1., 2., 3., 3.,
                             3., 5., 4., 1., 3., 1., 3., 4., 1., 2., 4., 2.,
                             1., 2., 4., 5., 3., 1., 4., 4., 4., 4., 3., 4.,
                             1., 3., 3., 3., 1., 3., 1., 3., 1., 1., 3., 5.,
                             4., 4., 1., 3., 5., 1., 4., 4., 1., 5., 4., 1.,
                             3., 3., 2., 4.])}

    u, i, r = bunch['users'], bunch['products'], bunch['ratings']
    sparse_csr = to_sparse_csr(u=u, i=i, r=r)

    cv = BootstrapCV(n_splits=3, random_state=42)
    splits = list(cv.split(sparse_csr))
    assert len(splits) == 3
    assert all(issparse(x[0]) and issparse(x[1]) for x in splits)

    # expected n stored train elements (all this tests is reproducibility)
    n_train = [77, 74, 77]
    actual_stored = [x[0].nnz for x in splits]
    assert n_train == actual_stored, (n_train, actual_stored)
