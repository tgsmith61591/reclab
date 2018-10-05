# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.utils.validation import check_consistent_length, \
    is_iterable, to_sparse_csr, check_permitted_value, check_sparse_array, \
    get_n_factors

from numpy.testing import assert_array_almost_equal
from scipy import sparse
import numpy as np

import pytest

# Multiple use test data
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = to_sparse_csr(u=row, i=col, r=data, axis=0)


def test_check_consistent_length():
    u = np.arange(5)
    i = np.arange(5)
    r = np.arange(5)

    # show they come back OK with u, i, r as the same refs
    users, items, ratings = check_consistent_length(u, i, r)
    assert u is users
    assert i is items
    assert ratings is r  # dtype does not change like it used to

    # change len of one
    i = np.arange(3)
    with pytest.raises(ValueError):
        check_consistent_length(u, i, r)


def test_is_iterable():
    assert is_iterable([1, 2, 3])
    assert is_iterable({1, 2, 3})
    assert is_iterable((1,))
    assert not is_iterable("some string")
    assert not is_iterable(None)
    assert not is_iterable(123)


def test_to_sparse_csr():
    assert sparse.issparse(csr)
    assert csr.nnz == 6, csr  # num stored
    assert_array_almost_equal(csr.toarray(),
                              np.array([[1, 0, 2],
                                        [0, 0, 3],
                                        [4, 5, 6]]))

    # show what happens if we use the diff axis (it's .T basically)
    csrT = to_sparse_csr(u=row, i=col, r=data, axis=1)
    assert sparse.issparse(csrT)
    assert csrT.nnz == 6, csrT
    assert_array_almost_equal(csr.toarray(),
                              csrT.T.toarray())

    # test failing
    with pytest.raises(ValueError):
        to_sparse_csr(row, col, data, axis=2)


def test_check_valid_mapping():
    d = {1: 2, 2: 3}
    assert check_permitted_value(d, 1) == 2
    with pytest.raises(KeyError):
        check_permitted_value(d, 4)


def test_check_sparse_array():
    csr_copy = csr.astype(np.float32)
    checked = check_sparse_array(csr_copy, dtype=np.float32, copy=False)
    assert checked is csr_copy

    # If copy is true it's not the same...
    checked2 = check_sparse_array(csr_copy, dtype=np.float32, copy=True)
    assert checked2 is not csr_copy
    assert_array_almost_equal(checked2.todense(), csr_copy.todense())

    # If it's not a CSR array we bomb out
    with pytest.raises(ValueError):
        check_sparse_array(csr_copy.todense())


def test_get_n_factors():
    assert get_n_factors(100, 10) == 10
    assert get_n_factors(100, 0.005) == 1

    # Fails if it's an int
    with pytest.raises(ValueError):
        get_n_factors(100, 0)

    # Fails if it's a float
    with pytest.raises(ValueError):
        get_n_factors(100, 0.)

    with pytest.raises(TypeError):
        get_n_factors(100, None)

    # Show it's ok if n_factors > 1.
    assert get_n_factors(100, 1.5) == 150
