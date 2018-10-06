# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
from os.path import join,  dirname

from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

from ..utils.validation import to_sparse_csr

__all__ = [
    'load_lastfm'
]


class __BunchCache(dict):
    """A cache singleton to hold large datasets.

    Datasets that are expensive to read from disk (i.e., lastfm) might
    want to be cached, especially for unit tests that would otherwise
    require multiple reads.
    """


_cache = __BunchCache()


def _get_or_create_cache_subdict(key):
    sub = _cache.get(key, None)
    if sub is None:
        sub = {}
        _cache[key] = sub
    return sub


def _cache_data(bunch, key, sparse):
    sub_dict = _get_or_create_cache_subdict(key)
    sub_dict[sparse] = bunch


def _get_cached_data(key, sparse, default=None):
    sub_dict = _get_or_create_cache_subdict(key)
    return sub_dict.get(sparse, default)


def load_lastfm(cache=False, as_sparse=False, dtype=np.float32):
    """Load and return the lastFM dataset.

    Load up the last.fm dataset. The entire dataset can be found here:
    http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html

    This returns a bunch of the users, artists (items) and listen count
    (ratings). There is also an array mapping artist index to artist name
    (in ``artists``).

    ================     =========================
    Samples total        92834
    Distinct users       1892
    Distinct artists     17632
    Num. Listens         integer, 1 <= x <= 352698
    ================     =========================

    Parameters
    ----------
    cache : bool, optional (default=False)
        Whether to cache the bunch result locally to avoid multiple re-reads
        from disk in the event of calling ``load_lastfm`` again. Default is
        False.

    as_sparse : bool, optional (default=False)
        Whether to return a sparse CSR array with users along the row axis,
        and items along the column axis.

    dtype : type, optional (default=np.float32)
        The dtype to use for ratings. Default is np.float32, which the
        Implicit library favors.

    Examples
    --------
    >>> from reclab.datasets import load_lastfm
    >>> from reclab.model_selection import train_test_split
    >>> lastfm = load_lastfm()
    >>> train, test = train_test_split(u=lastfm.users, i=lastfm.products,
    ...                                r=lastfm.ratings, random_state=42)
    >>> train
    <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
        with 72389 stored elements in Compressed Sparse Row format>
    >>> test
    <1892x17632 sparse matrix of type '<class 'numpy.float32'>'
        with 92834 stored elements in Compressed Sparse Row format>

    Returns
    -------
    data : Bunch or sparse CSR array
        The loaded data or a sparse CSR array, if ``as_sparse``.
    """
    # if we've already saved it, just return that result and
    # avoid the IO overhead
    cache_key = "lastfm"

    # Get cached data (maybe)
    cached_res = _get_cached_data(cache_key, sparse=as_sparse, default=None)
    if cached_res is not None:
        return cached_res

    # Otherwise unpack it for the first time
    module_path = dirname(__file__)
    base_dir = join(module_path, "data")
    data_filename = join(base_dir, "lastfm_user_artists_playcount.csv.gz")
    data = np.loadtxt(data_filename, delimiter=",", dtype=np.int)
    metadata_filename = join(base_dir, "lastfm_artists.tsv.gz")
    metadata = np.loadtxt(metadata_filename, delimiter="\t", dtype=str)

    # we need to label encode the users and artists in order to prevent
    # any index errors in the train/test splits, etc.
    users = LabelEncoder().fit_transform(data[:, 0])
    items = LabelEncoder().fit_transform(data[:, 1])

    # need to make the join key in the metadata an int...
    ratings = data[:, 2].astype(dtype)
    data = Bunch(users=users,
                 products=items,
                 ratings=ratings,
                 artists=metadata[:, 1])

    # cache if necessary
    if cache:
        _cache_data(data, cache_key, False)

    # we can also cache the sparse
    if as_sparse:
        data = to_sparse_csr(u=users, i=items, r=ratings,
                             axis=0, dtype=dtype)
        if cache:
            _cache_data(data, cache_key, True)

    return data
