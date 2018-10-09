# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith
#
# Recommender system ranking metrics derived from Spark source for use with
# Python-based recommender libraries (i.e.,
# implicit: https://github.com/benfred/implicit/)

from __future__ import absolute_import

import numpy as np

from ._ranking_fast import _precision_at, _mean_average_precision, _ndcg_at

__all__ = [
    'mean_average_precision',
    'ndcg_at',
    'precision_at',
]


def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if not isinstance(k, int):
        raise TypeError("K must be an integer")
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def _check_arrays(pred, lab):
    # pred = np.asarray(pred)
    # lab = np.asarray(lab)

    # If not 2d, raise
    # if any(arr.ndim != 2 for arr in (pred, lab)):
    #     raise ValueError("predicted and label arrays must be 2-dimensional")
    return pred, lab


def precision_at(predictions, labels, k=10, assume_unique=True):
    r"""Compute the precision at K.

    Compute the average precision of all the queries, truncated at
    ranking position k. If for a query, the ranking algorithm returns
    n (n is less than k) results, the precision value will be computed
    as #(relevant items retrieved) / k. This formula also applies when
    the size of the ground truth set is less than k.

    If a query has an empty ground truth set, zero will be used as
    precision together with a warning.

        :math:`p(k)=\frac{1}{M}\sum_{i=0}^{M-1}{\frac{1}{k}
        \sum_{j=0}^{\text{min}(\left|D\right|,k)-1}rel_{D_i}(R_i(j))}`

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    k : int, optional (default=10)
        The rank at which to measure the precision.

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)  # doctest: +SKIP
    0.33333333333333331
    >>> precision_at(preds, labels, 5)  # doctest: +SKIP
    0.26666666666666666
    >>> precision_at(preds, labels, 15)  # doctest: +SKIP
    0.17777777777777778
    """
    # validate K
    _require_positive_k(k)
    pred, lab = _check_arrays(predictions, labels)
    return _precision_at(pred, lab, k=k, assume_unique=assume_unique)


def mean_average_precision(predictions, labels, assume_unique=True):
    r"""Compute the mean average precision on predictions and labels.

    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.

        :math:`MAP=\frac{1}{M}\sum_{i=0}^{M-1}{\frac{1}{\left|D_i\right|}
        \sum_{j=0}^{Q-1}\frac{rel_{D_i}(R_i(j))}{j+1}}`

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> mean_average_precision(preds, labels)
    0.35502645502645497
    """
    pred, lab = _check_arrays(predictions, labels)
    return _mean_average_precision(pred, lab, assume_unique=assume_unique)


def ndcg_at(predictions, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.

    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:

        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)

    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.

    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.

    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.

    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).

    k : int, optional (default=10)
        The rank at which to measure the NDCG.

    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.

    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
    0.48791273434956867

    References
    ----------
    .. [1] Mean Average Precision - http://bit.ly/254gS9f
    .. [2] Ranking Systems - http://bit.ly/2zU9LGv
    .. [3] Stanford Rank-Based Measures - http://stanford.io/2yi1pMi
    .. [4] DCG - https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    .. [5] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)
    pred, lab = _check_arrays(predictions, labels)
    return _ndcg_at(pred, lab, k=k, assume_unique=assume_unique)
