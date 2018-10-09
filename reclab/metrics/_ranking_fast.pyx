#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#
# This Cython code dramatically speeds up the computation of various relevancy
# metrics for scoring recommenders
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

import numpy as np
import warnings

from cpython cimport bool
cimport numpy as np
from libc.math cimport log2

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef int [:, :] int_array_2d_t
ctypedef long [:, :] long_array_2d_t

ctypedef np.npy_intp INTP
ctypedef np.npy_float FLOAT
ctypedef np.float64_t DOUBLE

cdef fused floating1d:
    float[::1]
    double[::1]

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t

cdef fused intp1d:
    int[::1]
    long[::1]

cdef fused intp_array_2d_t:
    int_array_2d_t
    long_array_2d_t

np.import_array()


# Classes for metrics
cdef class Metric:
    cdef bool assume_unique
    cdef INTP k

    def __cinit__(self, bool assume_unique, INTP k):
        self.assume_unique = assume_unique
        self.k = k

    cdef float compute(self, long[::1] predicted, long[::1] labels):
        """Sub-classes should extend this!!!"""
        return 0.

    cdef float _warn_for_empty_labels(self):
        """Helper for missing ground truth sets"""
        warnings.warn("Empty ground truth set! Check input data")
        return 0.


cdef class PrecisionAtK(Metric):
    cdef float compute(self, long[::1] predicted, long[::1] labels):
        # need to compute the count of the number of values in the predictions
        # that are present in the labels. We'll use numpy in1d for this (set
        # intersection in O(1))
        cdef INTP n_labels = labels.shape[0]
        cdef INTP n_pred = predicted.shape[0]
        cdef INTP n, cnt
        cdef INTP k = self.k
        cdef bool assume_unique = self.assume_unique

        # This is the mask we'll sum up
        # cdef np.ndarray[np.bool, ndim=1, mode='c'] msk
        cdef np.ndarray msk

        if n_labels > 0:
            n = min(n_pred, k)
            msk = np.in1d(predicted[:n], labels, assume_unique=assume_unique)
            cnt = np.sum(msk)
            return float(cnt) / k
        else:
            return self._warn_for_empty_labels()


cdef class MeanAveragePrecision(Metric):
    cdef float compute(self, long[::1] predicted, long[::1] labels):

        cdef INTP n_labels = labels.shape[0]
        cdef INTP present_sum
        cdef bool assume_unique = self.assume_unique

        # compute the number of elements within the predictions that are
        # present in the actual labels, and get the cumulative sum weighted
        # by the index of the ranking
        cdef INTP n = predicted.shape[0]
        cdef np.ndarray[float, ndim=1, mode='c'] arange, denom
        cdef np.ndarray[double, ndim=1, mode='c'] ones, prec_sum
        # cdef np.ndarray[np.bool, ndim=1, mode='c'] present
        cdef np.ndarray present

        if n_labels > 0:
            # Scala code from Spark source:
            # var i = 0
            # var cnt = 0
            # var precSum = 0.0
            # val n = pred.length
            # while (i < n) {
            #     if (labSet.contains(pred(i))) {
            #         cnt += 1
            #         precSum += cnt.toDouble / (i + 1)
            #     }
            #     i += 1
            # }
            # precSum / labSet.size

            arange = np.arange(1, n + 1, dtype=np.float32)  # this is the denom
            present = np.in1d(predicted[:n], labels,
                              assume_unique=assume_unique)

            present_sum = np.sum(present)
            ones = np.ones(present_sum, dtype=float)
            prec_sum = ones.cumsum()
            denom = arange[present]

            return (prec_sum / denom).sum() / float(n_labels)

        else:
            return self._warn_for_empty_labels()


cdef class NDCG(Metric):
    cdef float compute(self, long[::1] predicted, long[::1] labels):
        cdef INTP n_labels = labels.shape[0]
        cdef INTP n_preds = predicted.shape[0]
        cdef INTP n, i
        cdef INTP k = self.k
        cdef bool assume_unique = self.assume_unique

        cdef np.ndarray[float, ndim=1, mode='c'] arange
        cdef np.ndarray[float, ndim=1, mode='c'] denom, gains
        cdef np.ndarray[long, ndim=1, mode='c'] dcg_mask
        cdef double dcg = 0., max_dcg = 0., d, g

        if n_labels:
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                labels = np.unique(labels)
                n_labels = labels.shape[0]

            # compute the gains where the prediction is present in the labels
            n = min(max(n_preds, n_labels), k)  # min(min(p, l), k)?
            dcg_mask = np.in1d(predicted[:n], labels,
                               assume_unique=assume_unique).astype(int)

            for i in range(n):
                d = log2(float(i + 2.))
                g = 1. / d

                if dcg_mask[i] == 1:
                    dcg += g
                if i < n_labels:
                    max_dcg += g

            return dcg / max_dcg

        else:
            return self._warn_for_empty_labels()


cdef _mean_ranking_metric(predictions, labels, Metric metric):
    # Helper function for precision_at_k and mean_average_precision

    # do not zip, as this will require an extra pass of O(N). Just assert
    # equal length and index (compute in ONE pass of O(N)).
    # if len(predictions) != len(labels):
    #     raise ValueError("dim mismatch in predictions and labels!")
    cdef INTP i = 0
    # cdef np.ndarray[int, ndim=1, mode='c'] prd, lab
    cdef np.ndarray lab, prd
    cdef float sm = 0.

    for p in predictions:
        prd = np.asarray(p, dtype=int)  # Not guaranteed to be an array
        lab = np.asarray(labels[i], dtype=int)
        sm += metric.compute(prd, lab)

        # Increment the index for labels
        i += 1

    return sm / float(i)


def _precision_at(predictions, labels, k=10, bool assume_unique=True):
    return _mean_ranking_metric(
        predictions, labels, PrecisionAtK(assume_unique, k))


def _mean_average_precision(predictions, labels, bool assume_unique=True):
    return _mean_ranking_metric(predictions, labels,
                                MeanAveragePrecision(assume_unique, -1))


def _ndcg_at(predictions, labels, INTP k=10, bool assume_unique=True):
    return _mean_ranking_metric(predictions, labels,
                                NDCG(assume_unique, k))
