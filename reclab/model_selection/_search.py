# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Provides an interface for selecting the optimal hyper-parameters for
# a recommender model. Then fit the parameters on the entire train set.

from __future__ import print_function, division, absolute_import

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection._search import ParameterSampler, ParameterGrid
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.externals.joblib import delayed, Parallel
from sklearn.externals import six

import numpy as np

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import product

import warnings
import time

from ..base import BaseRecommender
from ..metrics.ranking import ndcg_at, mean_average_precision, precision_at
from ..utils.validation import check_permitted_value
from ..utils.decorators import inherit_function_doc
from ._split import check_cv
# from .._config import set_blas_singlethread

__all__ = ['RandomizedRecommenderSearchCV',
           'RecommenderGridSearchCV']

_valid_metrics = {'mean_average_precision': mean_average_precision,
                  'precision_at_k': precision_at,
                  'ndcg': ndcg_at}


class _CVResults(namedtuple('_ScoreResults',
                            ('params', 'test_scores', 'fit_times'))):
    # This named tuple is lighter to pickle than a full class. This is
    # how scikit-learn handles their CV results as well
    def __repr__(self):
        return "Params: {0}, Test scores: {1}, Fit times (secs): {2}".format(
            self.params, self.test_scores, self.fit_times)


def _compute_score(scorer, X, recommendations, **scoring_kwargs):
    rated_items = np.split(X.indices, X.indptr)[1:-1]

    # produce the score and return it
    return scorer(predictions=recommendations, labels=rated_items,
                  **scoring_kwargs)


def _fit_and_score(estimator, train, val, parameters, verbose, metric,
                   param_num, recommend_kwargs, scoring_kwargs):

    # set the parameters
    estimator.set_params(**parameters)

    # fit a model and get the score on the validation set
    start = time.time()
    estimator.fit(train)
    fit_time = time.time() - start

    # The recommendations are a predicted FUTURE state of a user's ratings.
    # Therefore, we need to make sure to create FUTURE recommendations based
    # on the PAST state, which in this case is TRAIN. PREDICT from TRAIN, but
    # filter previously-rated (i.e., training ratings) such that we evaluate
    # only the VAL ratings (truth) against the recommendations.
    recs = estimator.recommend_for_all_users(
        train, filter_previously_rated=True,  # to eval test vs. new preds
        return_scores=False,
        **recommend_kwargs)  # this is a generator

    score_start = time.time()
    score = _compute_score(metric, val, recs, **scoring_kwargs)
    score_time = time.time() - score_start

    if verbose:
        print("[%i] - Fit fold for parameter set %i in %.4f seconds "
              "(%s=%.5f, predict & score time=%.5f)"
              % (param_num, param_num, fit_time, metric.__name__,
                 score, score_time))

    # namedtuple is a lightweight way to ship this back
    return parameters, score, fit_time


class _CVWrapper(object):
    """Wrapper class for either CV or train/validation evaluation.

    This is just a hacky wrapper that allows us to treat train/val splits in
    the same exact fashion as we'd treat cross-val splits. Train/val splits can
    be considered single splits that will be returned in a list with a single
    tuple.

    Notes
    -----
    The ``split`` function behaves like the cross validation API would. For
    either case, it returns a list of tuples, where for a true CV split, the
    list will be longer than 1 element, with tuples of length 2 (train & val
    matrices respectively)::

        [(train_1, val_1), (train_2, val_2), ..., (train_n, val_n)]

    For a train/validation split, will also return a  list but with only one
    tuple::

        [(train, val)]
    """
    def __init__(self, cv, validation):
        self.cv = check_cv(cv) if validation is None else None
        self.validation = validation

    def get_n_splits(self):
        """Get the number of splits for the CV class"""
        if self.cv is not None:
            return self.cv.get_n_splits()
        return 1

    def split(self, X):
        if self.cv is not None:
            return self.cv.split(X)
        # Otherwise it's just single-fold splitting basically
        return [(X, self.validation)]


class _BaseRecommenderSearchCV(six.with_metaclass(ABCMeta, BaseRecommender)):
    """Abstract base class for all recommender grid searches with or without
    cross-validation.
    """

    def __init__(self, estimator, scoring, cv, recommend_params,
                 scoring_params, n_jobs, verbose):

        # Call to super to get the model key for the search, though this is
        # pretty unnecessary...
        super(_BaseRecommenderSearchCV, self).__init__()

        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.recommend_params = recommend_params
        self.scoring_params = scoring_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def _get_param_iterator(self):
        """Get the parameter grid or sampler

        Returns
        -------
        grid : iterable
            The parameter space that will be searched.
        """

    def fit(self, X, validation_set=None):
        r"""Fit a grid search.

        Fit the search instance against the sparse ratings matrix, and select
        the optimal hyper parameters that maximize the scoring metric. Finally,
        refit the optimal parameters against the entire training set.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape=(n_samples, n_products)
            The sparse ratings matrix.

        validation_set : scipy.sparse.csr_matrix, optional (default=None)
            Recommender fits can take a long time. You don't *have* to use
            cross-validation, though it's highly recommended. If a validation
            set is provided, the score procedure will only evaluate each model
            against the validation set. The optimal hyper parameters will also
            *not* be refit, since we can simply return the best-fit model at
            the time.

        Notes
        -----
        * If ``validation_set`` is provided, CV will not be used
        * If "filter_previously_rated" is present in ``recommend_params``, it
          will be popped and a warning will be issued, as previously-rated
          items are required to be present in for most scoring metrics to show
          any lift.
        """
        # validate the CV
        cv = _CVWrapper(self.cv, validation_set)
        n_jobs = self.n_jobs
        n_splits = cv.get_n_splits()

        # get the estimator, make a clone
        est = clone(self.estimator)
        if hasattr(est, "num_threads") and \
                est.num_threads != 1 and \
                n_jobs != 1:
            warnings.warn("Parallelism set in both grid search as well as "
                          "estimator model (%s). Setting grid search "
                          "parallelism to 1 in favor of keeping algorithm "
                          "parallelized" % est.__class__.__name__)
            n_jobs = 1

        # if n_jobs is still parallel, set the MKL var
        # TODO: should we do this?...
        # if n_jobs != 1:
        #     # Set single threaded if we're using parallel
        #     set_blas_singlethread()

        # get the scoring metric
        scorer = check_permitted_value(_valid_metrics, self.scoring)

        # iterate each set of hyper parameters, fit a model and score the model
        # then sort it by the prescribed scoring metric (desc) with the shorter
        # time for fit being the tie-breaker
        candidate_params = list(self._get_param_iterator())

        # Get args for scoring and recommending
        rec_kwargs = self.recommend_params
        score_kwargs = self.scoring_params
        if not rec_kwargs:
            rec_kwargs = dict()

        filter_keys = ("filter_previously_rated", "return_scores")
        for filter_key in filter_keys:
            if filter_key in rec_kwargs:
                warnings.warn("%s cannot be set in grid search recommend "
                              "kwargs, as the scoring behavior is very "
                              "specific to the search process." % filter_key)
                rec_kwargs.pop(filter_key)

        if not score_kwargs:
            score_kwargs = dict()

        verbose = self.verbose
        n_iter = len(candidate_params)
        if verbose:
            print("Fitting {0} iterations with {1} CV fold(s), totalling {2} "
                  "fits (plus one refit at the end)"
                  .format(n_iter, n_splits, n_iter * n_splits))

        out = Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_score)(
                clone(est), train, test, parameters, verbose=verbose,
                metric=scorer, param_num=i // n_splits,
                recommend_kwargs=rec_kwargs,
                scoring_kwargs=score_kwargs)

            for i, (parameters, (train, test)) in
            enumerate(product(candidate_params, cv.split(X))))

        # consolidate the results
        def unpack_results(res):
            params, scores, times = list(zip(*res))
            return _CVResults(params=params[0], test_scores=np.asarray(scores),
                              fit_times=np.asarray(times))

        # TODO: fix???
        results = [unpack_results(out[i: i + n_splits])
                   for i in range(0, len(out), n_splits)]

        # sort based on the score descending (time is the tie-breaker)
        results = sorted(results,
                         key=(lambda r: (np.average(r.test_scores),
                                         -np.average(r.fit_times))),
                         reverse=True)

        # refit on the entire train set
        best_results = results[0]
        best_params = best_results.params

        # We could eventually avoid a re-fit for instances using a
        # validation_set, but for now I don't want to deal with the overhead
        # of passing around a bunch of model objects, so just refit...
        est = clone(est)  # re-clone
        est.set_params(**best_params)
        est.fit(X)  # fit on the entire train set

        # now we have our best model, params, etc
        self.search_results_ = results
        self.best_estimator_ = est
        self.best_params_ = best_params

        return self

    @inherit_function_doc(BaseRecommender)
    def n_users(self):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.n_users()

    @inherit_function_doc(BaseRecommender)
    def n_items(self):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.n_items()

    @inherit_function_doc(BaseRecommender)
    def recommend_for_user(self, userid, R, n=10, filter_previously_rated=True,
                           filter_items=None, return_scores=False, **kwargs):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.recommend_for_user(
            userid, R, n=n, filter_previously_rated=filter_previously_rated,
            filter_items=filter_items, return_scores=return_scores, **kwargs)

    def score(self, X, recommend_kwargs=None, **score_kwargs):
        r"""Compute the score on the test set.

        Generate recommendatinos for all users and compute the score for the
        estimator using the provided ``scoring`` metric.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape=(n_samples, n_products)
            The sparse ratings matrix.

        recommend_kwargs : dict or None
            Any keyword args to pass to the ``recommend_for_all_users``
            function.

        **score_kwargs : dict or None, optional (default=None)
            Any keyword args to pass to the scoring function.
        """
        check_is_fitted(self, "best_estimator_")
        scorer = check_permitted_value(_valid_metrics, self.scoring)

        # We have to make sure we pass this as a dict and not None!
        if not recommend_kwargs:
            recommend_kwargs = dict()

        # This is a generator of all the users' recommendations. Expensive to
        # iterate...
        recs = self.recommend_for_all_users(
            X, return_scores=False,
            **recommend_kwargs)
        return _compute_score(scorer, X, recs, **score_kwargs)


class RecommenderGridSearchCV(_BaseRecommenderSearchCV):
    """Exhaustive search on hyper parameters.

    Discover the best hyper parameters for a recommender model fit by searching
    exhaustively over the given parameter space. The parameters of the
    estimator are optimized by cross-validated search over parameter settings,
    with the parameters selected being those that maximize the mean score of
    the held-out cross validation folds, according to the ``scoring`` method.

    See notes for exceptions.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator we want to fit

    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values. This enables searching over
        any sequence of parameter settings.

    scoring : str or unicode, optional (default='mean_average_precision')
        The scoring metric used to select the optimal hyper
        parameters. Default is 'map', but can be also one of
        ('precision_at_k', 'ndcg')

    cv : BaseCrossValidator, int, or None, optional (default=3)
        The cross validation class instance. If None or an integer, KFold
        cross validation will be used by default (unless fit with a
        ``validation_set``)

    recommend_params : dict or None, optional (default=None)
        Any keyword args to pass to the ``recommend_for_all_users`` function.

    scoring_params : dict or None, optional (default=None)
        Any keyword args to pass to the scoring function.

    n_jobs : int, optional (default=1)
        The parallelism of the fit. Default is 1 (all in one thread).
        If n_jobs is not equal to 1, the MKL_BLAS environment variable will
        be set to 1 to avoid internal multi-threading.

    Notes
    -----
    * If a ``validation_set`` is provided to the ``fit`` method, the validation
      set will be used as the held-out fold for each fit, and CV will not be
      performed. This is not always recommended, but for extremely large data
      may be necessary.
    """
    def __init__(self, estimator, param_grid, scoring='mean_average_precision',
                 cv=3, recommend_params=None, scoring_params=None, n_jobs=1,
                 verbose=0):

        super(RecommenderGridSearchCV, self).__init__(
            estimator=estimator, cv=cv, scoring=scoring,
            recommend_params=recommend_params,
            scoring_params=scoring_params, n_jobs=n_jobs,
            verbose=verbose)

        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given distributions"""
        return ParameterGrid(self.param_grid)


class RandomizedRecommenderSearchCV(_BaseRecommenderSearchCV):
    """Randomized search on hyper parameters.

    Discover the best hyper parameters for a recommender model fit by randomly
    searching over the given parameter space. The parameters of the estimator
    are optimized by cross-validated search over parameter settings, with the
    parameters selected being those that maximize the mean score of the held-
    out cross validation folds, according to the ``scoring`` method.

    In contrast to :class:`RecommenderGridSearchCV`, not all parameter values
    are tried out, but rather a fixed number of parameter settings is sampled
    from the specified distributions. The number of parameter settings that are
    tried is given by ``n_iter``.

    See notes for exceptions.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator we want to fit

    param_distributions : dict
        Dictionary where the keys are parameters and values
        are distributions from which a parameter is to be sampled.
        Distributions either have to provide a ``rvs`` function
        to sample from them, or can be given as a list of values,
        where a uniform distribution is assumed.

    n_iter : int, optional (default=10)
        The number of models to fit. This is not to be confused with
        the ``n_iter`` parameter also used in the ALS ``fit``, which
        corresponds to the number alternating least squares iterations
        to perform.

    scoring : str or unicode, optional (default='mean_average_precision')
        The scoring metric used to select the optimal hyper
        parameters. Default is 'map', but can be also one of
        ('precision_at_k', 'ndcg')

    cv : BaseCrossValidator, int, or None, optional (default=3)
        The cross validation class instance. If None or an integer, KFold
        cross validation will be used by default (unless fit with a
        ``validation_set``)

    recommend_params : dict or None, optional (default=None)
        Any keyword args to pass to the ``recommend_for_all_users`` function.

    scoring_params : dict or None, optional (default=None)
        Any keyword args to pass to the scoring function.

    random_state : int or None (default=None)
        The random state seed to control initial parameter selection
        and the X, Y matrix initialization in ALS.

    n_jobs : int, optional (default=1)
        The parallelism of the fit. Default is 1 (all in one thread).
        If n_jobs is not equal to 1, the MKL_BLAS environment variable will
        be set to 1 to avoid internal multi-threading.

    Notes
    -----
    * If a ``validation_set`` is provided to the ``fit`` method, the validation
      set will be used as the held-out fold for each fit, and CV will not be
      performed. This is not always recommended, but for extremely large data
      may be necessary.
    """
    def __init__(self, estimator, param_distributions, n_iter=10,
                 scoring='mean_average_precision', cv=3, recommend_params=None,
                 scoring_params=None, n_jobs=1, verbose=0, random_state=None):

        super(RandomizedRecommenderSearchCV, self).__init__(
            estimator=estimator, cv=cv, scoring=scoring,
            recommend_params=recommend_params,
            scoring_params=scoring_params, n_jobs=n_jobs,
            verbose=verbose)

        self.random_state = random_state
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=check_random_state(self.random_state))
