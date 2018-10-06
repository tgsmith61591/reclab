.. _api_ref:

=============
API Reference
=============

.. include:: ../includes/api_css.rst

This is the class and function reference for reclab. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _collab_ref:

:mod:`reclab.collab`: Collaborative filtering in reclab
=======================================================

The ``reclab.collab`` sub-module defines a number of collaborative filtering
recommender algorithms, including popular matrix factorization techniques and
nearest neighbor methods.

.. automodule:: reclab.collab
    :no-members:
    :no-inherited-members:

Collaborative filtering estimators
----------------------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    reclab.AlternatingLeastSquares
    reclab.AnnoyAlternatingLeastSquares
    reclab.BM25Recommender
    reclab.CosineRecommender
    reclab.ItemItemRecommender
    reclab.NMSLibAlternatingLeastSquares
    reclab.TFIDFRecommender


.. _datasets_ref:

:mod:`reclab.datasets`: Benchmarking ratings data for recommenders
==================================================================

The ``reclab.datasets`` submodule provides several different ratings datasets
used in various examples and tests across the package. If you would like to
prototype a model, this is a good place to find easy-to-access data.

.. automodule:: reclab.datasets
    :no-members:
    :no-inherited-members:

Dataset loading functions
-------------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_lastfm


.. _metrics_ref:

:mod:`reclab.metrics`: Ranking metrics for scoring recommenders
===============================================================

The ``reclab.metrics`` submodule provides several different rankings metrics
that are widely used for benchmarking the efficacy of a recommender algorithm.

.. automodule:: reclab.metrics
    :no-members:
    :no-inherited-members:

Ranking metrics
---------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    metrics.mean_average_precision
    metrics.ndcg_at
    metrics.precision_at


.. _model_selection_ref:

:mod:`reclab.model_selection`: Model selection tools for recommenders
=====================================================================

The ``reclab.model_selection`` submodule provides many utilities for cross-
validating your recommender models, splitting your data into train/test splits
and performing grid searches.

Data splitting
--------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    model_selection.train_test_split

Grid searches
-------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_selection.RandomizedRecommenderSearchCV
    model_selection.RecommenderGridSearchCV


.. _utils_ref:

:mod:`reclab.utils`: Utilities
==============================

Utilities and validation functions used commonly across the package.

.. automodule:: reclab.utils
    :no-members:
    :no-inherited-members:

Validation & array-checking functions
-------------------------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.check_consistent_length
    utils.check_permitted_value
    utils.check_sparse_array
    utils.get_n_factors
    utils.inherit_function_doc
    utils.is_iterable
    utils.safe_mkdirs
    utils.to_sparse_csr

Testing utilities
-----------------

Utilities and base classes for testing modules.

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    utils.RecommenderTestClass
