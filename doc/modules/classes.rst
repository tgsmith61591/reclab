.. _api_ref:

=============
API Reference
=============

.. include:: ../includes/api_css.rst

This is the class and function reference for reclab. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. raw:: html

   <br/>


.. _base_ref:

:mod:`reclab.base`: Base classes and functions for reclab
=========================================================

.. automodule:: reclab.base
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

Base recommender class
----------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BaseRecommender

Common-use functions
--------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    base.clone
    base._recommend_items_and_maybe_scores


.. _collab_ref:

:mod:`reclab.collab`: Collaborative filtering in reclab
=======================================================

.. automodule:: reclab.collab
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

Collaborative filtering estimators
----------------------------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    collab.AlternatingLeastSquares
    collab.AnnoyAlternatingLeastSquares
    collab.ItemItemRecommender
    collab.NMSAlternatingLeastSquares


.. _datasets_ref:

:mod:`reclab.datasets`: Benchmarking ratings data for recommenders
==================================================================

.. automodule:: reclab.datasets
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

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

.. automodule:: reclab.metrics
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

Ranking metrics
---------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    metrics.mean_average_precision
    metrics.ndcg_at
    metrics.precision_at


.. _model_deployment_ref:

:mod:`reclab.model_deployment`: Model deployment tools for recommenders
=======================================================================

.. automodule:: reclab.model_deployment
    :no-members:
    :no-inherited-members:

Deploy wrapper
--------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_deployment.RecommenderDeployment


.. _model_selection_ref:

:mod:`reclab.model_selection`: Model selection tools for recommenders
=====================================================================

.. automodule:: reclab.model_selection
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

Data splitting
--------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: function.rst

    model_selection.train_test_split

Cross validation
----------------

.. currentmodule:: reclab

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_selection.KFold

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

.. automodule:: reclab.utils
    :no-members:
    :no-inherited-members:

.. raw:: html

   <br/>

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
