
===================================================
Reclab: A practical library for recommender systems
===================================================

.. raw:: html

   <!-- Block section -->
   <script src="_static/js/jquery.min.js"></script>

   <a href="https://travis-ci.org/tgsmith61591/reclab"><img alt="Travis status" src="https://travis-ci.org/tgsmith61591/reclab.svg?branch=master" /></a>
   <a href="https://codecov.io/gh/tgsmith61591/reclab"><img alt="Coverage" src="https://codecov.io/gh/tgsmith61591/reclab/branch/master/graph/badge.svg" /></a>
   <a href="https://github.com/tgsmith61591/reclab"><img id="nutrition" alt="gluten free" src="https://img.shields.io/badge/gluten_free-100%25-brightgreen.svg" /></a>

There are a lot of really great libraries out there for creating effective
recommender systems in Python. However, not many are easy to use, and the literature
and code availability of evaluating recommenders is few and far-between.

Reclab's aim is to provide a centralized hub of easy-to-use, serializable recommender
estimators as well as model selection tools such as data splitting, cross-validation and
model selection.

|

Example: Build a matrix factorization model on LastFM
-----------------------------------------------------

In this example, we'll build an Alternating Least Squares on the implicit ratings
of the LastFM dataset.

|

.. code-block:: python


   from reclab.datasets import load_lastfm
   from reclab.model_selection import train_test_split
   from reclab.collab import AlternatingLeastSquares as ALS
   import numpy as np

   # Load data and split into train/test (train/test splits for collaborative
   # filtering are a bit different than for other applications)
   lastfm = load_lastfm()
   train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                                  r=lastfm.ratings, random_state=42)

   # Fit our model
   als = ALS(factors=64, use_gpu=False, iterations=15)
   als.fit(train)

   # Generate predictions (on the test set) for user 0
   recommended_artists = als.recommend_for_user(
       0, test, n=5, return_scores=False)

   # Map the artist keys back to their actual names
   mapped_recs = [lastfm.artists[i] for i in rec_items]
   print("User 0's top 5 recommendations: %r" % mapped_recs)


Look how easy that was!

.. raw:: html

   <br/>

Quick refs, indices and tables
==============================

Helpful quickstart sections:

* :ref:`about`
* :ref:`setup`
* :ref:`contrib`
* :ref:`api_ref`

To search for specific sections or class documentation, visit the index.

* :ref:`genindex`
