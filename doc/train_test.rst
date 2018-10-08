.. _train_test:

=================
Train/test splits
=================

As with all machine learning techniques, data splitting is recommended (pun!) when
fitting recommender systems. However, unlike most other machine learning domains, the
train/test split for collaborative filtering is subject to several constraints.

|

Conventional splitting
----------------------

Most data can be randomly split, or split with stratification. A typical train/test split resembles the following:

.. image:: img/train_test.png
   :scale: 50 %
   :alt: Conventional train/test split
   :align: center

Here's how we'd do that in Python:

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)


This is due to the fact that most machine learning algorithms only depend on
consistency of matrix rank, and operate invariant to the number of samples in the
matrix.

|

**Collaborative filtering is not quite the same**

|

Splitting for collaborative filtering
-------------------------------------

In (most) collaborative filtering, we have to ensure that the train set contains
ratings events for all items and users present in the data. This is because matrix
factorization techniques are going to decompose factors for both users (rows), as well as
items (columns). Therefore, our split is subject to more constraints, and ultimately
must appear like:

.. image:: img/collab_split.png
   :scale: 50 %
   :alt: Collaborative filtering train/test split
   :align: center

The challenge is maintaining a replicable, random split with an approximate ratio of
train-to-test while staying within the boundaries of an acceptable split. For instance,
this is an illegal split (we lose user 5 and item 4):

.. image:: img/invalid_split.png
   :scale: 50 %
   :alt: Invalid train/test split
   :align: center

Here's how we can achieve a valid split with reclab:

.. code-block:: python

    from reclab.model_selection import train_test_split
    from reclab.datasets import load_lastfm
    import numpy as np

    lastfm = load_lastfm(cache=True)
    train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                                   r=lastfm.ratings, random_state=42,
                                   train_size=0.8)

    # We can assert that all users and items (artists) made it into the split:
    n_users, n_artists = train.shape  # 1892, 17632
    assert n_users == np.unique(lastfm.users).shape[0]
    assert n_artists == np.unique(lastfm.products).shape[0]

Other nuances
-------------

One of the first things you'll notice after splitting your data with reclab is that
the test set is not smaller than the training set. In fact, the test set contains *all* of
the data! This is due to the fact that recommender systems are not truly supervised learning
techniques; they are information retrieval methods. Therefore, the test set can be considered
a *future* state of the users' ratings, while the training set can be thought of as a *past state*.

The purpose of the test set during scoring is often to determine whether the recommendations
produced by the algorithm actually ended up being consumed/rated positively by the user. Therefore,
the scoring techniques we use for recommenders fall more into the family of information retrieval
and relevancy (think search engine metrics) than conventional "accuracy" measures.

|

Exceptions
----------

There are situations where a conventional train/test split could work for you:

* You don't care about user factors (in which case, you could use the ``recalculate_user``
  option when producing recommendations for users who didn't exist at the time of model fit,
  and a new test matrix that contains the new user(s))
* You are not using a matrix factorization or user-based method (i.e., cosine similarity
  between items)

However, since reclab uses sparse matrices, you'll likely have to perform your own
train/test split on a dense matrix and make it sparse prior to fitting any of the algorithms.
