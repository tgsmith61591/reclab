"""
=====================
Fitting ALS on LastFM
=====================

Demonstrates how to fit a vanilla Alternating Least Squares model on the
reclab-native LastFM dataset with implicit ratings.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.collab import AlternatingLeastSquares as ALS
import numpy as np

# #############################################################################
# Load data and split into train/test
lastfm = load_lastfm(cache=True, as_sparse=True)
train, test = train_test_split(lastfm.ratings, random_state=42)

print("Train:")
print(repr(train))
print("\nTest:")
print(repr(test))

# #############################################################################
# Fit our model
als = ALS(random_state=1, use_gpu=False, use_cg=True,
          iterations=25, factors=100)
als.fit(train)

# #############################################################################
# Generate predictions (on the test set) for a user who is a metal head like me
artists = lastfm.artists
mayhem_id = np.where(artists == "Mayhem")[0][0]
mayhem_listens = train[:, mayhem_id].toarray().ravel()
mayhem_listeners = np.argsort(-mayhem_listens)
mayhem_appreciator = mayhem_listeners[0]  # Has the best taste in music :)
print("\nUser #%i listened to Mayhem %i times.\nThis user's top 5 "
      "most-listened-to artists are:\n%s"
      % (mayhem_appreciator, int(train[mayhem_appreciator, mayhem_id]),
         str(artists[np.argsort(
             -train[mayhem_appreciator, :].toarray())][0, :5])))

# Use the most current ratings matrix and filter ones the user has seen
recommended = als.recommend_for_user(mayhem_appreciator, lastfm.ratings,
                                     return_scores=False, n=5,
                                     filter_previously_rated=True)
mapped_recs = lastfm.artists[recommended]
print("\nUser #%i's top 5 recommended (unheard) artists:\n%s"
      % (mayhem_appreciator, str(mapped_recs)))
