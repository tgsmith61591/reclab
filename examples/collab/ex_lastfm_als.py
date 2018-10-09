"""
=====================
Fitting ALS on LastFM
=====================

Demonstrates how to fit a vanilla Alternating Least Squares model on the
reclab-native LastFM dataset with implicit ratings.
"""
print(__doc__)

# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.collab import AlternatingLeastSquares as ALS

# #############################################################################
# Load data and split into train/test
lastfm = load_lastfm(cache=True)
train, test = train_test_split(u=lastfm.users, i=lastfm.products,
                               r=lastfm.ratings, random_state=42)

print("Train:")
print(repr(train))
print("\nTest:")
print(repr(test))

# #############################################################################
# Fit our model
als = ALS(factors=64, use_gpu=False, iterations=15)
als.fit(train)

# #############################################################################
# Generate predictions (on the test set) for user 0
recommended = als.recommend_for_user(0, test, return_scores=False, n=5)
mapped_recs = [lastfm.artists[i] for i in recommended]
print("User 0's top 5 recommended artists: %r" % mapped_recs)
