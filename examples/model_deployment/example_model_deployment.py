"""
============================
Deploying recommender models
============================

Deploying a recommendation models is not always easy. Typically a users and
products in a ratings matrix don't have positional keys and require a layer of
item or user encoding.

.. raw:: html

   <br/>

The example demonstrates how we can use the deployment wrapper to automatically
handle the encoding of user IDs as well as item IDs. This wrapper can easily
sit behind a REST endpoint to serve recommendations, given the user's rating
history.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

from reclab.datasets import load_lastfm
from reclab.model_selection import train_test_split
from reclab.model_deployment import RecommenderDeployment
from reclab.collab import AlternatingLeastSquares
from reclab.utils import to_sparse_csr
from sklearn.preprocessing import LabelEncoder
import numpy as np

# #############################################################################
# Load data and encode it, then split it
lastfm = load_lastfm(cache=True, as_sparse=False)
users = lastfm.users
items = lastfm.products
ratings = lastfm.ratings
artists = lastfm.artists

# We need to encode the users/items. If you use as_sparse=True, they come
# pre-encoded, but we will do it here manually for example.
user_le = LabelEncoder()
item_le = LabelEncoder()
users_transformed = user_le.fit_transform(users)
items_transformed = item_le.fit_transform(items)

# Split the data
X = to_sparse_csr(u=users_transformed,
                  i=items_transformed,
                  r=ratings, axis=0, dtype=np.float32)
train, test = train_test_split(X, train_size=0.75, random_state=42)

# #############################################################################
# Fit our model, make our deployment object
als = AlternatingLeastSquares(
    random_state=42, use_gpu=False, use_cg=True,
    iterations=50, factors=100)
als.fit(train)

# This is what you'd persist:
wrapper = RecommenderDeployment(
    estimator=als, user_missing_strategy="error",

    # These are optional, and can be None if you don't want transformed recs
    item_encoder=item_le, user_encoder=user_le)

# #############################################################################
# Generate predictions for a fan of classic rock

def top_listener(of):
    musician_id = [i for i, v in artists.items() if v == of][0]
    listen_mask = items == musician_id
    musician_listens = ratings[listen_mask]
    musician_listeners = users[listen_mask]
    sorted_listeners = np.argsort(-musician_listens)
    return musician_listeners[sorted_listeners[0]]

# Get the top listener of the band
rock_listener = top_listener("Led Zeppelin")

# Get the ratings for a user in a dictionary.
def ratings_for_user(user):
    mask = users == user
    return dict(zip(items[mask], ratings[mask]))

# User the wrapper to get a recommendation for the user based on all of his/her
# ratings to date. NOTE:
#
#   * The userid is NOT encoded!
#   * The item IDs are NOT encoded!
#
# These are handled by the wrapper
recs = wrapper.recommend_for_user(userid=rock_listener,
                                  ratings=ratings_for_user(rock_listener),
                                  n=5, filter_previously_rated=False)

# We can map these back to the artists, if we so choose. Since the recs are
# already inverse-transformed, we can look them up in the artist dictionary.
mapped_recs = [artists[i] for i in recs]
print("User #%i's top 5 recommended artists: %s"
      % (rock_listener, str(mapped_recs)))
