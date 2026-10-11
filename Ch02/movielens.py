"""MovieLens loading for the Chapter 2 walkthrough.

Not a chapter section: this is the data plumbing that the experiment harness and
the app both need, kept in one place so the two cannot drift apart. It downloads
MovieLens once, builds the binary interaction matrix `R`, and carries the movie
titles so the app can show recommendations a human can read.

    ml-latest-small   610 users, runs in about a second, the default everywhere
    ml-25m            162,539 users, the configuration the book reports
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def _download(dataset):
    os.makedirs(DATA, exist_ok=True)
    zpath = os.path.join(DATA, dataset + ".zip")
    if not os.path.exists(os.path.join(DATA, dataset)):
        if not os.path.exists(zpath):
            print(f"downloading {dataset} (once) ...")
            urllib.request.urlretrieve(URLS[dataset], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA)


def load(dataset="ml-latest-small", top_items=None, min_item_ratings=20,
         like_threshold=4.0, holdout=0.0, seed=0):
    """Return `(R, titles, info)`.

    `R` is a dense [n_users, n_items] binary matrix: 1 where a user INTERACTED with
    an item, which is what equation 2.1 means by `R[u, i]`. Any rating counts as an
    interaction, whatever its value; `like_threshold` is not used to build `R`. It
    is used only to decide which held-out interactions count as a hit, because
    "did the model recommend something they went on to rate highly" is a fairer
    question than "did it recommend something they went on to rate at all".

    Getting that distinction wrong quietly changes every byte number in the
    chapter: on MovieLens 25M there are about 17.9 million interactions but only
    about 11.6 million ratings of four or more, so building `R` from likes alone
    would put the raw exchange near 93 MB instead of the 143 MB the book reports.

    `titles` is the item-indexed list of movie names. Items are kept only if at
    least `min_item_ratings` users rated them, and `top_items` further restricts to
    the most-rated ones, which is how the book gets its 4,000-item catalogue.

    With `holdout > 0`, that fraction of each user's interactions is removed from
    `R`, and the removed ones that were rated at least `like_threshold` are
    returned in `info["relevant"]`. This is how the book judges a recommender
    honestly: build the model from what it is allowed to see, then ask whether it
    recommends the things it was not shown. The app leaves `holdout` at zero,
    because there it is serving rather than being scored.
    """
    _download(dataset)
    root = os.path.join(DATA, dataset)

    import pandas as pd

    df = pd.read_csv(os.path.join(root, "ratings.csv"),
                     usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(os.path.join(root, "movies.csv"), usecols=["movieId", "title"])
    names = dict(zip(movies.movieId.values, movies.title.values))

    # The catalogue: items with at least `min_item_ratings` ratings, or the
    # `top_items` most-rated items when that is given.
    counts = df.movieId.value_counts()
    keep = counts[counts >= min_item_ratings].index
    if top_items is not None:
        keep = counts.nlargest(top_items).index
    df = df[df.movieId.isin(keep)].copy()

    users = df.userId.unique()
    items = np.sort(df.movieId.unique())
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {i: k for k, i in enumerate(items)}
    df["u"] = df.userId.map(uidx)
    df["i"] = df.movieId.map(iidx)
    n_users, n_items = len(users), len(items)

    # The hold-out: shuffle every interaction once with the seed, then hold out the
    # first `holdout` fraction of each user's interactions (at least one, and only
    # for users with five or more). Every rating is an interaction; the held-out
    # ones rated at least `like_threshold` are the hits.
    relevant = [set() for _ in range(n_users)]
    if holdout > 0.0:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        rank = df.groupby("u").cumcount()
        cnt = df.groupby("u")["u"].transform("size")
        is_test = (rank < np.maximum(1, (holdout * cnt).round())) & (cnt >= 5)
        test = df[is_test & (df.rating >= like_threshold)]
        df = df[~is_test]
        for u, i in zip(test.u.values, test.i.values):
            relevant[u].add(int(i))

    R = np.zeros((n_users, n_items), dtype=np.float32)
    R[df.u.values, df.i.values] = 1.0

    titles = [names.get(i, f"item {i}") for i in items]
    info = {
        "dataset": dataset,
        "n_users": len(users),
        "n_items": len(items),
        "n_interactions": int(len(df)),                  # counted exactly, not as a float32 sum
        "popularity": counts.reindex(items).to_numpy(dtype=np.int64),
        "relevant": relevant,
        "holdout": holdout,
    }
    return R, titles, info


def recall_at_k(scores, relevant, k=50):
    """Held-out recall@k: of the likes the model never saw, what fraction land in
    its top k recommendations. Averaged over the users who had a hold-out at all.
    `scores` is [n_eval_users, n_items] with already-seen items masked out."""
    hits, n = 0.0, 0
    for row, rel in zip(scores, relevant):
        if not rel:
            continue
        top = np.argpartition(-row, min(k, row.size - 1))[:k]
        hits += len(rel & set(int(i) for i in top)) / len(rel)
        n += 1
    return float(hits / n) if n else 0.0
