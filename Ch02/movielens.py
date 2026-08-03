"""MovieLens loading for the Chapter 2 walkthrough.

Not a chapter section: this is the data plumbing that the experiment harness and
the app both need, kept in one place so the two cannot drift apart. It downloads
MovieLens once, builds the binary interaction matrix `R`, and carries the movie
titles so the app can show recommendations a human can read.

    ml-latest-small   610 users, runs in about a second, the default everywhere
    ml-25m            162,539 users, the configuration the book reports
"""

from __future__ import annotations

import csv
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

    # Every rating is an interaction, whatever its value. The threshold decides
    # only what counts as a HIT among the held-out ones.
    ratings = []
    with open(os.path.join(root, "ratings.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ratings.append((int(row[0]), int(row[1]), float(row[2])))

    names = {}
    with open(os.path.join(root, "movies.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            names[int(row[0])] = row[1]

    counts = {}
    for _, i, _rating in ratings:
        counts[i] = counts.get(i, 0) + 1
    keep = [i for i, c in counts.items() if c >= min_item_ratings]
    if top_items is not None:
        keep = sorted(keep, key=lambda i: -counts[i])[:top_items]
    keep_set = set(keep)

    items = sorted(keep_set)
    iidx = {i: k for k, i in enumerate(items)}
    users = sorted({u for u, i, _ in ratings if i in keep_set})
    uidx = {u: k for k, u in enumerate(users)}

    R = np.zeros((len(users), len(items)), dtype=np.float32)
    liked = np.zeros_like(R, dtype=bool)
    for u, i, rating in ratings:
        if i in keep_set:
            R[uidx[u], iidx[i]] = 1.0
            if rating >= like_threshold:
                liked[uidx[u], iidx[i]] = True

    relevant = [set() for _ in users]
    if holdout > 0.0:
        rng = np.random.default_rng(seed)
        for u in range(R.shape[0]):
            inter = np.flatnonzero(R[u])
            if inter.size < 5:                     # too few to split meaningfully
                continue
            n_out = max(1, int(round(holdout * inter.size)))
            held = rng.choice(inter, size=n_out, replace=False)
            R[u, held] = 0.0                       # the model never sees these
            # only the held-out interactions the user actually liked count as hits
            relevant[u] = set(int(i) for i in held if liked[u, i])

    titles = [names.get(i, f"item {i}") for i in items]
    info = {
        "dataset": dataset,
        "n_users": len(users),
        "n_items": len(items),
        "n_interactions": int(R.sum()),
        "popularity": np.array([counts[i] for i in items], dtype=np.int64),
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
