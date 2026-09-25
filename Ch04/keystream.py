"""Chapter 4, section 4.2.1: the reference key stream.

Plumbing for the harness rather than a listing. One stream of 200,000 events is
drawn once from a Zipf distribution over 16,384 keys and then split across sixteen
workers by event, so every worker sees a slice of the same distribution and a hot
key belongs to no single worker. Every table in the chapter is measured against
this stream, or against a variant of it that changes one setting.

    build_stream(...)         the stream, sharded into sixteen unequal batches
    KeyStream.true_hist()     CPU-exact: numpy.bincount over the complete stream
    KeyStream.partial(w)      worker w's contribution, computed by the chapter's
                              kernel (listing 4.4) when it fits, else by NumPy

Settings a variant may change: `alpha` (table 4.3), `num_bins` (table 4.8), and
`correlate_keys_with_speed` (table 4.9), which hands the rare tail of the key
distribution to the workers least likely to arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_WORKERS = 16
NUM_BINS = 16384
N_EVENTS = 200_000
ZIPF_ALPHA = 1.2
HEAVY_K = 32
STREAM_SEED = 20260824

# Unequal batch shares: the largest worker holds about four times the smallest,
# so a quorum of twelve workers is almost never 75 percent of the events.
BATCH_SHARES = (
    0.115, 0.098, 0.091, 0.084, 0.078, 0.071, 0.066, 0.061,
    0.056, 0.051, 0.046, 0.042, 0.038, 0.034, 0.029, 0.040,
)


def zipf_stream(n_events, num_bins, alpha=ZIPF_ALPHA, seed=STREAM_SEED):
    """`n_events` keys in [0, num_bins), Zipf with exponent `alpha`.

    The key at rank r is drawn with probability proportional to 1 / r^alpha, and
    the ranks are then shuffled across the key space, so the hot keys are
    scattered the way hashed identifiers are.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, num_bins + 1)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    keys = rng.choice(num_bins, size=n_events, p=weights)
    perm = rng.permutation(num_bins)
    return perm[keys].astype(np.int64)


@dataclass(frozen=True)
class Worker:
    worker_id: int
    event_count: int        # declared before the epoch opens
    group: int              # 0 for even IDs, 1 for odd; table 4.9 uses it


@dataclass
class KeyStream:
    workers: tuple
    keys: dict = field(repr=False)
    num_bins: int = NUM_BINS
    _partial: dict = field(default_factory=dict, repr=False)
    computed_by: str = "unused"

    def true_hist(self):
        """CPU-exact: the histogram of every event, whoever holds it."""
        allk = np.concatenate([self.keys[w.worker_id] for w in self.workers])
        return np.bincount(allk, minlength=self.num_bins).astype(np.int64)

    def total_events(self):
        return int(sum(w.event_count for w in self.workers))

    def partial(self, worker_id):
        """Worker `worker_id`'s count vector, from listing 4.4 where it can run.

        Above the per-block shared-memory limit listing 4.2 cannot privatize the
        key universe, and the vector is counted on the host instead. The two agree
        exactly wherever both run.
        """
        if worker_id not in self._partial:
            from partial_histogram import gpu_available, privatization_fits
            if privatization_fits(self.num_bins):
                from worker_contribution import worker_contribution
                h = worker_contribution(0, self.keys[worker_id], self.num_bins)
                self.computed_by = ("worker kernel (listing 4.2)" if gpu_available()
                                    else "worker kernel, CPU path")
            else:
                h = np.bincount(self.keys[worker_id],
                                minlength=self.num_bins).astype(np.int64)
                self.computed_by = "host histogram (beyond shared memory)"
            self._partial[worker_id] = h
        return self._partial[worker_id]

    def distinct_keys(self, worker_id):
        return int(np.count_nonzero(self.partial(worker_id)))


def build_stream(num_bins=NUM_BINS, alpha=ZIPF_ALPHA, n_events=N_EVENTS,
                 n_workers=N_WORKERS, seed=STREAM_SEED,
                 correlate_keys_with_speed=False):
    """Draw the stream once, then split it across the workers by event.

    With `correlate_keys_with_speed`, events are scored by how rare their key is,
    plus Gaussian noise with standard deviation 0.15, and dealt out in that order
    with the even-numbered (more available) workers first, so the odd-numbered
    workers receive more of the rare tail.
    """
    rng = np.random.default_rng(seed)

    shares = np.array(BATCH_SHARES[:n_workers], dtype=float)
    shares = shares / shares.sum()
    counts = np.floor(shares * n_events).astype(int)
    counts[-1] += n_events - counts.sum()

    all_keys = zipf_stream(n_events, num_bins, alpha=alpha, seed=seed)

    if correlate_keys_with_speed:
        hist = np.bincount(all_keys, minlength=num_bins)
        rank = np.empty(num_bins, dtype=np.int64)
        rank[np.argsort(hist)[::-1]] = np.arange(num_bins)
        score = rank[all_keys] / float(num_bins) + rng.normal(0.0, 0.15, size=n_events)
        order = np.argsort(score)
        worker_order = ([w for w in range(n_workers) if w % 2 == 0]
                        + [w for w in range(n_workers) if w % 2 == 1])
    else:
        order = rng.permutation(n_events)
        worker_order = list(range(n_workers))

    keys, cursor = {}, 0
    for wid in worker_order:
        take = int(counts[wid])
        keys[wid] = all_keys[order[cursor:cursor + take]]
        cursor += take

    workers = tuple(Worker(w, int(counts[w]), w % 2) for w in range(n_workers))
    return KeyStream(workers=workers, keys=keys, num_bins=num_bins)
