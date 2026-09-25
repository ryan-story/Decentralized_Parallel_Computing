"""Chapter 4, section 4.4.3: merging under an influence limit.

    bounded_merge(contributions, event_counts, tau)    listing 4.9. Merge the
                                                       accepted histograms. With
                                                       `tau=None` the merge is an
                                                       exact sum, as the Quorum
                                                       Histogram does. With a
                                                       `tau`, each worker's count
                                                       in each bin is admitted only
                                                       up to its ceiling, equation
                                                       4.3.

A worker's ceiling for a bin is `tau` times the median per-event rate across the
accepted workers, scaled back to that worker's own batch. Counts above it are
removed; counts below it pass unchanged. The limit bounds how far one worker can
move one bin. It does not make the merged histogram correct, and it removes
honest counts too, from any bin that fewer than half the accepted workers observed.

`contributions` maps worker ID to histogram, and `event_counts` maps worker ID to
the number of events the epoch assigned it. A missing declared count raises a
`KeyError` rather than being filled in, because the denominator sets both the
worker's rate and its ceiling.
"""

from __future__ import annotations

import numpy as np


def bounded_merge(contributions, event_counts, tau):
    worker_ids = sorted(contributions)

    histograms = np.stack([
        contributions[worker_id]
        for worker_id in worker_ids
    ]).astype(np.int64)

    if tau is None:
        return histograms.sum(axis=0)

    totals = np.array([
        event_counts[worker_id]
        for worker_id in worker_ids
    ], dtype=np.float64)

    rates = histograms / totals[:, None]
    median_rate = np.median(rates, axis=0)

    ceilings = np.floor(
        tau * totals[:, None] * median_rate[None, :]
    ).astype(np.int64)

    admitted = np.minimum(histograms, ceilings)

    return admitted.sum(axis=0)
