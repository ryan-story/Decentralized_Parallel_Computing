"""Chapter 4, section 4.4.2: a histogram contribution and its admission check.

    HistogramContribution                              listing 4.7. The worker's
                                                       histogram, plus the event
                                                       total it claims to have
                                                       counted.
    validate_histogram(c, num_bins, expected_total)    listing 4.8. The five
                                                       checks a coordinator can
                                                       make from the contribution
                                                       and the epoch's
                                                       declarations alone.

Validation establishes that a contribution is well formed and internally
consistent. It does not establish that each event landed in the right bin: a
worker can move counts between bins and keep its total, and every check still
passes. Bounding that case is `bounded_merge`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HistogramContribution:
    worker_id: int
    histogram: np.ndarray
    claimed_total: int


def validate_histogram(c, num_bins, expected_total):
    h = np.asarray(c.histogram)

    if h.shape != (num_bins,):
        return False
    if not np.issubdtype(h.dtype, np.integer):
        return False
    if np.any(h < 0):
        return False
    if int(h.sum()) != c.claimed_total:
        return False
    if c.claimed_total != expected_total:
        return False
    return True
