"""Chapter 4, section 4.3.1: from a local result to a worker contribution.

    worker_contribution(device_id, shard, num_bins)    listing 4.4. Bind one
                                                       worker's work to one GPU,
                                                       count its shard with
                                                       listing 4.3, and copy the
                                                       merged histogram to the
                                                       host as an int64 vector.

Every worker uses the same `num_bins` and the same key-to-bin mapping, so position
`b` means the same bin in every contribution and the vectors merge by element-wise
addition.

Deviation from the book's listing, and why:

  * Without CuPy or a CUDA device there is no device to bind, so the function
    calls `partial_histogram`'s CPU path directly. The returned vector is the same.
"""

from __future__ import annotations

import numpy as np

from partial_histogram import gpu_available, partial_histogram

if gpu_available():
    import cupy as cp


def worker_contribution(device_id, shard, num_bins):
    if not gpu_available():
        _, worker_hist = partial_histogram(shard, num_bins)
        return worker_hist.get().astype(np.int64)

    with cp.cuda.Device(device_id):
        _, worker_hist = partial_histogram(
            shard,
            num_bins,
        )

        return worker_hist.get().astype(np.int64)
