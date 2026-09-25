"""Chapter 4, section 4.4.4: the decentralized histogram (ILQH).

    histogram_worker(worker_id, device_id, shard, num_bins)    listing 4.10. One
                                                               worker's histogram,
                                                               packaged as a
                                                               `HistogramContribution`.
    decentralized_histogram(worker_ids, device_ids, shards,    listing 4.10. Launch
                            num_bins, event_counts,            every worker, admit
                            k, alpha, deadline, tau)           contributions as
                                                               they arrive, close at
                                                               the collection window
                                                               or the deadline,
                                                               freeze the accepted
                                                               set, and merge it
                                                               with `bounded_merge`.

The close rule is section 4.2.3's: before the k-th valid contribution the epoch
closes at the deadline; after it, at `alpha` times the elapsed quorum time, and
never later than the deadline. The influence limit is applied inside
`bounded_merge` whenever `tau` is set; `tau=None` is the Quorum Histogram.

Here the workers are threads on one machine and time is the real wall clock, so
`deadline` is in seconds. The chapter's experiments replace the wall clock with
seeded completion times; that driver is `histogram_experiments.py`, and it admits
and merges with the same `validate_histogram` and `bounded_merge`.

Deviation from the book's listing, and why:

  * None in the function bodies. As with listing 4.5, a device ID may repeat, and
    without a GPU the launcher's CPU path runs.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from bounded_merge import bounded_merge
from contribution import HistogramContribution, validate_histogram
from worker_contribution import worker_contribution


def histogram_worker(worker_id, device_id, shard, num_bins):
    h = worker_contribution(device_id, shard, num_bins)
    return HistogramContribution(worker_id, h, int(h.sum()))


def decentralized_histogram(worker_ids, device_ids, shards,
                            num_bins, event_counts,
                            k, alpha, deadline, tau):
    accepted, collection_close = {}, None
    eligible = set(worker_ids)
    t0 = time.monotonic()

    ex = ThreadPoolExecutor(max_workers=len(device_ids))
    pending = {
        ex.submit(histogram_worker, w, d, s, num_bins)
        for w, d, s in zip(worker_ids, device_ids, shards)
    }

    try:
        while pending:
            close_at = deadline
            if collection_close is not None:
                close_at = min(collection_close, deadline)

            remaining = close_at - (time.monotonic() - t0)
            if remaining <= 0: break

            done, pending = wait(
                pending, timeout=remaining,
                return_when=FIRST_COMPLETED)

            for future in done:
                try: c = future.result()
                except Exception: continue
                if c.worker_id not in eligible: continue
                if validate_histogram(c, num_bins,
                                      event_counts[c.worker_id]):
                    accepted[c.worker_id] = c.histogram
                if collection_close is None and len(accepted) >= k:
                    collection_close = alpha * (time.monotonic() - t0)
    finally:
        ex.shutdown(wait=False)

    if len(accepted) < k: return None
    return bounded_merge(accepted, event_counts, tau)


if __name__ == "__main__":
    import numpy as np

    from keystream import build_stream
    from partial_histogram import gpu_available

    stream = build_stream()
    ids = [w.worker_id for w in stream.workers]
    shards = [stream.keys[w] for w in ids]
    counts = {w.worker_id: w.event_count for w in stream.workers}
    exact = stream.true_hist()

    print("CUDA kernel path live:", gpu_available())
    for tau in (None, 3.0):
        merged = decentralized_histogram(ids, [0] * len(ids), shards, stream.num_bins,
                                         counts, k=12, alpha=1.5, deadline=30.0, tau=tau)
        name = "Quorum Histogram (tau=None)" if tau is None else f"ILQH (tau={tau:g})"
        if merged is None:
            print(f"{name:28}: fewer than k contributions before the deadline")
            continue
        tvd = 0.5 * np.abs(merged / merged.sum() - exact / exact.sum()).sum()
        print(f"{name:28}: {int(merged.sum()):,} events merged, "
              f"TVD from CPU-exact {tvd:.3f}")
    # On one machine every worker finishes within milliseconds, so the collection
    # window is short and the accepted set can hold fewer than sixteen workers.
    # That is the close rule working, not an error.
