"""Chapter 4, section 4.3.2: merging every contribution.

    distributed_histogram(device_ids, shards, num_bins)    listing 4.5. Run one
                                                           worker per declared
                                                           shard, check each
                                                           contribution's shape,
                                                           type and sign, and add
                                                           it the moment it
                                                           arrives.

This is the all-worker coordinator. Contributions merge in any order, because
integer addition is associative and commutative, but no global histogram exists
until every declared shard has contributed.

On a machine with fewer GPUs than workers, device IDs may repeat: `[0] * 16` gives
sixteen workers that take turns on one GPU. Without a GPU the CPU path of
`partial_histogram` runs instead, and the result is the same.
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from worker_contribution import worker_contribution


def distributed_histogram(device_ids, shards, num_bins):
    total = np.zeros(num_bins, dtype=np.int64)

    with ThreadPoolExecutor(
        max_workers=len(device_ids)
    ) as executor:

        futures = [
            executor.submit(
                worker_contribution,
                device_id,
                shard,
                num_bins,
            )
            for device_id, shard in zip(device_ids, shards)
        ]

        for future in as_completed(futures):
            h = future.result()

            if (
                h.shape != (num_bins,)
                or not np.issubdtype(h.dtype, np.integer)
                or np.any(h < 0)
            ):
                raise ValueError(
                    "invalid histogram contribution"
                )

            total += h

    return total


if __name__ == "__main__":
    from keystream import build_stream
    from partial_histogram import gpu_available

    stream = build_stream()
    shards = [stream.keys[w.worker_id] for w in stream.workers]
    total = distributed_histogram([0] * len(shards), shards, stream.num_bins)

    print("CUDA kernel path live:", gpu_available())
    print(f"workers           : {len(shards)}, one shard each")
    print(f"events counted    : {int(total.sum()):,}")
    print(f"equals CPU-exact  : {np.array_equal(total, stream.true_hist())}")
