"""Chapter 3, section 3.3.1: coordinating a mean reduction across workers.

    distributed_mean(v, device_ids)    listing 3.5. Cut `v` into one contiguous
                                       shard per worker, run `partial_launcher`
                                       for every shard at once, wait for every
                                       contribution, and merge the (sum, count)
                                       pairs before the one division.

This is the all-participant coordinator. It calls `result()` on every future, so
the mean cannot exist until every worker has answered.

Deviations from the book's listing, and why:

  * The book writes this coordinator as a program with `device_ids = [0, 1, 2, 3]`
    at module level. Here the body is a function so the notebook can import it;
    running this file still executes the program as the book shows it.
  * The book's program assumes four GPUs. On a machine with fewer, a device ID
    may repeat: `[0, 0, 0, 0]` gives four workers that take turns on one GPU. The
    arithmetic is identical, only the concurrency differs. Without a GPU the
    launcher's CPU path runs instead, and the result is again the same.
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from partial_launcher import gpu_available, partial_launcher


def distributed_mean(v, device_ids):
    n = v.shape[0]
    b_r = len(device_ids)

    boundaries = np.linspace(
        0,
        n,
        b_r + 1,
        dtype=int,
    )

    with ThreadPoolExecutor(max_workers=b_r) as executor:
        futures = []

        for r in range(b_r):
            s_r = boundaries[r]
            e_r = boundaries[r + 1]
            v_r = v[s_r:e_r]

            futures.append(
                executor.submit(
                    partial_launcher,
                    device_ids[r],
                    v_r,
                )
            )

    total_sum = 0.0
    total_count = 0

    for future in futures:
        partial_sum, partial_count = future.result()
        total_sum += float(partial_sum.get()[0])
        total_count += int(partial_count.get()[0])

    return total_sum / total_count


if __name__ == "__main__":
    device_ids = [0, 1, 2, 3]
    if gpu_available():
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        device_ids = [d % n_gpus for d in device_ids]    # repeat IDs on a smaller machine

    v = np.random.default_rng(0).normal(12.0, 3.0, size=19_965).astype(np.float32)
    mean = distributed_mean(v, device_ids)

    print("CUDA kernel path live:", gpu_available(), " workers on devices:", device_ids)
    print(f"distributed mean : {mean:.6f}")
    print(f"NumPy float64    : {float(np.mean(v, dtype=np.float64)):.6f}")
