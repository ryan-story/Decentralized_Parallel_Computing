"""Chapter 3, section 3.4.7: running a commit-and-merge reduction.

    run_worker(cfg, worker_id, device_id, shard)    listing 3.14. Reduce one shard
                                                    with `partial_launcher` and
                                                    package the (sum, count) as a
                                                    `Contribution` for the epoch.

    run_commit_merge(v, device_ids, cfg)            listing 3.15. Launch every
                                                    worker, feed each contribution
                                                    to the epoch as it arrives,
                                                    close under the quorum,
                                                    collection and deadline rules,
                                                    freeze, merge, divide once.

Here the workers are threads on one machine and time is the real wall clock, so
`cfg.deadline` is in seconds. The chapter's experiments replace the wall clock
with seeded arrival times; that driver is `epoch_experiments.py`, and it hands
contributions to the same `receive()`.

Deviation from the book's listings, and why:

  * None in the function bodies. As with listing 3.5, a device ID may repeat, so
    sixteen workers can share one GPU, and without a GPU the launcher's CPU path
    runs instead.
"""

from __future__ import annotations

import time
import numpy as np

from concurrent.futures import (
    ThreadPoolExecutor,
    FIRST_COMPLETED,
    wait,
)

from commit_merge import CommitMergeEpoch, Contribution, canonical_merge
from partial_launcher import partial_launcher


def run_worker(cfg, worker_id, device_id, shard):

    partial_sum, partial_count = partial_launcher(
        device_id,
        shard,
    )

    partial_sum = float(partial_sum.get()[0])
    partial_count = int(partial_count.get()[0])

    contribution = Contribution(
        epoch_id=cfg.epoch_id,
        worker_id=worker_id,
        revision=0,
        base_state_id=cfg.base_state_id,
        partial_sum=partial_sum,
        partial_count=partial_count,
        op_id=cfg.op_id,
        schema_id=cfg.schema_id,
    )

    contribution.claimed_digest = contribution.digest

    return contribution


def run_commit_merge(v, device_ids, cfg):

    epoch = CommitMergeEpoch(cfg)

    shards = np.array_split(
        v,
        len(device_ids),
    )

    started = time.monotonic()

    executor = ThreadPoolExecutor(
        max_workers=len(device_ids)
    )

    futures = set()

    for worker_id, device_id, shard in zip(
        cfg.eligible_workers,
        device_ids,
        shards,
    ):
        futures.add(
            executor.submit(
                run_worker,
                cfg,
                worker_id,
                device_id,
                shard,
            )
        )

    pending = futures

    while not epoch.closed:

        now = time.monotonic() - started
        epoch.close_if_due(now)

        if epoch.closed:
            break

        close_at = epoch.next_close_time()
        timeout = max(0.0, close_at - now)

        if pending:
            done, pending = wait(
                pending,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
        else:
            time.sleep(timeout)
            done = set()

        if not done:
            continue

        for future in done:

            arrival_time = time.monotonic() - started

            if arrival_time > epoch.next_close_time():
                epoch.close_if_due(arrival_time)

            contribution = future.result()

            epoch.receive(
                contribution,
                arrival_time,
            )

    for future in pending:
        future.cancel()

    executor.shutdown(
        wait=False,
        cancel_futures=True,
    )

    manifest = epoch.freeze()

    if manifest is None:
        return None

    merged_sum, merged_count = canonical_merge(
        [
            (
                entry.worker_id,
                entry.partial_sum,
                entry.partial_count,
            )
            for entry in manifest
        ],
        cfg.merge_profile_id,
    )

    mean = merged_sum / merged_count

    commit = {
        "epoch_id": cfg.epoch_id,
        "close_reason": epoch.close_reason,
        "close_time": epoch.close_time,
        "manifest": manifest,
        "merge_profile_id": cfg.merge_profile_id,
        "merged_sum": merged_sum,
        "merged_count": merged_count,
        "mean": mean,
    }

    return commit


if __name__ == "__main__":
    from commit_merge import EpochConfig
    from partial_launcher import gpu_available

    v = np.random.default_rng(0).normal(12.0, 3.0, size=19_965).astype(np.float32)
    device_ids = [0] * 16                         # sixteen workers, one GPU
    cfg = EpochConfig(
        epoch_id=0,
        eligible_workers=tuple(range(16)),
        base_state_id="s0",
        op_id="mean",
        schema_id="sumcount.v1",
        k=12,
        alpha=1.5,
        deadline=5.0,                             # seconds, on the wall clock
        record_counts={w: len(s) for w, s in enumerate(np.array_split(v, 16))},
        merge_profile_id="sorted-by-worker-f64-intcount",
    )
    commit = run_commit_merge(v, device_ids, cfg)

    # The collection window is alpha * t_k, so it scales with how fast the quorum
    # arrived. Local workers finish within milliseconds of each other, and a worker
    # thread scheduled a moment late can miss a window that short; the manifest
    # then holds fewer than 16 workers. That is the rule of section 3.4.4 at work.
    print("CUDA kernel path live:", gpu_available())
    print(f"closed           : {commit['close_reason'].name} at {commit['close_time']:.3f} s")
    print(f"manifest         : {len(commit['manifest'])} of 16 workers, "
          f"{commit['merged_count']:,} records")
    print(f"committed mean   : {commit['mean']:.6f}")
    print(f"NumPy float64    : {float(np.mean(v, dtype=np.float64)):.6f}")
