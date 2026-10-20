"""Chapter 5, section 5.3.1: the distributed scan.

    distributed_scan(device_ids, blocks, _SRC)    listing 5.5. Run every worker's
                                                  local scan concurrently, wait for
                                                  all of them, compute the carries
                                                  as the exclusive scan of the
                                                  summaries in logical order, and
                                                  finalize every block.

This is the all-worker coordinator. Only the summaries reach the coordinator; each
local scan stays on its worker's device until its carry is applied. No carry is
computed until every worker has returned, so one worker that never returns stops
the scan.

On a machine with fewer GPUs than workers, device IDs may repeat: `[0] * 64` gives
64 workers that take turns on one GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from finalize_block import finalize_block
from scan_launcher import scan_launcher


def distributed_scan(device_ids, blocks, _SRC):
    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [executor.submit(scan_launcher, d, b, _SRC)
                   for d, b in zip(device_ids, blocks)]

    results = [f.result() for f in futures]

    carries, carry = [], 0.0
    for _, summary in results:
        carries.append(carry)
        carry += summary

    return [finalize_block(d, local, c, _SRC)
            for d, (local, _), c in zip(device_ids, results, carries)]


if __name__ == "__main__":
    import numpy as np

    from scan_launcher import _SRC, gpu_available

    rng = np.random.default_rng(0)
    blocks = [rng.uniform(0.0, 1.0, 4096).astype(np.float32) for _ in range(64)]
    out = distributed_scan([0] * len(blocks), blocks, _SRC)
    got = np.concatenate([o.get() if hasattr(o, "get") else o for o in out])
    x = np.concatenate(blocks).astype(np.float64)
    ref = np.concatenate(([0.0], np.cumsum(x)[:-1]))

    print("CUDA kernel path live:", gpu_available())
    print(f"workers           : {len(blocks)}, 4,096 values each")
    print(f"max normalized error against a float64 scan: "
          f"{np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)):.1e}")
