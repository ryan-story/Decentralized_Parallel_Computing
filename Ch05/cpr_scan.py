"""Chapter 5, section 5.4.3: running CPR.

    cpr_worker(run_id, s, attempt_id, device_id, block, _SRC)
        listing 5.10. Scan one block and return its local scan with a commit.
    deliver(state, done, held, staged, out, now, _SRC)
        listing 5.10. Offer one batch of arrivals, and any held deferred commits,
        to `receive`; keep each accepted local scan; finalize every block as soon
        as its carry is known. Returns the commits still deferred.
    cpr_scan(run_id, device_ids, spare_ids, blocks, _SRC,
             lookahead, repair_deadline, checkpoint_interval)
        listing 5.11. The coordinator's loop on the wall clock: one first attempt
        per block, wake on an arrival or every 50 ms, and submit a replacement to
        a spare GPU when `check_for_gap` reports a block. Each spare is used once;
        when none is left the scan returns None.

The coordinator holds `blocks`, standing in for the replica or durable source a
deployment would read a repaired block from.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from cpr_coordinator import check_for_gap, receive
from finalize_block import finalize_block
from prefix_repair_state import PrefixRepairState, SummaryCommit
from scan_launcher import scan_launcher


def cpr_worker(run_id, s, attempt_id, device_id, block, _SRC):
    local, summary = scan_launcher(device_id, block, _SRC)
    return device_id, local, SummaryCommit(run_id, s, attempt_id, summary)

def deliver(state, done, held, staged, out, now, _SRC):
    arrived = [f.result() for f in done if f.exception() is None]
    still_held = []
    for device_id, local, commit in held + arrived:
        outcome, finalized = receive(state, commit, now)
        if outcome == "accepted":
            staged[commit.block_seq] = (device_id, local)
        elif outcome == "deferred":
            still_held.append((device_id, local, commit))
        for b in finalized:
            out[b] = finalize_block(*staged[b], state.carries[b], _SRC)
    return still_held


def cpr_scan(run_id, device_ids, spare_ids, blocks, _SRC,
             lookahead, repair_deadline, checkpoint_interval):
    state = PrefixRepairState(run_id, len(blocks), lookahead,
                              repair_deadline, checkpoint_interval)
    staged, held, out = {}, [], {}
    spares = list(spare_ids)
    t0 = time.monotonic()
    ex = ThreadPoolExecutor(max_workers=len(device_ids) + len(spare_ids))
    pending = {ex.submit(cpr_worker, run_id, s, f"{s}.0", d, b, _SRC)
               for s, (d, b) in enumerate(zip(device_ids, blocks))}
    try:
        while state.frontier + 1 < state.n_blocks:
            done, pending = wait(pending, timeout=0.05,
                                 return_when=FIRST_COMPLETED)
            now = time.monotonic() - t0
            held = deliver(state, done, held, staged, out, now, _SRC)
            s = check_for_gap(state, now)
            if s is not None:
                if not spares:
                    return None
                pending.add(ex.submit(cpr_worker, run_id, s, f"{s}.1",
                                      spares.pop(0), blocks[s], _SRC))
    finally:
        ex.shutdown(wait=False)
    return [out[s] for s in range(state.n_blocks)]


if __name__ == "__main__":
    import numpy as np

    from distributed_scan import distributed_scan
    from scan_launcher import _SRC, gpu_available

    rng = np.random.default_rng(0)
    blocks = [rng.uniform(0.0, 1.0, 4096).astype(np.float32) for _ in range(16)]
    real_worker = cpr_worker

    def worker_with_one_dead(run_id, s, attempt_id, device_id, block, _SRC):
        if s == 5 and attempt_id.endswith(".0"):
            raise RuntimeError("worker 5 stopped")          # its first attempt is lost
        return real_worker(run_id, s, attempt_id, device_id, block, _SRC)

    cpr_worker = worker_with_one_dead
    out = cpr_scan("demo", [0] * 16, [0], blocks, _SRC,
                   lookahead=16, repair_deadline=0.2, checkpoint_interval=4)
    got = np.concatenate([o.get() if hasattr(o, "get") else o for o in out])
    ref = distributed_scan([0] * 16, blocks, _SRC)
    ref = np.concatenate([r.get() if hasattr(r, "get") else r for r in ref])

    print("CUDA kernel path live:", gpu_available())
    print("block 5's first worker failed; a spare repaired it")
    print(f"bitwise equal to listing 5.5 with every block present: {np.array_equal(got, ref)}")
