"""Chapter 4, section 4.3.3: the same merge as a collective.

    allreduce_histogram(counts)    listing 4.6. Every rank enters with its own
                                   local histogram and leaves with the global one,
                                   summed bin by bin across the process group.

Before the listing runs, the ranks must already have joined one process group and
each must hold its histogram as a tensor over the same bins. Running this file
does that setup and then calls the listing:

    python allreduce_histogram.py            # 4 ranks on this machine

What it needs. PyTorch with `torch.distributed`. The demo uses the gloo backend,
which runs on the CPU on Linux, macOS and Windows, so it needs no GPU and no
second machine. On a cluster the same call runs over NCCL between GPUs; only the
backend named in `init_process_group` changes.

Deviation from the book's listing, and why:

  * None in the function body. The import is guarded so the rest of the chapter
    still imports without PyTorch; `allreduce_available()` reports whether the
    collective can run.
"""

from __future__ import annotations

try:
    import torch.distributed as dist
    _DIST = dist.is_available()
except Exception:                                  # no PyTorch
    dist = None
    _DIST = False


def allreduce_available() -> bool:
    """True when PyTorch's distributed package is importable."""
    return _DIST


def allreduce_histogram(counts):
    dist.all_reduce(
        counts,
        op=dist.ReduceOp.SUM,
    )

    return counts


# ------------------------------------------------------------------ demo

def _rank_main(rank, world_size, init_file, result_file):
    """One rank: count its slice of the reference stream, then All-Reduce."""
    import numpy as np
    import torch

    from keystream import build_stream

    dist.init_process_group("gloo", init_method=f"file:///{init_file}",
                            rank=rank, world_size=world_size)
    stream = build_stream()
    workers = [w.worker_id for w in stream.workers][rank::world_size]
    local = sum(np.bincount(stream.keys[w], minlength=stream.num_bins) for w in workers)
    counts = torch.tensor(local, dtype=torch.int64)

    allreduce_histogram(counts)

    if rank == 0:
        np.save(result_file, counts.numpy())
    dist.destroy_process_group()


if __name__ == "__main__":
    import os
    import tempfile

    import numpy as np

    from keystream import build_stream

    if not allreduce_available():
        raise SystemExit("PyTorch with torch.distributed is required for listing 4.6")

    import torch.multiprocessing as mp

    world_size = 4
    tmp = tempfile.mkdtemp()
    init_file = os.path.join(tmp, "pg_init").replace("\\", "/")
    result_file = os.path.join(tmp, "global.npy")
    mp.spawn(_rank_main, args=(world_size, init_file, result_file),
             nprocs=world_size, join=True)

    global_hist = np.load(result_file)
    exact = build_stream().true_hist()
    print(f"ranks             : {world_size} (gloo, one process each)")
    print(f"events after sum  : {int(global_hist.sum()):,}")
    print(f"equals CPU-exact  : {np.array_equal(global_hist, exact)}")
