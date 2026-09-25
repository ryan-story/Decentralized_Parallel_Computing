"""Chapter 4, section 4.1.3: launching the worker kernel.

    partial_histogram(keys, num_bins)    listing 4.3. Compile `kernels.cu`, give
                                         every block one shared-memory counter per
                                         bin, launch `PartialHistogramKernel`, and
                                         sum the published block rows into one
                                         worker histogram. Returns both.

Deviations from the book's listing, and why:

  * The CPU fallback below. Without CuPy or a CUDA device, the same work runs on
    the central processing unit: every event is assigned to the block that the
    kernel's grid-stride loop would give it, each block's row is an exact count,
    and the rows are summed. The result is the exact histogram on either path.
  * `gpu_available`, `privatization_fits` and `naive_histogram` are helpers the
    notebook needs. `privatization_fits` says whether a block-private histogram of
    `num_bins` counters fits in one block's shared memory on this device;
    `naive_histogram` launches listing 4.1 once so its lost updates can be
    measured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("kernels.cu").read_text()

BLOCK = 256
GRID = 64

try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
    cp.RawModule(code=_SRC, options=("--std=c++14",)).get_function("PartialHistogramKernel")
except Exception:                                  # no CuPy, no driver, no device
    cp = None
    _GPU = False


def gpu_available() -> bool:
    """True when the CUDA kernel path is live; False when NumPy is standing in."""
    return _GPU


def partial_histogram(keys, num_bins, grid=GRID):
    """Listing 4.3: one worker's published block rows and its merged histogram."""
    if not _GPU:
        return _host_partial_histogram(keys, num_bins, grid)

    keys = cp.ascontiguousarray(cp.asarray(keys), dtype=cp.int32)
    n = int(keys.size)
    smem = num_bins * cp.dtype(cp.int32).itemsize

    block_hists = cp.empty(
        (grid, num_bins),
        dtype=cp.int32,
    )

    module = cp.RawModule(
        code=_SRC,
        options=("--std=c++14",),
    )
    fn = module.get_function("PartialHistogramKernel")
    fn.max_dynamic_shared_size_bytes = smem

    fn(
        (grid,),
        (BLOCK,),
        (keys, block_hists.ravel(), n, num_bins),
        shared_mem=smem,
    )

    worker_hist = block_hists.sum(axis=0, dtype=cp.int32)

    return block_hists, worker_hist


def privatization_fits(num_bins) -> bool:
    """Whether one block can hold `num_bins` int32 counters in shared memory.

    On the GPU this is the device's opt-in per-block limit. Beyond it, listing 4.2
    cannot privatize the key universe, and section 4.5.3 counts on the host.
    """
    if not _GPU:
        return True
    dev = cp.cuda.Device()
    try:
        limit = int(dev.attributes["MaxSharedMemoryPerBlockOptin"])
    except KeyError:
        limit = int(dev.attributes["MaxSharedMemoryPerBlock"])
    return num_bins * 4 <= limit


def naive_histogram(keys, num_bins):
    """Launch listing 4.1 once, one thread per event, and return what survived.

    The result is not reproducible: which increments are lost depends on how the
    hardware scheduled the threads. Needs a GPU; there is no race to show on the
    CPU path, where the host counts sequentially.
    """
    if not _GPU:
        raise RuntimeError("listing 4.1's race needs a CUDA device")
    keys = cp.asarray(keys, dtype=cp.int32)
    n = int(keys.size)
    h = cp.zeros(num_bins, dtype=cp.int32)
    fn = cp.RawModule(code=_SRC, options=("--std=c++14",)).get_function(
        "NaiveHistogramKernel")
    fn(((n + BLOCK - 1) // BLOCK,), (BLOCK,), (keys, h, n, num_bins))
    return cp.asnumpy(h).astype(np.int64)


# ------------------------------------------------------------------ CPU path

class _HostArray(np.ndarray):
    """A NumPy array that answers `.get()` the way a CuPy array does."""

    def get(self):
        return np.asarray(self)


def _host_partial_histogram(keys, num_bins, grid=GRID):
    """The kernel's block rows, computed on the host.

    Event `i` is handled by thread `i mod (BLOCK * grid)` in the grid-stride loop,
    which belongs to block `(i mod (BLOCK * grid)) // BLOCK`. Each block's row is
    the exact count of its events, as the atomic in listing 4.2 guarantees.
    """
    keys = np.asarray(keys, dtype=np.int64)
    ok = (keys >= 0) & (keys < num_bins)
    idx = np.flatnonzero(ok)
    block = (idx % (BLOCK * grid)) // BLOCK
    flat = np.bincount(block * num_bins + keys[idx], minlength=grid * num_bins)
    block_hists = flat.reshape(grid, num_bins).astype(np.int32)
    worker_hist = block_hists.sum(axis=0, dtype=np.int32)
    return block_hists.view(_HostArray), worker_hist.view(_HostArray)
