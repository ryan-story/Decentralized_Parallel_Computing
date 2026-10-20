"""Chapter 5, section 5.1.2: a worker's local scan and block summary.

    scan_launcher(device_id, block, _SRC, tile=1024)    listing 5.3. Scan every
                                                        tile with listing 5.1,
                                                        scan the tile totals by
                                                        calling itself, and add
                                                        each tile's carry with
                                                        listing 5.2. Returns the
                                                        local exclusive scan, which
                                                        stays on the device, and
                                                        the block summary.

`_SRC` is the text of `kernels.cu`, compiled at run time. Every later listing
takes it as an argument, as in the book.

Deviations from the book's listing, and why:

  * The CPU fallback. Without CuPy or a CUDA device, `scan_launcher` runs the same
    algorithm on the host: each tile is scanned by replaying listing 5.1's
    up-sweep and down-sweep in NumPy float32, the tile totals are scanned the same
    way, and the carries are added in float32. The steps and their order are the
    kernel's, so the result is the same float32 values.
  * `gpu_available()` is a helper for the notebook and the other modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("kernels.cu").read_text()

try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
    cp.RawModule(code=_SRC, options=("--std=c++14",)).get_function("TileExclusiveScan")
except Exception:                                  # no CuPy, no driver, no device
    cp = None
    _GPU = False


def gpu_available() -> bool:
    """True when the CUDA kernel path is live; False when NumPy is standing in."""
    return _GPU


def scan_launcher(device_id, block, _SRC, tile=1024):
    if not _GPU:
        return _host_scan_launcher(block, tile)

    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _scan = _MODULE.get_function("TileExclusiveScan")
        _carry = _MODULE.get_function("ApplyCarryKernel")

        x = cp.asarray(block, dtype=cp.float32)
        n = x.size
        n_tiles = (n + tile - 1) // tile

        local = cp.empty(n, dtype=cp.float32)
        totals = cp.empty(n_tiles, dtype=cp.float32)

        for k in range(n_tiles):
            lo, hi = k * tile, min((k + 1) * tile, n)
            _scan((1,), (tile,), (x[lo:hi], local[lo:hi], totals[k:k + 1],
                  np.int32(hi - lo)), shared_mem=tile * 4)

        if n_tiles == 1:
            return local, float(totals[0])

        carries, summary = scan_launcher(device_id, totals, _SRC, tile)

        for k, carry in enumerate(cp.asnumpy(carries)):
            lo, hi = k * tile, min((k + 1) * tile, n)
            _carry(((hi - lo + 255) // 256,), (256,),
                   (local[lo:hi], np.float32(carry), np.int32(hi - lo)))

        return local, summary


# ------------------------------------------------------------------ CPU path

def host_tile_scan(x, threads):
    """Listing 5.1 on the host: one tile, `threads` a power of two, len(x) <= threads.

    Each loop below is one level of the kernel's tree; the slice holds the threads
    for which `i < blockDim.x`. Returns the local exclusive scan and the tile total.
    """
    temp = np.zeros(threads, dtype=np.float32)
    temp[:len(x)] = x                                      # pad with the identity
    offset = 1
    while offset < threads:                                # up-sweep
        i = np.arange(2 * offset - 1, threads, 2 * offset)
        temp[i] += temp[i - offset]
        offset <<= 1
    total = np.float32(temp[threads - 1])                  # save the tile total
    temp[threads - 1] = 0.0                                # seed the root
    offset = threads >> 1
    while offset > 0:                                      # down-sweep
        i = np.arange(2 * offset - 1, threads, 2 * offset)
        left = temp[i - offset].copy()
        temp[i - offset] = temp[i]
        temp[i] += left
        offset >>= 1
    return temp[:len(x)].copy(), total


def _host_scan_launcher(block, tile=1024):
    x = np.asarray(block, dtype=np.float32)
    n = x.size
    n_tiles = (n + tile - 1) // tile
    local = np.empty(n, dtype=np.float32)
    totals = np.empty(n_tiles, dtype=np.float32)
    for k in range(n_tiles):
        lo, hi = k * tile, min((k + 1) * tile, n)
        local[lo:hi], totals[k] = host_tile_scan(x[lo:hi], tile)
    if n_tiles == 1:
        return local, float(totals[0])
    carries, summary = _host_scan_launcher(totals, tile)
    for k, carry in enumerate(carries):
        lo, hi = k * tile, min((k + 1) * tile, n)
        local[lo:hi] += np.float32(carry)
    return local, summary


if __name__ == "__main__":
    # The twelve-value example of section 5.1.2, in tiles of four.
    x = [2, 6, 1, 3, 5, 2, 4, 7, 1, 3, 2, 2]
    local, summary = scan_launcher(0, x, _SRC, tile=4)
    if _GPU:
        local = cp.asnumpy(local)
    print("CUDA kernel path live:", _GPU)
    print("block           :", x)
    print("exclusive scan  :", local.astype(int).tolist())
    print("block summary   :", summary)
