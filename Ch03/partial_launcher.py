"""Chapter 3, section 3.3.1: the hierarchical worker reduction launcher.

    partial_launcher(device_id, v)    listing 3.4. One worker's whole local job:
                                      cut its shard into power-of-two blocks of
                                      at most 1,024 records, reduce each block
                                      with `PartialMeanKernel`, and merge the
                                      block partials into one (sum, count).

Deviations from the book's listing, and why:

  * The CPU fallback below. Without CuPy or a CUDA device, the same reduction
    runs on the central processing unit: each block goes through the identical
    halving tree in float32 and matches the kernel bit for bit. The last step,
    adding the block partials, can differ in the final bits, because `cp.sum`
    picks its own summation order. The returned arrays answer `.get()` like CuPy
    arrays, which is what lets listings 3.5, 3.6 and 3.14 run unchanged on
    either path.
  * `gpu_available` and `mean_kernel` are helpers the book states in prose.
    `mean_kernel` launches listing 3.2 on one block so the notebook can show it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("meanreductionkernel.cu").read_text(encoding="utf-8")
MAX_BLOCK = 1024

try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
    cp.RawModule(code=_SRC).get_function("PartialMeanKernel")   # compiles, or fails here
except Exception:                                  # no CuPy, no driver, no device
    cp = None
    _GPU = False


def gpu_available() -> bool:
    """True when the CUDA kernel path is live; False when NumPy is standing in."""
    return _GPU


def partial_launcher(device_id, v):
    """Listing 3.4: one worker's contribution, as two one-element arrays."""
    if not _GPU:
        return _host_partial(v)

    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=_SRC)
        partial = module.get_function("PartialMeanKernel")

        sums, counts = [], []
        offset = 0

        while offset < len(v):
            remaining = len(v) - offset
            n = min(MAX_BLOCK, 1 << (remaining.bit_length() - 1))
            chunk = cp.asarray(v[offset:offset + n], dtype=cp.float32)
            count = cp.empty(n, dtype=cp.uint32)
            block_sum = cp.empty(1, dtype=cp.float32)
            block_count = cp.empty(1, dtype=cp.uint32)

            partial((1,), (n,),
                    (chunk, count, block_sum, block_count))

            sums.append(block_sum)
            counts.append(block_count)
            offset += n

        partial_sum = cp.sum(cp.concatenate(sums), dtype=cp.float32, keepdims=True)
        partial_count = cp.sum(cp.concatenate(counts), dtype=cp.uint32, keepdims=True)
        return partial_sum, partial_count


def mean_kernel(v) -> float:
    """Launch listing 3.2 on one block and return the mean it writes.

    `v` must have a power-of-two length of at most 1,024, because the kernel's
    tree has one thread per element and halves the active threads each stage.
    """
    n = len(v)
    if n < 1 or n > MAX_BLOCK or n & (n - 1):
        raise ValueError("MeanReductionKernel needs a power-of-two block of at most 1,024")
    if not _GPU:
        s, c = _tree(np.asarray(v, dtype=np.float32))
        return float(np.float32(s) / np.float32(c))

    module = cp.RawModule(code=_SRC)
    kernel = module.get_function("MeanReductionKernel")
    buf = cp.asarray(v, dtype=cp.float32)
    count = cp.empty(n, dtype=cp.uint32)
    reduction = cp.empty(1, dtype=cp.float32)
    kernel((1,), (n,), (buf, count, reduction))
    return float(reduction.get()[0])


# ------------------------------------------------------------------ CPU path

class _HostArray(np.ndarray):
    """A NumPy array that answers `.get()` the way a CuPy array does."""

    def get(self):
        return np.asarray(self)


def _tree(block):
    """The kernel's reduction tree, stage by stage, in float32."""
    v = np.array(block, dtype=np.float32)
    count = np.ones(v.size, dtype=np.uint32)
    stride = v.size // 2
    while stride >= 1:
        v[:stride] += v[stride:2 * stride]
        count[:stride] += count[stride:2 * stride]
        stride //= 2
    return v[0], count[0]


def _host_partial(v):
    sums, counts = [], []
    offset = 0
    while offset < len(v):
        remaining = len(v) - offset
        n = min(MAX_BLOCK, 1 << (remaining.bit_length() - 1))
        s, c = _tree(np.asarray(v[offset:offset + n], dtype=np.float32))
        sums.append(s)
        counts.append(c)
        offset += n
    partial_sum = np.sum(np.array(sums, dtype=np.float32), dtype=np.float32, keepdims=True)
    partial_count = np.sum(np.array(counts, dtype=np.uint32), dtype=np.uint32, keepdims=True)
    return partial_sum.view(_HostArray), partial_count.view(_HostArray)
