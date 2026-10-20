"""Chapter 5, section 5.3.1: finalizing a block with its carry.

    finalize_block(device_id, local, carry, _SRC)    listing 5.4. Add the carry to
                                                     a copy of a worker's local
                                                     scan with listing 5.2 and
                                                     return the finalized block,
                                                     still on the GPU.

Because it writes to a copy, applying the same carry twice gives the same output
and leaves the local scan untouched. Section 5.4.4 relies on this when a recovered
coordinator finalizes a block a second time.

Deviation from the book's listing, and why:

  * Without CuPy or a CUDA device, the copy and the float32 addition run on the
    host through NumPy, which gives the same values.
"""

from __future__ import annotations

import numpy as np

from scan_launcher import gpu_available

if gpu_available():
    import cupy as cp


def finalize_block(device_id, local, carry, _SRC):
    if not gpu_available():
        return local.copy() + np.float32(carry)

    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _carry = _MODULE.get_function("ApplyCarryKernel")
        out = local.copy()
        n = out.size
        _carry(((n + 255) // 256,), (256,),
               (out, np.float32(carry), np.int32(n)))
        return out
