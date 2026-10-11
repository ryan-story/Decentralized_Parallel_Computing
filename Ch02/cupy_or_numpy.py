"""CuPy when there is a CUDA device, a NumPy stand-in when there is not.

The chapter's listings are written against CuPy (`import cupy as cp`). The modules
here import `cp` from this file instead, so the function bodies stay exactly as
the book prints them and still run on a machine without a GPU.

    from cupy_or_numpy import cp, gpu_available

On a GPU machine `cp` IS CuPy, and `MatrixMulKernel` is compiled from
`matmulkernel.cu` and launched on the device. Without one, `cp` is a small object
that answers the same calls with NumPy, and `RawModule(...).get_function(
"MatrixMulKernel")` returns a function that computes what the kernel computes,
`C = A B`, on the CPU. The arithmetic is the same; only the hardware differs, and
every entry point prints which path it took.
"""

from __future__ import annotations

import glob
import os
from contextlib import contextmanager

import numpy as np

# On Windows, CuPy's linear algebra (QR, SVD) loads the CUDA toolkit's cuSOLVER and
# cuBLAS libraries, which live in the toolkit's bin directory. Python does not
# search PATH for those, so add the directory explicitly when it exists.
if os.name == "nt":
    _roots = [os.environ.get("CUDA_PATH", "")]
    _roots += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*"),
                     reverse=True)
    for _root in _roots:
        _bin = os.path.join(_root, "bin") if _root else ""
        if _bin and os.path.isdir(_bin):
            os.add_dll_directory(_bin)

try:
    import cupy as _cupy
    _GPU = _cupy.cuda.runtime.getDeviceCount() > 0
except Exception:                                  # no CuPy, no driver, no device
    _cupy = None
    _GPU = False


def gpu_available() -> bool:
    """True when `cp` is CuPy on a CUDA device; False when NumPy is standing in."""
    return _GPU


class _HostKernel:
    """What `MatrixMulKernel` computes, for the CPU path: C[:] = A @ B."""

    def __call__(self, grid, block, args):
        A, B, C, m, n, p = args
        C[...] = (np.asarray(A, dtype=np.float32).reshape(int(m), int(p))
                  @ np.asarray(B, dtype=np.float32).reshape(int(p), int(n)))


class _HostModule:
    def __init__(self, code=None, options=None):
        self.code = code

    def get_function(self, name):
        if name != "MatrixMulKernel":
            raise ValueError(f"the CPU path only provides MatrixMulKernel, not {name!r}")
        return _HostKernel()


class _HostRandomState:
    def __init__(self, seed):
        self._rs = np.random.RandomState(seed)

    def standard_normal(self, size, dtype=np.float32):
        return self._rs.standard_normal(size).astype(dtype)


class _HostRandom:
    RandomState = _HostRandomState


class _HostCuda:
    @staticmethod
    @contextmanager
    def Device(device_id=0):
        yield


class _HostCupy:
    """The handful of CuPy calls the chapter uses, answered by NumPy."""

    float32 = np.float32
    uint32 = np.uint32
    int32 = np.int32
    linalg = np.linalg
    random = _HostRandom()
    cuda = _HostCuda()
    RawModule = _HostModule

    asarray = staticmethod(np.asarray)
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    empty = staticmethod(np.empty)
    zeros = staticmethod(np.zeros)
    sqrt = staticmethod(np.sqrt)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)


cp = _cupy if _GPU else _HostCupy()


def to_host(x):
    """A NumPy array, wherever `x` lives."""
    return _cupy.asnumpy(x) if (_GPU and isinstance(x, _cupy.ndarray)) else np.asarray(x)
