"""Chapter 3, section 3.3.2: coordinating the mean with All-Reduce.

    all_reduce_mean(device_ids, shards)    listing 3.6. Each GPU reduces its own
                                           shard with `partial_launcher`, then two
                                           NCCL All-Reduce calls sum the partial
                                           sums and partial counts across every
                                           GPU, leaving the same global (sum,
                                           count) on each of them.

What it needs. NCCL runs on Linux with NVIDIA GPUs, through CuPy's
`cupy.cuda.nccl` (install NCCL alongside `cupy-cuda12x`). NCCL allows one rank
per GPU, so `device_ids` must name distinct devices; on a one-GPU machine that
means `device_ids=[0]` and one shard. `nccl_available()` reports whether the
collective path can run.

Deviation from the book's listing, and why:

  * Where NCCL or a GPU is missing (Windows, macOS, or no CUDA device), the
    function falls back to reducing each shard with `partial_launcher`'s CPU or
    single-GPU path and summing the pairs on the host, then returns one mean per
    rank as the collective would. The arithmetic is the same; there is no
    collective communication. The notebook says which path ran.
"""

from __future__ import annotations

from partial_launcher import gpu_available, partial_launcher

try:
    import cupy as cp
    from cupy.cuda import nccl
    _NCCL = gpu_available() and bool(nccl.available)
except Exception:                                  # no CuPy, or CuPy built without NCCL
    cp = None
    nccl = None
    _NCCL = False


def nccl_available() -> bool:
    """True when listing 3.6 runs as written, over NCCL."""
    return _NCCL


def all_reduce_mean(device_ids, shards):
    if not _NCCL:
        return _host_all_reduce_mean(device_ids, shards)

    comms = nccl.NcclCommunicator.initAll(device_ids)

    states = []

    for r, device_id in enumerate(device_ids):
        with cp.cuda.Device(device_id):
            partial_sum, partial_count = partial_launcher(
                device_id,
                shards[r],
            )

            global_sum = cp.empty_like(partial_sum)
            global_count = cp.empty_like(partial_count)

            stream = cp.cuda.get_current_stream()

            states.append(
                (partial_sum, partial_count, global_sum, global_count, stream)
            )

    nccl.groupStart()

    for r, device_id in enumerate(device_ids):
        with cp.cuda.Device(device_id):
            partial_sum, partial_count, \
                global_sum, global_count, stream = states[r]

            comms[r].allReduce(
                partial_sum.data.ptr,
                global_sum.data.ptr,
                1,
                nccl.NCCL_FLOAT32,
                nccl.NCCL_SUM,
                stream.ptr,
            )

            comms[r].allReduce(
                partial_count.data.ptr,
                global_count.data.ptr,
                1,
                nccl.NCCL_UINT32,
                nccl.NCCL_SUM,
                stream.ptr,
            )

    nccl.groupEnd()

    means = []

    for r, device_id in enumerate(device_ids):
        with cp.cuda.Device(device_id):
            _, _, global_sum, global_count, stream = states[r]

            stream.synchronize()

            mean = global_sum / global_count

            means.append(float(mean[0].get()))

    return means


def _host_all_reduce_mean(device_ids, shards):
    """The same two sums without a collective: reduce, add on the host, broadcast."""
    total_sum = 0.0
    total_count = 0
    for r, device_id in enumerate(device_ids):
        partial_sum, partial_count = partial_launcher(device_id, shards[r])
        total_sum += float(partial_sum.get()[0])
        total_count += int(partial_count.get()[0])
    return [total_sum / total_count for _ in device_ids]
