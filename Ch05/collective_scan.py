"""Chapter 5, section 5.3.2: the carries as an MPI collective.

    collective_scan(device_id, block, _SRC)    listing 5.6. Each MPI rank scans its
                                               own block, `MPI_Exscan` gives it the
                                               combined summaries of the ranks
                                               before it, and it finalizes its
                                               block. Rank order is block order.

Run it with an MPI launcher, one process per rank:

    mpiexec -n 4 python collective_scan.py

Rank 0 gathers the finalized blocks and checks them against listing 5.5 run on the
same blocks; the two are bitwise identical. Every rank must enter the collective:
if one rank never does, the others cannot complete it.
"""

from __future__ import annotations

from mpi4py import MPI

from finalize_block import finalize_block
from scan_launcher import scan_launcher


def collective_scan(device_id, block, _SRC):
    comm = MPI.COMM_WORLD
    local, summary = scan_launcher(device_id, block, _SRC)

    carry = comm.exscan(summary, op=MPI.SUM)
    if comm.Get_rank() == 0:
        carry = 0.0

    return finalize_block(device_id, local, carry, _SRC)


if __name__ == "__main__":
    import numpy as np

    from distributed_scan import distributed_scan
    from scan_launcher import _SRC, gpu_available

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(0.0, 1.0, 4096).astype(np.float32) for _ in range(size)]

    mine = collective_scan(0, blocks[rank], _SRC)
    mine = mine.get() if hasattr(mine, "get") else mine
    gathered = comm.gather(mine, root=0)

    if rank == 0:
        got = np.concatenate(gathered)
        ref = distributed_scan([0] * size, blocks, _SRC)
        ref = np.concatenate([r.get() if hasattr(r, "get") else r for r in ref])
        print("CUDA kernel path live:", gpu_available())
        print(f"ranks             : {size}, one logical block each")
        print(f"bitwise equal to listing 5.5: {np.array_equal(got, ref)}")
