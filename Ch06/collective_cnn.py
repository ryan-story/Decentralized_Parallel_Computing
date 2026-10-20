"""Chapter 6, section 6.3.2: the halo exchange with MPI.

    mpi_halo_exchange(cart, tile)                 listing 6.11. Pad one tile with the
                                                  halo from its eight neighbours in
                                                  two phases: columns along the row,
                                                  then rows along the column, so the
                                                  corners arrive through the second
                                                  phase.
    collective_cnn(device_id, image_tile, params, _SRC, comm, grid=4)
                                                  listing 6.11. The network with one
                                                  tile per MPI rank.

Run it with one process per tile:

    mpiexec -n 16 python collective_cnn.py

Rank 0 gathers the 16 final tiles and compares them with listing 6.8 on one GPU.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from cnn import POOL_AFTER, conv_layer, max_pool


def mpi_halo_exchange(cart, tile):
    C, T, _ = tile.shape
    p = np.zeros((C, T + 2, T + 2), dtype=np.float32)
    p[:, 1:T + 1, 1:T + 1] = tile
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    for send, dest, source, recv in ((T, right, left, 0), (1, left, right, T + 1)):
        buf = np.zeros((C, T), dtype=np.float32)
        cart.Sendrecv(np.ascontiguousarray(p[:, 1:T + 1, send]), dest=dest,
                      recvbuf=buf, source=source)
        p[:, 1:T + 1, recv] = buf
    for send, dest, source, recv in ((T, down, up, 0), (1, up, down, T + 1)):
        buf = np.zeros((C, T + 2), dtype=np.float32)
        cart.Sendrecv(np.ascontiguousarray(p[:, send, :]), dest=dest,
                      recvbuf=buf, source=source)
        p[:, recv, :] = buf
    return p


def collective_cnn(device_id, image_tile, params, _SRC, comm, grid=4):
    cart = comm.Create_cart((grid, grid), periods=(False, False))
    t = cp.asarray(image_tile, dtype=cp.float32)
    for k in range(6):
        p = mpi_halo_exchange(cart, cp.asnumpy(t))
        t = conv_layer(device_id, p, params[f"w{k}"], params[f"b{k}"], _SRC)
        if k in POOL_AFTER:
            t = max_pool(t)
    return cart, t


if __name__ == "__main__":
    from mpi4py import MPI

    from cnn import _SRC, cnn_forward, load_weights
    from model import validation_images, normalize, center_crop

    comm = MPI.COMM_WORLD
    images, _ = validation_images()
    params = load_weights()
    image = normalize(center_crop(images[:1]))[0]
    T = image.shape[1] // 4
    r, c = divmod(comm.Get_rank(), 4)
    cart, t = collective_cnn(0, image[:, r * T:(r + 1) * T, c * T:(c + 1) * T],
                             params, _SRC, comm)
    row, col = cart.Get_coords(cart.Get_rank())
    parts = comm.gather((row * 4 + col, cp.asnumpy(t)), root=0)
    if comm.Get_rank() == 0:
        parts = dict(parts)
        got = np.concatenate([np.concatenate([parts[i * 4 + j] for j in range(4)], axis=2)
                              for i in range(4)], axis=1)
        ref, _ = cnn_forward(0, image, params, _SRC)
        print(f"{comm.Get_size()} ranks, bitwise equal to one GPU:",
              bool(np.array_equal(got, cp.asnumpy(ref))))
