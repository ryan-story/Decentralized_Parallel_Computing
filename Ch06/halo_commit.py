"""Chapter 6, section 6.4.1: VHC's message and a tile's state.

    HaloCommit       listing 6.12. One delivery of halo cells: the request, the layer,
                     the logical tile that produced it, the direction from receiver
                     to producer, the attempt, the values, and which of them are
                     exact.
    TileState        listing 6.12. One tile's padded input and output at every layer,
                     with masks of the cells known and computed.
    band(p, T, d, sending)
                     listing 6.13. The rows (or columns) of a halo of depth p around
                     a tile of side T in direction d: the cells a tile sends, or the
                     cells it receives.
    open_tile(request_id, tile, grid, image_tile, attempt_id, pads, exchange)
                     listing 6.13. A tile's state at the start of a request: its own
                     input known, padding known, every halo unknown.

`pads` is the halo depth at each convolution's input; `exchange` lists the
convolutions whose input needs a halo. (1,) * 6 and range(6) exchange at every layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from cnn import CHANNELS


@dataclass(frozen=True)
class HaloCommit:
    request_id: str
    layer: int                     # the input of convolution `layer`
    tile: int                      # the logical tile that produced the strip
    direction: tuple               # from the receiver to that tile
    attempt_id: str
    strip: object                  # (channels, rows, columns) on the device
    cells: object                  # (rows, columns): which values are exact


@dataclass
class TileState:
    request_id: str
    tile: int
    grid: int
    attempt_id: str
    pads: tuple                    # halo depth at each convolution's input
    exchange: tuple                # convolutions whose input needs a halo
    inp: dict = field(default_factory=dict)       # padded input, on the GPU
    have: dict = field(default_factory=dict)      # input cells known exactly
    inside: dict = field(default_factory=dict)    # input cells inside the image
    done: dict = field(default_factory=dict)      # output cells computed
    out: dict = field(default_factory=dict)       # convolution output, on the GPU
    expected: dict = field(default_factory=dict)  # (layer, direction) -> tile
    accepted: dict = field(default_factory=dict)  # (layer, direction) -> attempts
    sent: dict = field(default_factory=dict)      # (layer, direction) -> cells sent


NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy, dx) != (0, 0)]


def band(p, T, d, sending):
    if sending:
        return {-1: slice(p, 2 * p), 0: slice(p, p + T), 1: slice(T, T + p)}[d]
    return {-1: slice(0, p), 0: slice(p, p + T), 1: slice(p + T, 2 * p + T)}[d]


def open_tile(request_id, tile, grid, image_tile, attempt_id, pads, exchange):
    state = TileState(request_id, tile, grid, attempt_id, pads, exchange)
    row, col = divmod(tile, grid)
    for k, p in enumerate(pads):
        T = image_tile.shape[1] // (1, 1, 2, 2, 4, 4)[k]
        g = np.arange(T + 2 * p) - p
        ok_y = (0 <= row * T + g) & (row * T + g < grid * T)
        ok_x = (0 <= col * T + g) & (col * T + g < grid * T)
        state.inside[k] = ok_y[:, None] & ok_x[None, :]
        state.have[k] = ~state.inside[k]
        state.inp[k] = cp.zeros((CHANNELS[k], T + 2 * p, T + 2 * p), cp.float32)
        state.done[k] = np.zeros((T + 2 * p - 2,) * 2, dtype=bool)
        state.out[k] = cp.zeros((CHANNELS[k + 1],) + state.done[k].shape, cp.float32)
        for dy, dx in NEIGHBOURS:
            if k in exchange and 0 <= row + dy < grid and 0 <= col + dx < grid:
                state.expected[(k, (dy, dx))] = (row + dy) * grid + col + dx
    p, T = pads[0], image_tile.shape[1]
    state.inp[0][:, p:p + T, p:p + T] = cp.asarray(image_tile, dtype=cp.float32)
    state.have[0][p:p + T, p:p + T] = True
    return state
