"""Chapter 6, sections 6.4.2 and 6.4.3: exact progress on one tile.

    receive_halo(state, commit, params, device_id, _SRC)
                     listing 6.14. Classify a delivery as rejected, duplicate, stale
                     or accepted; install the exact cells it carries that the tile
                     does not yet have; advance.
    advance(state, params, device_id, _SRC)
                     listing 6.15. At every layer, compute every output cell whose
                     3 x 3 input neighbourhood is now known, and nothing else.
    carry(state, k)  listing 6.16. Hand layer k's newly computed cells, pooled if a
                     pool follows, to layer k + 1's input.
    ready_strips(state)
                     listing 6.16. The exact cells of each outgoing halo that have
                     not been sent yet, one commit per neighbour and layer.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from cnn import POOL_AFTER, conv_layer, max_pool
from halo_commit import HaloCommit, band


def receive_halo(state, commit, params, device_id, _SRC):
    key = (commit.layer, commit.direction)
    if (commit.request_id != state.request_id
            or state.expected.get(key) != commit.tile):
        return "rejected", []
    p = state.pads[commit.layer]
    T = state.inp[commit.layer].shape[1] - 2 * p
    rows = band(p, T, commit.direction[0], sending=False)
    cols = band(p, T, commit.direction[1], sending=False)
    target = state.inp[commit.layer][:, rows, cols]
    if (commit.strip.shape != target.shape
            or commit.cells.shape != target.shape[1:]
            or not bool(cp.isfinite(commit.strip).all())):
        return "rejected", []
    new = commit.cells & ~state.have[commit.layer][rows, cols]
    seen = state.accepted.setdefault(key, set())
    if not new.any():
        return ("duplicate" if commit.attempt_id in seen else "stale"), []
    seen.add(commit.attempt_id)
    target[...] = cp.where(cp.asarray(new), commit.strip, target)
    state.have[commit.layer][rows, cols] |= new
    return "accepted", advance(state, params, device_id, _SRC)


def advance(state, params, device_id, _SRC):
    computed = []
    for k in range(6):
        h = state.have[k]
        n = h.shape[0] - 2
        exact = np.ones((n, n), dtype=bool)
        for p in range(3):
            for q in range(3):
                exact &= h[p:p + n, q:q + n]
        new = exact & ~state.done[k]
        if new.any():
            ys, xs = np.nonzero(new)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            y = conv_layer(device_id, state.inp[k][:, y0:y1 + 2, x0:x1 + 2],
                           params[f"w{k}"], params[f"b{k}"], _SRC)
            region = state.out[k][:, y0:y1, x0:x1]
            region[...] = cp.where(cp.asarray(new[y0:y1, x0:x1]), y, region)
            state.done[k] |= new
            computed.append((k, int(new.sum())))
        if k < 5:
            carry(state, k)
    return computed


def carry(state, k):
    src, done = state.out[k], state.done[k]
    if k in POOL_AFTER:
        n = done.shape[0]
        src = max_pool(src)
        done = done.reshape(n // 2, 2, n // 2, 2).all(axis=(1, 3))
    o = (state.have[k + 1].shape[0] - done.shape[0]) // 2
    win = (slice(o, o + done.shape[0]),) * 2
    keep = done & state.inside[k + 1][win] & ~state.have[k + 1][win]
    dest = state.inp[k + 1][(slice(None),) + win]
    dest[...] = cp.where(cp.asarray(keep), src, dest)
    state.have[k + 1][win] |= keep


def ready_strips(state):
    commits = []
    for (k, (dy, dx)), target in state.expected.items():
        p = state.pads[k]
        T = state.inp[k].shape[1] - 2 * p
        rows, cols = band(p, T, dy, sending=True), band(p, T, dx, sending=True)
        exact = state.have[k][rows, cols]
        sent = state.sent.setdefault((k, (dy, dx)), np.zeros_like(exact))
        new = exact & ~sent
        if new.any():
            sent |= new
            commits.append((target, HaloCommit(
                state.request_id, k, state.tile, (-dy, -dx), state.attempt_id,
                state.inp[k][:, rows, cols].copy(), new)))
    return commits
