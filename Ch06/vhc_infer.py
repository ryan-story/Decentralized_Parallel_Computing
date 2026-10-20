"""Chapter 6, section 6.4.4: running VHC on the wall clock.

    vhc_worker(state, inbox, route, report, params, device_id, _SRC, poll)
        listing 6.18. One tile's loop: send newly exact halo cells, report, and wait
        for the next delivery; report the tile's final output once it is exact.
    final_logits(coord, grid, params)
        listing 6.18. Assemble the final tiles and apply the head.
    vhc_infer(request_id, device_ids, spare_ids, tiles, params, _SRC,
              repair_deadline, pads=(1,) * 6, exchange=tuple(range(6)), poll=0.05)
        listing 6.19. Start one thread per tile, route halos between them, and give a
        silent tile's slot to a spare. Each spare is used once; when none is left,
        it returns None.

Every halo routed to a tile is logged, and a spare that takes a tile over receives
that log, standing in for the neighbours resending what they already sent.
"""

from __future__ import annotations

import queue
import threading
import time

import cupy as cp

from cnn import head
from halo_commit import open_tile
from vhc_coordinator import CompletionState, accept_final, check_for_missing, heard
from vhc_tile import advance, ready_strips, receive_halo


def vhc_worker(state, inbox, route, report, params, device_id, _SRC, poll):
    advance(state, params, device_id, _SRC)
    p = state.pads[5]
    own = (slice(p - 1, p - 1 + state.inp[5].shape[1] - 2 * p),) * 2
    while True:
        for target, commit in ready_strips(state):
            route(target, commit)
        if state.done[5][own].all():
            report(state.tile, state.out[5][(slice(None),) + own])
            return
        report(state.tile, None)
        try:
            commit = inbox.get(timeout=poll)
        except queue.Empty:
            continue
        if commit is None:
            return
        receive_halo(state, commit, params, device_id, _SRC)


def final_logits(coord, grid, params):
    rows = [cp.concatenate([coord.final[r * grid + c] for c in range(grid)], axis=2)
            for r in range(grid)]
    features = cp.concatenate(rows, axis=1)
    return features, head(features, params)


def vhc_infer(request_id, device_ids, spare_ids, tiles, params, _SRC,
              repair_deadline, pads=(1,) * 6, exchange=tuple(range(6)), poll=0.05):
    grid, spares = int(len(tiles) ** 0.5), list(spare_ids)
    coord = CompletionState(request_id, len(tiles), repair_deadline)
    inboxes, log, events = {}, {t: [] for t in range(len(tiles))}, queue.Queue()
    route = lambda t, c: (log[t].append(c), inboxes[t].put(c))
    def start(t, device_id, attempt):
        if t in inboxes:
            inboxes[t].put(None)
        inboxes[t] = queue.Queue()
        for c in log[t]:
            inboxes[t].put(c)
        s = open_tile(request_id, t, grid, tiles[t], f"{t}.{attempt}", pads, exchange)
        threading.Thread(target=vhc_worker, daemon=True, args=(
            s, inboxes[t], route, lambda u, f: events.put((u, f)),
            params, device_id, _SRC, poll)).start()
    for t in range(len(tiles)):
        start(t, device_ids[t], 0)
    while True:
        try:
            t, features = events.get(timeout=poll)
            heard(coord, t, time.monotonic())
            if features is not None and accept_final(coord, t, features):
                return final_logits(coord, grid, params)
        except queue.Empty:
            pass
        for t in check_for_missing(coord, time.monotonic()):
            if not spares:
                return None
            start(t, spares.pop(0), 1)


if __name__ == "__main__":
    from cnn import _SRC, cnn_forward, load_weights
    from model import validation_images, normalize, center_crop

    images, _ = validation_images()
    params = load_weights()
    image = normalize(center_crop(images[:1]))[0]
    T = image.shape[1] // 4
    tiles = [image[:, r * T:(r + 1) * T, c * T:(c + 1) * T]
             for r in range(4) for c in range(4)]
    ref, _ = cnn_forward(0, image, params, _SRC)
    vhc_infer("warm-up", [0] * 16, [0], tiles, params, _SRC, repair_deadline=10.0)

    real_worker = vhc_worker

    def worker_with_one_dead(state, *args):
        if state.tile == 5 and state.attempt_id == "5.0":
            return                                      # tile 5's first worker is lost
        return real_worker(state, *args)

    vhc_worker = worker_with_one_dead
    out = vhc_infer("demo", [0] * 16, [0], tiles, params, _SRC, repair_deadline=0.5)
    print("tile 5's worker never reported; a spare took its slot")
    print("bitwise equal to one GPU:", out is not None and bool(cp.array_equal(out[0], ref)))
