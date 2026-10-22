"""Chapter 6 experiment harness: every experiment result the chapter reports.

Real: every convolution, pool and head runs through the chapter's listings on the
GPU in float32, and every VHC decision is made by the chapter's own `receive_halo`,
`advance`, `ready_strips`, `check_for_missing` and `accept_final`.

Modelled (`simulated_decentralized`): time. A worker's compute time is the measured
GPU kernel cost of the cells it computes (`costs.json`, fitted per layer on an RTX
5070), scaled by a seeded per-worker speed. A strip's delivery time is the link's
one-way latency, times seeded jitter, plus its bytes over the link bandwidth. Live
workers' heartbeats reach the coordinator continuously; a stalled or failed worker's
stop. Nothing crosses a network.
"""
from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cupy as cp
import numpy as np

import model
from image_convolution import conv2d_basic, conv2d_const
from cnn import CHANNELS, POOL_AFTER, cnn_forward, conv_layer, head, load_weights, max_pool
from distributed_cnn import distributed_cnn
from halo_commit import HaloCommit, NEIGHBOURS, band, open_tile
from vhc_tile import advance, ready_strips, receive_halo
from vhc_coordinator import CompletionState, accept_final, check_for_missing, heard

HERE = Path(__file__).resolve().parent
OUT = HERE / "model"
COSTS = OUT / "costs.json"                  # measured on the book's RTX 5070
MEASURED = OUT / "costs_measured.json"      # measure_costs() writes here
_SRC = (HERE / "kernels.cu").read_text()

GRID, TILES, SIDE = 4, 16, 128
FINAL_CELLS = 32 * 32                       # the final feature map, 128 channels deep
LINKS = {                                   # name: (round trip s, bandwidth bytes/s)
    "NVLink class": (0.05e-3, 200e9),
    "100 GbE": (0.1e-3, 12.5e9),
    "10 GbE": (1e-3, 1.25e9),
    "wide area": (50e-3, 125e6),
}
WORKING_LINK = "wide area"
SCHEDULES = {                               # name: (pads, exchange layers)
    "every layer": ((1, 1, 1, 1, 1, 1), (0, 1, 2, 3, 4, 5)),
    "every stage": ((2, 1, 2, 1, 2, 1), (0, 2, 4)),
    "once": ((14, 13, 6, 5, 2, 1), (0,)),
}
JITTER_SIGMA = 0.25                         # lognormal, on the one-way latency
SPEED_SIGMA = 0.25                          # lognormal, per worker
RETRY = 0.05                                # coordinator poll, s
REPAIR_DEADLINE = 0.25                      # s
REPLACE_COST = 0.45                         # s from assignment to a spare's start
SPARES = 2
OBSERVE = 5.0                               # s
REPETITIONS = 20
FINAL_BYTES = 128 * 8 * 8 * 4


# ----------------------------------------------------------------- the model

_PARAMS = None
_VAL = None


def params():
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = load_weights()
    return _PARAMS


def val_images():
    """Validation images as uint8 (N, 160, 160, 3) and labels."""
    global _VAL
    if _VAL is None:
        if not model.CACHE.exists():
            model.dataset()
        z = np.load(model.CACHE)
        _VAL = (z["xva"], z["yva"])
    return _VAL


def image(i, offset_x=None):
    xva, _ = val_images()
    return model.normalize(model.center_crop(xva[i:i + 1], offset_x))[0]


def split(img):
    T = img.shape[1] // GRID
    return [img[:, r * T:(r + 1) * T, c * T:(c + 1) * T]
            for r in range(GRID) for c in range(GRID)]


def neighbours_of(t):
    row, col = divmod(t, GRID)
    return [(row + dy) * GRID + col + dx for dy, dx in NEIGHBOURS
            if 0 <= row + dy < GRID and 0 <= col + dx < GRID]


# ------------------------------------------------------ measured kernel costs

def _time_kernel(fn, launches=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(launches):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e) / 1000.0)
    return np.array(ts)


def measure_costs():
    """Fit time = floor + per_cell * cells for ConvLayerKernel at every layer."""
    mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
    kern = mod.get_function("ConvLayerKernel")
    P, fits = params(), []
    for k in range(6):
        w, b = cp.asarray(P[f"w{k}"]), cp.asarray(P[f"b{k}"])
        C, M = CHANNELS[k], w.shape[0]
        sides, med = (2, 4, 8, 16, 24, 32, 48, 64), []
        for s in sides:
            x = cp.random.rand(C, s + 2, s + 2).astype(cp.float32)
            y = cp.empty((M, s, s), cp.float32)
            g = ((s + 15) // 16, (s + 15) // 16, M)
            args = (x, w, b, y, np.int32(C), np.int32(s), np.int32(s))
            med.append(float(np.median(_time_kernel(lambda: kern(g, (16, 16, 1), args),
                                                    launches=500))))
        cells = np.array(sides) ** 2
        floor = float(np.median(med[:3]))
        big = cells >= 16 * 16
        per_cell = float(np.median(np.array(med)[big] / cells[big]))
        fits.append({"layer": k, "floor_s": floor, "per_cell_s": per_cell,
                     "sides": list(sides), "median_s": med})
    info = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "layers": fits}
    MEASURED.write_text(json.dumps(info, indent=2))
    return info


_COSTS = None


def compute_time(k, cells):
    global _COSTS
    if _COSTS is None:
        _COSTS = json.loads(COSTS.read_text())["layers"]
    c = _COSTS[k]
    return max(c["floor_s"], c["per_cell_s"] * cells)


# ---------------------------------------------------------------- the trace

@dataclass
class Trace:
    """One repetition's seeded draws, shared by both arms."""
    speed: np.ndarray                       # per tile, and per spare after it
    jitter: dict = field(default_factory=dict)
    rng: np.random.Generator = None

    def delay(self, link, src, dst, layer, nbytes):
        key = (src, dst, layer)
        if key not in self.jitter:
            self.jitter[key] = float(self.rng.lognormal(0.0, JITTER_SIGMA))
        rtt, bw = LINKS[link]
        return rtt / 2 * self.jitter[key] + nbytes / bw


def make_trace(seed):
    rng = np.random.default_rng(seed)
    return Trace(speed=rng.lognormal(0.0, SPEED_SIGMA, TILES + SPARES), rng=rng)


@dataclass
class Fault:
    gone: dict = field(default_factory=dict)      # tile -> time it stops for good
    stall: dict = field(default_factory=dict)     # tile -> time it resumes


# ------------------------------------------------------------- VHC, simulated

def whole_strips(state):
    """Whole-strip granularity, for comparison with listing 6.16's ready_strips:
    a strip is held until every one of its cells is exact, then sent once."""
    out = []
    for (k, (dy, dx)), target in state.expected.items():
        p = state.pads[k]
        T = state.inp[k].shape[1] - 2 * p
        rows, cols = band(p, T, dy, sending=True), band(p, T, dx, sending=True)
        exact = state.have[k][rows, cols]
        sent = state.sent.setdefault((k, (dy, dx)), np.zeros_like(exact))
        if exact.all() and not sent.any():
            sent |= exact
            out.append((target, HaloCommit(
                state.request_id, k, state.tile, (-dy, -dx), state.attempt_id,
                state.inp[k][:, rows, cols].copy(), exact.copy())))
    return out


def independent_fraction(missing):
    """The fraction of the final feature map that does not depend on any input
    pixel of the missing tiles: the geometric bound on exact progress."""
    T = SIDE // GRID
    unknown = np.zeros((SIDE, SIDE), bool)
    for t in missing:
        r, c = divmod(t, GRID)
        unknown[r * T:(r + 1) * T, c * T:(c + 1) * T] = True
    for k in range(6):
        unknown = _dilate(unknown)
        if k in POOL_AFTER:
            n = unknown.shape[0] // 2
            unknown = unknown.reshape(n, 2, n, 2).any(axis=(1, 3))
    return 1.0 - unknown.sum() / unknown.size


def run_vhc(img, trace, link=WORKING_LINK, fault=None, schedule="every layer",
            repair_deadline=REPAIR_DEADLINE, spares=SPARES, observe=OBSERVE,
            granularity="cells"):
    """VHC on one image in simulated time. Returns timings, counts, and the output."""
    fault = fault or Fault()
    strips = ready_strips if granularity == "cells" else whole_strips
    P, (pads, exchange) = params(), SCHEDULES[schedule]
    tiles = split(img)
    coord = CompletionState("q", TILES, repair_deadline)
    workers = {}                                   # worker id -> dict
    serving = {t: t for t in range(TILES)}         # tile -> worker id now routed to
    log = {t: [] for t in range(TILES)}            # strips routed to each tile
    events, seq = [], [0]
    area = {}                                      # tile -> [(time, owned cells done)]
    stats = {"messages": 0, "bytes": 0, "cells": 0, "repairs": 0,
             "classes": {}, "final_at": None, "repair_times": []}

    def push(t, kind, payload):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, payload))

    def new_worker(wid, tile, attempt, start):
        st = open_tile("q", tile, GRID, tiles[tile], f"{tile}.{attempt}", pads, exchange)
        workers[wid] = {"state": st, "tile": tile, "busy": start, "final": False}
        push(start, "start", wid)

    def alive(wid, t):
        tile = workers[wid]["tile"]
        if wid < TILES and tile in fault.gone and t >= fault.gone[tile]:
            return False
        return True

    def ready_at(wid, t):
        if wid < TILES and wid in fault.stall:
            return max(t, fault.stall[wid])
        return t

    def owned_done(st):
        p = st.pads[5]
        own = (slice(p - 1, p - 1 + st.inp[5].shape[1] - 2 * p),) * 2
        return int(st.done[5][own].sum()), own

    def after_compute(wid, computed, t0):
        w = workers[wid]
        dt = sum(compute_time(k, n) for k, n in computed) * trace.speed[wid]
        stats["cells"] += sum(n for _, n in computed)
        w["busy"] = t0 + dt
        if not alive(wid, w["busy"]):
            return
        done, own = owned_done(w["state"])
        area.setdefault(w["tile"], []).append((w["busy"], done))
        for target, commit in strips(w["state"]):
            nbytes = int(commit.cells.sum()) * commit.strip.shape[0] * 4
            stats["messages"] += 1
            stats["bytes"] += nbytes
            d = trace.delay(link, w["tile"], target, commit.layer, nbytes)
            log[target].append(commit)
            push(w["busy"] + d, "halo", (serving[target], commit))
        if done == 64 and not w["final"]:                          # 8 x 8 owned
            w["final"] = True
            d = trace.delay(link, w["tile"], -1, 6, FINAL_BYTES)
            push(w["busy"] + d, "final", (w["tile"], w["state"].out[5][(slice(None),) + own]))

    for t in range(TILES):
        new_worker(t, t, 0, ready_at(t, 0.0))
        heard(coord, t, 0.0)
    for k in range(1, int(observe / RETRY) + 1):
        push(k * RETRY, "tick", None)

    next_spare = TILES
    while events:
        t, _, kind, payload = heapq.heappop(events)
        if t > observe:
            break
        if kind == "start":
            wid = payload
            if not alive(wid, t):
                continue
            computed = advance(workers[wid]["state"], P, 0, _SRC)
            after_compute(wid, computed, t)
        elif kind == "halo":
            wid, commit = payload
            if not alive(wid, t):
                continue
            t0 = max(ready_at(wid, t), workers[wid]["busy"])
            if not alive(wid, t0):
                continue
            status, computed = receive_halo(workers[wid]["state"], commit, P, 0, _SRC)
            stats["classes"][status] = stats["classes"].get(status, 0) + 1
            after_compute(wid, computed, t0)
        elif kind == "final":
            tile, features = payload
            if accept_final(coord, tile, features):
                stats["final_at"] = t
                break
        elif kind == "tick":
            for wid, w in workers.items():                 # heartbeats
                if alive(wid, t) and ready_at(wid, t) == t:
                    heard(coord, w["tile"], t)
            for tile in check_for_missing(coord, t):
                if spares == 0:
                    stats["exhausted"] = True
                    continue
                spares -= 1
                stats["repairs"] += 1
                stats["repair_times"].append(t)
                wid = next_spare
                next_spare += 1
                serving[tile] = wid
                start = t + REPLACE_COST
                new_worker(wid, tile, 1, start)
                for commit in log[tile]:                     # replay the routed log
                    d = trace.delay(link, commit.tile, tile, commit.layer,
                                    int(commit.cells.sum()) * commit.strip.shape[0] * 4)
                    push(start + d, "halo", (wid, commit))

    out = None
    if stats["final_at"] is not None:
        rows = [cp.concatenate([coord.final[r * GRID + c] for c in range(GRID)], axis=2)
                for r in range(GRID)]
        feats = cp.concatenate(rows, axis=1)
        out = (feats, head(feats, P))
    done_map = np.zeros((32, 32), bool)            # exact cells of the final map
    for w in workers.values():
        st = w["state"]
        _, own = owned_done(st)
        r, c = divmod(w["tile"], GRID)
        done_map[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] |= st.done[5][own]
    stats["area"] = area
    stats["done_map"] = done_map
    stats["out"] = out
    return stats


def area_curve(area, ts):
    """Exact fraction of the final feature map at each time in ts."""
    total = np.zeros(len(ts))
    for tile, pts in area.items():
        if not pts:
            continue
        pts = sorted(pts)
        times = np.array([p[0] for p in pts])
        vals = np.maximum.accumulate(np.array([p[1] for p in pts]))
        idx = np.searchsorted(times, ts, side="right") - 1
        total += np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0)
    return total / FINAL_CELLS


# ------------------------------------------------- conventional, simulated

def run_conventional(trace, link=WORKING_LINK, fault=None, observe=OBSERVE):
    """Listing 6.9's schedule with the interior overlapped with the exchange.

    Layer k of a worker: start the interior when layer k-1 is done, wait for all
    eight strips of layer k's input, then compute the boundary band. Same delay and
    speed draws as VHC. Returns per-tile final times (inf when never) and the area.
    """
    fault = fault or Fault()
    T = [32, 32, 16, 16, 8, 8]
    inf = float("inf")
    start = {t: (fault.stall.get(t, 0.0) if t not in fault.gone else
                 (inf if fault.gone[t] <= 0 else fault.stall.get(t, 0.0)))
             for t in range(TILES)}
    send = {t: start[t] for t in range(TILES)}       # layer-k input ready to send
    interior_done, area = {}, {t: [] for t in range(TILES)}
    for k in range(6):
        Tk = T[k]
        c_int = compute_time(k, (Tk - 2) ** 2)
        c_bnd = compute_time(k, Tk * Tk - (Tk - 2) ** 2)
        done = {}
        for t in range(TILES):
            s = trace.speed[t]
            if start[t] == inf:
                done[t] = inf
                continue
            arrive = 0.0
            for n in neighbours_of(t):
                nbytes = CHANNELS[k] * _strip_cells(t, n, Tk) * 4
                a = send[n] + trace.delay(link, n, t, k, nbytes)
                if n in fault.gone and send[n] >= fault.gone[n]:
                    a = inf
                arrive = max(arrive, a)
            d = max(start[t] + c_int * s, arrive) + c_bnd * s
            if t in fault.gone and d >= fault.gone[t]:
                d = inf
            done[t] = d
            if k == 5:
                interior_done[t] = start[t] + c_int * s
        for t in range(TILES):
            start[t] = send[t] = done[t]
    final = {t: start[t] + trace.delay(link, t, -1, 6, FINAL_BYTES) for t in range(TILES)}
    for t in range(TILES):
        if interior_done.get(t, inf) < inf:
            area[t].append((interior_done[t], 36))
        if start[t] < inf:
            area[t].append((start[t], 64))
    t100 = max(final.values())
    return {"t100": t100 if t100 <= observe else None, "area": area}


def _strip_cells(t, n, Tk):
    (r1, c1), (r2, c2) = divmod(t, GRID), divmod(n, GRID)
    dy, dx = r2 - r1, c2 - c1
    return (Tk if dy == 0 else 1) * (Tk if dx == 0 else 1)


# ------------------------------------------------- synchronous VHC driver

def run_sync(img, schedule="every layer", substitute=None, corrupt=None):
    """Every tile present, messages delivered in waves. `substitute` maps a source
    tile to {(layer, direction, target): (strip, attempt_id)} that replaces its
    strips (a declared stale or padding halo). `corrupt` = (target tile, layer,
    direction, index, delta) adds delta to one value of one delivered strip.
    Returns the states and every commit each tile sent."""
    P, (pads, exchange) = params(), SCHEDULES[schedule]
    tiles = split(img)
    states = [open_tile("q", t, GRID, tiles[t], f"{t}.0", pads, exchange)
              for t in range(TILES)]
    keys = {t: set() for t in range(TILES)}
    for s in states:
        advance(s, P, 0, _SRC)
    while True:
        msgs = [(t, c) for s in states for t, c in ready_strips(s)]
        if not msgs:
            break
        for target, c in msgs:
            keys[c.tile].add((c.layer, c.direction, target))
            if substitute and c.tile in substitute:
                strip, attempt = substitute[c.tile][(c.layer, c.direction, target)]
                c = HaloCommit(c.request_id, c.layer, c.tile, c.direction,
                                 attempt, strip, c.cells)
            if corrupt and (target, c.layer, c.direction) == corrupt[:3]:
                strip = c.strip.copy()
                strip[corrupt[3]] += corrupt[4]
                c = HaloCommit(c.request_id, c.layer, c.tile, c.direction,
                                 c.attempt_id, strip, c.cells)
            status, _ = receive_halo(states[target], c, P, 0, _SRC)
            assert status == "accepted", status
    sent = {}                                      # every full band each tile sent
    for t, ks in keys.items():
        st = states[t]
        sent[t] = {}
        for layer, (dy, dx), target in ks:
            p = st.pads[layer]
            T = st.inp[layer].shape[1] - 2 * p
            rows = band(p, T, -dy, sending=True)
            cols = band(p, T, -dx, sending=True)
            sent[t][(layer, (dy, dx), target)] = st.inp[layer][:, rows, cols].copy()
    return states, sent


def assemble(states, k):
    """The owned part of convolution k's output, across the grid."""
    rows = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            st = states[r * GRID + c]
            o = st.pads[k] - 1
            T = st.inp[k].shape[1] - 2 * st.pads[k]
            row.append(st.out[k][:, o:o + T, o:o + T])
        rows.append(cp.concatenate(row, axis=2))
    return cp.concatenate(rows, axis=1)


# ------------------------------------------------------- 6.1 kernel ladder

def kernel_ladder(side=4096, radii=(1, 2, 4), launches=200):
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(6000)
    mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
    basic = mod.get_function("Conv2DBasicKernel")
    const = mod.get_function("Conv2DConstKernel")
    tiled = mod.get_function("Conv2DTiledKernel")
    sym = mod.get_global("FILTER")
    N = cp.asarray(rng.uniform(0, 1, (side, side)).astype(np.float32))
    P = cp.empty_like(N)
    rows = []
    for r in radii:
        filt = rng.normal(0, 1, (2 * r + 1, 2 * r + 1)).astype(np.float32)
        F_d = cp.asarray(filt)
        cp.ndarray((filt.size,), cp.float32, sym)[...] = F_d.ravel()
        a = (np.int32(side), np.int32(side), np.int32(r))
        ref = F.conv2d(torch.as_tensor(cp.asnumpy(N), dtype=torch.float64,
                                       device="cuda")[None, None],
                       torch.as_tensor(filt, dtype=torch.float64, device="cuda")[None, None],
                       padding=r)[0, 0].cpu().numpy()
        row = {"radius": r, "filter_side": 2 * r + 1}

        def stat(name, fn, out):
            ts = _time_kernel(fn, launches)
            row[name] = {"median_ms": float(np.median(ts) * 1e3),
                         "p99_ms": float(np.percentile(ts, 99) * 1e3),
                         "max_abs_error": float(np.abs(cp.asnumpy(out()) - ref).max())}
        g16 = ((side + 15) // 16, (side + 15) // 16)
        g32 = ((side + 31) // 32, (side + 31) // 32)
        sh16, sh32 = (16 + 2 * r) ** 2 * 4, (32 + 2 * r) ** 2 * 4
        stat("basic", lambda: basic(g16, (16, 16), (N, F_d, P) + a), lambda: P)
        stat("constant", lambda: const(g16, (16, 16), (N, P) + a), lambda: P)
        stat("tiled_16", lambda: tiled(g16, (16, 16), (N, P) + a, shared_mem=sh16),
             lambda: P)
        stat("tiled_32", lambda: tiled(g32, (32, 32), (N, P) + a, shared_mem=sh32),
             lambda: P)
        x_t = torch.as_tensor(N, device="cuda")[None, None]
        f_t = torch.as_tensor(filt, device="cuda")[None, None]
        res = {}

        def cudnn():
            res["y"] = F.conv2d(x_t, f_t, padding=r)
        stat("cudnn", cudnn, lambda: cp.asarray(res["y"][0, 0]))
        rows.append(row)
    # the printed launchers give the same output as the timed launches
    img = cp.asnumpy(N[:512, :512])
    filt = rng.normal(0, 1, (5, 5)).astype(np.float32)
    outs = [conv2d_basic(0, img, filt, _SRC), conv2d_const(0, img, filt, _SRC),
            conv2d_const(0, img, filt, _SRC, tiled=True)]
    agree = all(float(cp.abs(o - outs[0]).max()) < 1e-4 for o in outs)
    return {"side": side, "launches": launches, "rows": rows, "listings_agree": agree,
            "device": torch.cuda.get_device_name(0)}


def layer_kernels(launches=200):
    """ConvLayerKernel (listing 6.6) against cuDNN on every layer, whole image."""
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
    kern = mod.get_function("ConvLayerKernel")
    P = params()
    x = cp.asarray(image(0))
    rows = []
    for k in range(6):
        C, H = x.shape[0], x.shape[1]
        w, b = cp.asarray(P[f"w{k}"]), cp.asarray(P[f"b{k}"])
        xp = cp.pad(x, ((0, 0), (1, 1), (1, 1)))
        y = cp.empty((w.shape[0], H, H), cp.float32)
        g = ((H + 15) // 16, (H + 15) // 16, w.shape[0])
        args = (xp, w, b, y, np.int32(C), np.int32(H), np.int32(H))
        ours = _time_kernel(lambda: kern(g, (16, 16, 1), args), launches)
        xt, wt, bt = (torch.as_tensor(v, device="cuda") for v in (x[None], w, b))
        res = {}

        def cudnn():
            res["y"] = F.conv2d(xt, wt, bt, padding=1)
        theirs = _time_kernel(cudnn, launches)
        err = float(cp.abs(y - cp.maximum(cp.asarray(res["y"][0]), 0)).max())
        rows.append({"layer": k, "c_in": C, "c_out": int(w.shape[0]), "side": H,
                     "macs": int(H * H * C * w.shape[0] * 9),
                     "ours_median_ms": float(np.median(ours) * 1e3),
                     "cudnn_median_ms": float(np.median(theirs) * 1e3),
                     "max_abs_difference": err})
        x = y if k not in POOL_AFTER else max_pool(y)
    return rows


# ------------------------------------------------- 6.1 a filter on a photograph

CAT = HERE / "assets" / "chelsea.png"       # CC0, Stefan van der Walt, via scikit-image
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)


def feature_maps():
    """The photograph through listing 6.5 with two edge filters, and through the
    trained network one layer at a time (listing 6.7, as listing 6.8 applies it)."""
    from PIL import Image
    rgb = np.asarray(Image.open(CAT).convert("RGB"))[:, 1:449]   # 300 x 448, even sides
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    edges = {"vertical": cp.asnumpy(conv2d_const(0, gray, SOBEL_X, _SRC)),
             "horizontal": cp.asnumpy(conv2d_const(0, gray, SOBEL_X.T.copy(), _SRC))}
    P, x, layers = params(), cp.asarray(model.normalize(rgb[None])[0]), []
    for k in range(6):
        x = conv_layer(0, cp.pad(x, ((0, 0), (1, 1), (1, 1))), P[f"w{k}"], P[f"b{k}"], _SRC)
        layers.append(cp.asnumpy(x))
        if k in POOL_AFTER:
            x = max_pool(x)
    features, logits = cnn_forward(0, model.normalize(rgb[None])[0], P, _SRC)
    assert cp.array_equal(features, cp.asarray(layers[5]))      # the same as listing 6.8
    return {"rgb": rgb, "edges": edges, "layers": layers, "logits": cp.asnumpy(logits)}


# ------------------------------------------------------------- exactness

def exactness(limit=None):
    """Every validation image: one GPU, the distributed baseline, and VHC (each
    halo schedule), bitwise; and the class against PyTorch/cuDNN and the label."""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    xva, yva = val_images()
    n = len(xva) if limit is None else limit
    net = model.frozen_torch_net(params()).cuda()
    P = params()
    counts = {"images": n, "distributed_bitwise": 0,
              "vhc_bitwise": {s: 0 for s in SCHEDULES},
              "top1_equal_cudnn": 0, "correct_ours": 0, "correct_cudnn": 0,
              "max_logit_difference_cudnn": 0.0}
    for i in range(n):
        img = image(i)
        f1, l1 = cnn_forward(0, img, P, _SRC)
        f2, l2 = distributed_cnn([0] * TILES, img, P, _SRC)
        counts["distributed_bitwise"] += int(bool(cp.array_equal(f1, f2)
                                                  and cp.array_equal(l1, l2)))
        for s in SCHEDULES:
            states, _ = run_sync(img, s)
            counts["vhc_bitwise"][s] += int(bool(cp.array_equal(assemble(states, 5), f1)))
        with torch.no_grad():
            lt = net(torch.as_tensor(img, device="cuda")[None])[0]
        l1n, ltn = cp.asnumpy(l1), lt.cpu().numpy()
        counts["top1_equal_cudnn"] += int(l1n.argmax() == ltn.argmax())
        counts["correct_ours"] += int(l1n.argmax() == yva[i])
        counts["correct_cudnn"] += int(ltn.argmax() == yva[i])
        counts["max_logit_difference_cudnn"] = max(counts["max_logit_difference_cudnn"],
                                                   float(np.abs(l1n - ltn).max()))
    return counts


# ----------------------------------------------------- 6.2 working example

CONDITIONS = [
    ("All tiles present", Fault()),
    ("One tile 1 s late", Fault(stall={5: 1.0})),
    ("One interior tile missing", Fault(gone={5: 0.0})),
    ("One corner tile missing", Fault(gone={0: 0.0})),
    ("One tile fails mid-request", Fault(gone={5: 0.08})),
    ("Two adjacent tiles missing", Fault(gone={5: 0.0, 6: 0.0})),
]


def chosen_images(reps, seed=6000):
    return np.random.default_rng(seed).choice(len(val_images()[0]), reps, replace=False)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def working_example(reps=REPETITIONS, dt=0.005, horizon=1.5):
    P = params()
    ts = np.arange(0, horizon + dt / 2, dt)
    rows, curves = [], {}
    images = chosen_images(reps)
    for name, fault in CONDITIONS:
        c_t, v_t, v_rep, plateau, bitwise, c_curve, v_curve = [], [], [], [], [], [], []
        for r in range(reps):
            img = image(int(images[r]))
            c = run_conventional(make_trace(6100 + r), fault=fault)
            v = run_vhc(img, make_trace(6100 + r), fault=fault)
            f1, _ = cnn_forward(0, img, P, _SRC)
            bitwise.append(v["out"] is not None and bool(cp.array_equal(v["out"][0], f1)))
            c_t.append(c["t100"])
            v_t.append(v["final_at"])
            v_rep.append(v["repairs"])
            if v["repair_times"]:
                at = np.array([v["repair_times"][0]])
                plateau.append(float(area_curve(v["area"], at)[0]))
            c_curve.append(area_curve(c["area"], ts))
            v_curve.append(area_curve(v["area"], ts))
        rows.append({"condition": name,
                     "independent_bound": float(independent_fraction(list(fault.gone) + list(fault.stall))),
                     "conventional_completed": all(t is not None for t in c_t),
                     "conventional_t100": _mean(c_t),
                     "vhc_t100": _mean(v_t), "vhc_repairs": float(np.mean(v_rep)),
                     "vhc_area_before_repair": float(np.mean(plateau)) if plateau else None,
                     "bitwise": all(bitwise)})
        curves[name] = {"conventional": np.mean(c_curve, axis=0).tolist(),
                        "vhc": np.mean(v_curve, axis=0).tolist()}
    return {"rows": rows, "t": ts.tolist(), "curves": curves}


# --------------------------------------------------------- 6.5 evaluation

def halo_depth(reps=REPETITIONS):
    rows = []
    images = chosen_images(reps)
    for link in LINKS:
        for s in SCHEDULES:
            t100, msgs, nbytes, cells = [], [], [], []
            for r in range(reps):
                v = run_vhc(image(int(images[r])), make_trace(6100 + r), link=link,
                            schedule=s)
                t100.append(v["final_at"])
                msgs.append(v["messages"])
                nbytes.append(v["bytes"])
                cells.append(v["cells"])
            rows.append({"link": link, "schedule": s, "t100": _mean(t100),
                         "messages": float(np.mean(msgs)),
                         "halo_bytes": float(np.mean(nbytes)),
                         "computed_cells": float(np.mean(cells))})
    owned = TILES * sum(t * t for t in (32, 32, 16, 16, 8, 8))
    for row in rows:
        row["redundant_compute"] = row["computed_cells"] / owned
    return {"rows": rows, "owned_cells": owned}


def repair_deadline(deadlines=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5), reps=REPETITIONS):
    rows = []
    images = chosen_images(reps)
    for d in deadlines:
        t100, rep = [], []
        for r in range(reps):
            v = run_vhc(image(int(images[r])), make_trace(6100 + r),
                        fault=Fault(stall={5: 1.0}), repair_deadline=d)
            t100.append(v["final_at"])
            rep.append(v["repairs"])
        rows.append({"repair_deadline": d, "t100": _mean(t100),
                     "repairs": float(np.mean(rep))})
    return {"rows": rows}


def staleness(n_images=200, frames=(8, 16, 24, 32), ages=(1, 2, 4, 8), source=5):
    """The panning stream: frame f is the 128 x 128 crop at horizontal offset f, so
    the camera moves one pixel per frame. Tile `source`'s strips, at every layer and
    to every neighbour, are replaced by its strips from frame f - s (a stale halo)
    or by zeros (padding in place of data). Everything else is exact."""
    P = params()
    idx = np.random.default_rng(6300).choice(len(val_images()[0]), n_images, replace=False)
    modes = [f"stale {s}" for s in ages] + ["padding"]
    acc = {m: {"agree": 0, "rel_err": [], "n": 0} for m in modes}
    for i in idx:
        cache = {}

        def exact(f):
            if f not in cache:
                st, sent = run_sync(image(int(i), offset_x=f))
                cache[f] = (head(assemble(st, 5), P), sent[source])
            return cache[f]
        for f in frames:
            l_exact, sent_now = exact(f)
            for m in modes:
                if m == "padding":
                    sub = {k: (cp.zeros_like(v), "padding") for k, v in sent_now.items()}
                else:
                    s = int(m.split()[1])
                    sub = {k: (v, f"stale:{s}") for k, v in exact(f - s)[1].items()}
                st, _ = run_sync(image(int(i), offset_x=f), substitute={source: sub})
                lg = head(assemble(st, 5), P)
                acc[m]["agree"] += int(int(cp.argmax(lg)) == int(cp.argmax(l_exact)))
                acc[m]["rel_err"].append(float(cp.linalg.norm(lg - l_exact)
                                               / cp.linalg.norm(l_exact)))
                acc[m]["n"] += 1
    rows = [{"mode": m, "top1_agreement": a["agree"] / a["n"],
             "mean_relative_logit_error": float(np.mean(a["rel_err"])),
             "evaluations": a["n"]} for m, a in acc.items()]
    return {"rows": rows, "images": n_images, "frames": list(frames), "source_tile": source}


def _dilate(m):
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def containment(n_images=200, delta=1.0):
    """One value of one layer-0 strip from tile 5 to tile 6 is changed by delta.
    The cells that differ at each layer are compared with the cone predicted by
    dilating the changed cell by one 3 x 3 neighbourhood per convolution and
    pooling it with every max pool."""
    idx = np.random.default_rng(6400).choice(len(val_images()[0]), n_images, replace=False)
    target, layer, direction = 6, 0, (0, -1)       # tile 6 receives tile 5's column
    T = SIDE // GRID
    y0, x0 = T + 12, 2 * T - 1                     # the changed input pixel
    pred = np.zeros((SIDE, SIDE), bool)
    pred[y0, x0] = True
    tile6 = np.zeros((SIDE, SIDE), bool)
    tile6[T:2 * T, 2 * T:3 * T] = True
    predicted = []
    for k in range(6):
        m = _dilate(pred)
        if k == 0:
            m &= tile6                             # only tile 6 received it
        predicted.append(m)
        pred = (m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).any(axis=(1, 3))
                if k in POOL_AFTER else m)
    measured, escaped = [[] for _ in range(6)], 0
    for i in idx:
        img = image(int(i))
        clean, _ = run_sync(img)
        dirty, _ = run_sync(img, corrupt=(target, layer, direction, (0, 12, 0), delta))
        for k in range(6):
            diff = cp.abs(assemble(dirty, k) - assemble(clean, k)).max(axis=0) > 0
            d = cp.asnumpy(diff)
            measured[k].append(int(d.sum()))
            escaped += int((d & ~predicted[k]).sum())
    rows = [{"layer": k, "side": int(predicted[k].shape[0]),
             "predicted_cells": int(predicted[k].sum()),
             "mean_differing_cells": float(np.mean(measured[k])),
             "max_differing_cells": int(np.max(measured[k]))} for k in range(6)]
    return {"rows": rows, "cells_outside_cone": escaped, "images": n_images,
            "delta": delta}


def fault_injection():
    """Each injected message, the class receive_halo gives it, and the output."""
    P = params()
    img = image(0)
    f1, _ = cnn_forward(0, img, P, _SRC)
    tiles = split(img)
    pads, exchange = SCHEDULES["every layer"]
    states = [open_tile("q", t, GRID, tiles[t], f"{t}.0", pads, exchange)
              for t in range(TILES)]
    for s in states:
        advance(s, P, 0, _SRC)
    rows, injected = [], False

    def inject(name, target, commit, expected):
        status, _ = receive_halo(states[target], commit, P, 0, _SRC)
        rows.append({"injection": name, "expected": expected, "status": status})
    while True:
        msgs = [(t, c) for s in states for t, c in ready_strips(s)]
        if not msgs:
            break
        for target, c in msgs:
            if not injected and c.tile == 5 and target == 6:
                injected = True
                bad = c.strip.copy()
                bad[0, 0, 0] = np.nan
                H = HaloCommit
                inject("a non-finite value", target, H("q", c.layer, 5, c.direction, "5.0", bad, c.cells), "rejected")
                inject("the wrong shape", target, H("q", c.layer, 5, c.direction, "5.0", c.strip[:, :-1], c.cells), "rejected")
                inject("another request", target, H("q-old", c.layer, 5, c.direction, "5.0", c.strip, c.cells), "rejected")
                inject("a tile that is not that neighbour", target, H("q", c.layer, 4, c.direction, "4.0", c.strip, c.cells), "rejected")
                inject("the strip itself", target, c, "accepted")
                inject("the same attempt again", target, c, "duplicate")
                inject("a replacement's strip after the original's", target,
                       H("q", c.layer, 5, c.direction, "5.1", c.strip, c.cells), "stale")
                continue
            receive_halo(states[target], c, P, 0, _SRC)
    return {"rows": rows, "output_bitwise": bool(cp.array_equal(assemble(states, 5), f1)),
            "all_as_expected": all(r["expected"] == r["status"] for r in rows)}


def granularity(reps=REPETITIONS):
    """Whole strips against exact cells: messages and bytes with every tile present,
    and the exact area reached while one interior tile is missing (no spare)."""
    rows = []
    images = chosen_images(reps)
    for g in ("strip", "cells"):
        msgs, nbytes, t100, area = [], [], [], []
        for r in range(reps):
            img = image(int(images[r]))
            v = run_vhc(img, make_trace(6100 + r), granularity=g)
            msgs.append(v["messages"])
            nbytes.append(v["bytes"])
            t100.append(v["final_at"])
            m = run_vhc(img, make_trace(6100 + r), fault=Fault(gone={5: 0.0}),
                        spares=0, observe=2.0, granularity=g)
            area.append(float(area_curve(m["area"], np.array([2.0]))[0]))
        rows.append({"granularity": g, "messages": float(np.mean(msgs)),
                     "halo_bytes": float(np.mean(nbytes)), "t100": _mean(t100),
                     "exact_area_one_missing": float(np.mean(area))})
    return {"rows": rows, "independent_bound": float(independent_fraction([5]))}


def independence_maps():
    """The final feature map's exact cells with tile 5 missing and no spare, for
    each granularity, and the geometric bound."""
    img = image(int(chosen_images(1)[0]))
    maps = {}
    for g in ("cells", "strip"):
        v = run_vhc(img, make_trace(6100), fault=Fault(gone={5: 0.0}), spares=0,
                    observe=2.0, granularity=g)
        maps[g] = v["done_map"].astype(int).tolist()
    T = SIDE // GRID
    unknown = np.zeros((SIDE, SIDE), bool)
    unknown[T:2 * T, T:2 * T] = True
    for k in range(6):
        unknown = _dilate(unknown)
        if k in POOL_AFTER:
            n = unknown.shape[0] // 2
            unknown = unknown.reshape(n, 2, n, 2).any(axis=(1, 3))
    maps["bound"] = (~unknown).astype(int).tolist()
    return maps
