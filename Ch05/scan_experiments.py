"""Chapter 5, sections 5.1 to 5.5: the experiment harness behind the chapter's results.

Plumbing for the notebook rather than a listing. Every coordinator decision is made
by the chapter's own functions: `receive` and `advance` (listing 5.8),
`check_for_gap` (listing 5.9) and `recover` (listing 5.12). Every block is scanned
by `scan_launcher` (listing 5.3) and finalized by `finalize_block` (listing 5.4).
What the harness supplies is time: seeded worker finish times, failures and
replacement delays, replayed in order as listing 5.11's loop would see them.

    run_cpr(...)              one CPR run in simulated time on one arrival trace
    distributed_outcome(...)  the distributed scan on the same trace: it completes
                              at the last arrival, or never if a block is missing

and one function per result the chapter reports, in chapter order:

    down_sweep()              table 5.1
    settings()                table 5.2
    worker_scan_time()        table 5.2, the measured row
    missing_blocks()          table 5.3, figure 5.3, and the unfinalized areas and
                              normalized error quoted in section 5.2.3
    participants()            table 5.4, and the permanently missing and late
                              participant results quoted after it
    repair_deadline()         table 5.5
    lookahead()               table 5.6
    checkpoint_interval()     table 5.7

Both arms of every comparison see the same input and the same arrival times, so
only the coordination rule changes. Time is in seconds.

    evidence class: simulated_decentralized
    The scans, carries and finalized outputs are real arithmetic, computed by the
    chapter's kernels (on the GPU when one is present). Finish times, failures and the replacement delay are seeded
    draws, not measured on a network.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from cpr_coordinator import check_for_gap, receive
from finalize_block import finalize_block
from prefix_repair_state import PrefixRepairState, SummaryCommit
from recover import recover
from scan_launcher import _SRC, gpu_available, scan_launcher

N_BLOCKS = 64
BLOCK_LEN = 4096
OBSERVE = 5.0             # the working example's observation window, in seconds
REPAIR_DEADLINE = 0.75
REPAIR_COST = 0.45        # a replacement worker reports this long after assignment
LOOKAHEAD = 64
CHECKPOINT_INTERVAL = 16
RETRY = 0.05              # a deferred commit is offered again after this long
REPETITIONS = 20
EPOCHS = 2000
HORIZON = 600.0           # a run with no repair source stops here
TOL = 1e-6                # normalized error allowed against the float64 scan
REGIONS = ("early", "middle", "late")


# ------------------------------------------------------------------ the input

def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class Blocks:
    """The logical blocks of one run, each scanned once by its worker.

    `summaries[s]` is what worker s reports. `finalize(s, carry)` is listing 5.4 on
    worker s's local scan, returned to the host.
    """

    def __init__(self, raw):
        self.raw = [np.ascontiguousarray(b, dtype=np.float32) for b in raw]
        self.local, self.summaries = [], []
        for b in self.raw:
            local, summary = scan_launcher(0, b, _SRC)
            self.local.append(local)
            self.summaries.append(summary)

    def finalize(self, s, carry):
        return _host(finalize_block(0, self.local[s], carry, _SRC))

    def distributed_reference(self, n):
        """Listing 5.5's carries, every block present, applied by listing 5.4."""
        carries, carry = [], 0.0
        for summary in self.summaries[:n]:
            carries.append(carry)
            carry += summary
        return np.concatenate([self.finalize(s, c) for s, c in enumerate(carries)])

    def float64_reference(self):
        x = np.concatenate(self.raw).astype(np.float64)
        return np.concatenate(([0.0], np.cumsum(x)[:-1]))


def make_blocks(rng, n_blocks=N_BLOCKS, block_len=BLOCK_LEN):
    """Uniform [0, 1) values, drawn in float64 and stored as float32."""
    return Blocks([rng.uniform(0.0, 1.0, block_len) for _ in range(n_blocks)])


def normalized_error(got, ref):
    """max |got - ref| / max(|ref|, 1), the measure section 5.2.2 defines."""
    return float(np.max(np.abs(got.astype(np.float64) - ref) / np.maximum(np.abs(ref), 1.0)))


# ------------------------------------------------------------------ arrivals

@dataclass
class Arrival:
    t: float
    block_seq: int
    attempt_id: str


@dataclass
class Trace:
    arrivals: list            # sorted by time
    dead: set                 # blocks whose worker never reports
    n_blocks: int


def make_trace(rng, n_blocks, *, base=0.10, spread=0.35, dead=(), late=(),
               late_delay=3.0, churn=0.0, stagger=0.0):
    """When each block's summary arrives. Table 5.2's infrastructure model.

    A worker finishes `base` seconds in at the median, lognormal with log-space
    spread `spread`. `dead` blocks never arrive. `late` blocks arrive `late_delay`
    seconds behind. Every other block is slow with probability `churn`, by 0.4 to
    1.2 seconds. `stagger` spreads the blocks out in sequence order, as when a long
    input reaches its workers over time.
    """
    arrivals = []
    for s in range(n_blocks):
        if s in dead:
            continue
        t = stagger * s + base * rng.lognormal(0.0, spread)
        if s in late:
            t += late_delay
        elif churn > 0 and rng.random() < churn:
            t += base * rng.uniform(4.0, 12.0)
        arrivals.append(Arrival(t, s, f"{s}.0"))
    arrivals.sort(key=lambda a: a.t)
    return Trace(arrivals, set(dead), n_blocks)


def epoch_trace(rng, n, *, base=0.10, spread=0.35, dropout=0.0, dead=None,
                late=None, late_delay=3.0):
    """One epoch of the participant sweep. Every participant draws a finish time
    whether or not it drops out, so later epochs do not depend on who failed."""
    arrivals, gone = [], set()
    for s in range(n):
        missing = (s == dead) or (dropout > 0 and rng.random() < dropout)
        t = base * rng.lognormal(0.0, spread)
        if s == late:
            t += late_delay
        if missing:
            gone.add(s)
        else:
            arrivals.append(Arrival(t, s, f"{s}.0"))
    arrivals.sort(key=lambda a: a.t)
    return Trace(arrivals, gone, n)


def region_blocks(r, n_blocks=N_BLOCKS):
    """The failed blocks of repetition r: one drawn from each third of the sequence."""
    rng = np.random.default_rng(7000 + r)
    cuts = np.linspace(0, n_blocks, len(REGIONS) + 1).round().astype(int)
    return {name: int(rng.integers(cuts[i], cuts[i + 1])) for i, name in enumerate(REGIONS)}


# ------------------------------------------------------------- the two arms

def distributed_outcome(trace, observe=None):
    """The distributed scan: complete at the last arrival, or never. Communication
    time is not modeled, so this is the earliest it could finish."""
    if trace.dead or len(trace.arrivals) < trace.n_blocks:
        return None
    t = max(a.t for a in trace.arrivals)
    return None if (observe is not None and t > observe) else t


def _check_for_gap_at(state, t):
    """`check_for_gap` at the instant the deadline expires. The deadline time is
    blocked_since + repair_deadline, and subtracting blocked_since back can round
    a hair below the deadline, so step to the next representable time if needed."""
    for _ in range(4):
        s = check_for_gap(state, t)
        if s is not None:
            return s
        t = math.nextafter(t, math.inf)
    raise RuntimeError("the gap deadline did not fire")


def run_cpr(summaries, trace, *, repair_deadline=REPAIR_DEADLINE, lookahead=LOOKAHEAD,
            checkpoint_interval=CHECKPOINT_INTERVAL, observe=None, crash_at=None):
    """One CPR run in simulated time.

    Events are taken in time order: the next arrival, the next gap deadline, and
    an optional coordinator restart. A gap reported by `check_for_gap` is repaired
    by a replacement attempt that arrives REPAIR_COST seconds later. A deferred
    commit is offered again after RETRY seconds. A restart replaces the state with
    `recover(state, now)`.
    """
    n = trace.n_blocks
    state = PrefixRepairState("ch05", n, lookahead, repair_deadline, checkpoint_interval)
    pq = [(a.t, i, a) for i, a in enumerate(trace.arrivals)]
    heapq.heapify(pq)
    seq = len(pq)
    emitted = {}                               # block -> every carry it was given
    history = [(0.0, -1)]                      # (time, frontier) after each event
    repairs, max_pending, replayed = [], 0, 0
    crashed = False
    area, prev_t, t = 0.0, 0.0, 0.0
    limit = observe if observe is not None else HORIZON

    def record(finalized):
        for b in finalized:
            emitted.setdefault(b, []).append(state.carries[b])

    while t < limit and state.frontier + 1 < n:
        s = state.frontier + 1
        blocked = s not in state.summaries
        gap_at = (state.blocked_since + repair_deadline
                  if blocked and state.repairing != s else math.inf)
        next_arrival = pq[0][0] if pq else math.inf
        crash_t = crash_at if (crash_at is not None and not crashed) else math.inf

        step = min(gap_at, next_arrival, crash_t, limit)
        area += max(0.0, step - prev_t) * (n - (state.frontier + 1))
        prev_t = step
        if step >= limit and not (next_arrival <= limit or crash_t <= limit):
            t = step
            break
        t = step

        if crash_t <= min(gap_at, next_arrival):
            crashed = True
            replayed = len(state.summary_log)
            state, finalized = recover(state, t)
            record(finalized)
            history.append((t, state.frontier))
            continue

        if gap_at <= next_arrival:
            g = _check_for_gap_at(state, t)
            repairs.append((g, t))
            heapq.heappush(pq, (t + REPAIR_COST, seq, Arrival(t + REPAIR_COST, g, f"{g}.1")))
            seq += 1
            continue

        _, _, a = heapq.heappop(pq)
        commit = SummaryCommit("ch05", a.block_seq, a.attempt_id, summaries[a.block_seq])
        outcome, finalized = receive(state, commit, t)
        if outcome == "deferred":
            heapq.heappush(pq, (t + RETRY, seq, a))
            seq += 1
            continue
        record(finalized)
        history.append((t, state.frontier))
        max_pending = max(max_pending, sum(1 for b in state.summaries if b > state.frontier))

    if observe is not None and prev_t < limit:
        area += (limit - prev_t) * (n - (state.frontier + 1))

    t100 = next((h for h, f in history if f >= n - 1), None)
    return {"state": state, "t100": t100, "history": history, "repairs": repairs,
            "unfinalized_area": area, "max_pending": max_pending, "replayed": replayed,
            "emitted": emitted}


def check_exact(blocks, run):
    """Rebuild the finalized output from the carries the run emitted (the first,
    if a block was finalized twice) and compare it with listing 5.5 bitwise and
    with the float64 scan."""
    n = run["state"].frontier + 1
    if n == 0:
        return True, 0.0
    got = np.concatenate([blocks.finalize(s, run["emitted"][s][0]) for s in range(n)])
    bitwise = bool(np.array_equal(got, blocks.distributed_reference(n)))
    return bitwise, normalized_error(got, blocks.float64_reference()[:got.size])


# --------------------------------------------------------------- the results

def down_sweep(values=(2, 6, 1, 3, 5, 2, 4, 7)):
    """Table 5.1: the down-sweep of figure 5.2, level by level.

    Listing 5.1 keeps its tree in shared memory, where the host cannot see it, so
    this tabulates the intermediate values by stepping through the same two passes.
    Each row is one parent: its value, its left child's (a copy), and its right
    child's (parent plus the left sibling's up-sweep sum). The leaves are then
    checked against listing 5.1 run on the GPU.
    """
    n = len(values)
    temp = np.array(values, dtype=np.float32)
    offset = 1
    while offset < n:                                       # up-sweep, in place
        i = np.arange(2 * offset - 1, n, 2 * offset)
        temp[i] += temp[i - offset]
        offset <<= 1
    temp[n - 1] = 0.0
    names = {n // 2: "Root to halves", n // 4: "Halves to quarters", n // 8: "Quarters to leaves"}
    rows, offset = [], n >> 1
    while offset > 0:
        for i in range(2 * offset - 1, n, 2 * offset):
            parent, sibling = int(temp[i]), int(temp[i - offset])
            rows.append({"level": names.get(offset, f"offset {offset}"), "parent": parent,
                         "left": parent, "right": parent + sibling, "added": sibling})
        i = np.arange(2 * offset - 1, n, 2 * offset)
        left = temp[i - offset].copy()
        temp[i - offset] = temp[i]
        temp[i] += left
        offset >>= 1
    leaves = temp.astype(int).tolist()
    kernel, _ = scan_launcher(0, list(values), _SRC, tile=n)
    return {"rows": rows, "leaves": leaves,
            "matches_listing_5_1": leaves == _host(kernel).astype(int).tolist()}


def settings():
    """Table 5.2: the working example's workload, infrastructure model and settings."""
    return [
        ("input", f"{N_BLOCKS * BLOCK_LEN:,} float32 values, uniform on [0, 1)"),
        ("logical blocks and workers", f"{N_BLOCKS} blocks of {BLOCK_LEN:,} values, one worker per block"),
        ("worker finish time", "0.10 s median, lognormal with log-space spread 0.35"),
        ("slow workers", "each block has a 5% chance of an extra 0.4 to 1.2 s (25% under heavy churn)"),
        ("failures", "none; one early, middle or late block never reports; one early and one "
                     "middle block never report; one early block reports 2 s late"),
        ("failed block positions", "drawn at random in each repetition, one from each third of "
                                   "the sequence: early 0-20, middle 21-42, late 43-63"),
        ("CPR repair deadline", f"{REPAIR_DEADLINE} s"),
        ("replacement worker", f"reads the block from its source and reports {REPAIR_COST} s after assignment"),
        ("CPR lookahead window", f"{LOOKAHEAD} blocks, the whole sequence"),
        ("CPR checkpoint interval", f"every {CHECKPOINT_INTERVAL} blocks"),
        ("observation window", f"{OBSERVE:g} seconds"),
        ("repetitions", f"{REPETITIONS} per condition, with new input values and finish times each time"),
        ("reported value", f"mean over the {REPETITIONS} repetitions"),
    ]


def worker_scan_time(repeats=500, seed=7):
    """Table 5.2's measured row: listing 5.3 on one 4,096-value block, timed on the
    GPU from the copy to the device to the summary read back. None without a GPU."""
    if not gpu_available():
        return None
    from cupyx.profiler import benchmark
    import cupy as cp
    x = np.random.default_rng(seed).random(BLOCK_LEN).astype(np.float32)
    r = benchmark(scan_launcher, (0, x, _SRC, 1024), n_repeat=repeats, n_warmup=20)
    gpu_ms = r.gpu_times[0] * 1e3
    return {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "median_ms": float(np.median(gpu_ms)), "p99_ms": float(np.percentile(gpu_ms, 99))}


CONDITIONS = [
    ("All blocks arrive", lambda b: dict(churn=0.05)),
    ("Early block missing", lambda b: dict(dead=(b["early"],), churn=0.05)),
    ("Middle block missing", lambda b: dict(dead=(b["middle"],), churn=0.05)),
    ("Late block missing", lambda b: dict(dead=(b["late"],), churn=0.05)),
    ("Early and middle blocks missing", lambda b: dict(dead=(b["early"], b["middle"]), churn=0.05)),
    ("Early block delayed by 2 s", lambda b: dict(late=(b["early"],), late_delay=2.0, churn=0.05)),
    ("Heavy churn, none missing", lambda b: dict(churn=0.25)),
]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def missing_blocks(reps=REPETITIONS, dt=0.01):
    """Table 5.3 and figure 5.3: the distributed scan and CPR on the same traces.

    Each repetition draws a new input and new finish times (seed 3000 + r) and one
    failed block from each third of the sequence. Figure 5.3's curves are the mean
    over the repetitions of the one-missing-block conditions, sampled every `dt`.
    """
    grid = np.round(np.arange(0.0, OBSERVE + dt / 2, dt), 6)
    rows, curves = [], {}
    worst_error, all_bitwise = 0.0, True
    for label, kw_of in CONDITIONS:
        d_t100, d_area, c_t100, c_area, c_rep = [], [], [], [], []
        received, finalized, declared = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(3000 + r)
            blocks = make_blocks(rng)
            failed = region_blocks(r)
            trace = make_trace(rng, N_BLOCKS, **kw_of(failed))

            t = distributed_outcome(trace, observe=OBSERVE)
            d_t100.append(t)
            d_area.append(N_BLOCKS * (t if t is not None else OBSERVE))
            if t is not None:
                worst_error = max(worst_error, normalized_error(
                    blocks.distributed_reference(N_BLOCKS), blocks.float64_reference()))

            run = run_cpr(blocks.summaries, trace, observe=OBSERVE)
            c_t100.append(run["t100"]); c_area.append(run["unfinalized_area"])
            c_rep.append(len(run["repairs"]))
            bitwise, err = check_exact(blocks, run)
            all_bitwise &= bitwise
            worst_error = max(worst_error, err)

            if label.endswith("block missing") and "and" not in label:
                missing = next(iter(trace.dead))
                received.append(_step([(a.t, i + 1) for i, a in enumerate(trace.arrivals)], grid))
                finalized.append(_step([(h, f + 1) for h, f in run["history"]], grid))
                declared += [h for s, h in run["repairs"] if s == missing]
        rows.append({"condition": label, "distributed_t100": _mean(d_t100),
                     "distributed_completed": all(x is not None for x in d_t100),
                     "cpr_t100": _mean(c_t100), "cpr_repairs": float(np.mean(c_rep)),
                     "distributed_area": float(np.mean(d_area)),
                     "cpr_area": float(np.mean(c_area))})
        if received:
            region = label.split()[0].lower()
            curves[region] = {"received": np.mean(received, axis=0),
                              "cpr_finalized": np.mean(finalized, axis=0),
                              "declared": float(np.mean(declared))}
    return {"rows": rows, "curves": curves, "t": grid, "max_normalized_error": worst_error,
            "bitwise_equal_to_listing_5_5": all_bitwise}


def _step(series, grid):
    """Sample a step function, given as sorted (t, value) pairs, on a time grid."""
    t = np.array([p[0] for p in series]); v = np.array([p[1] for p in series], dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], 0.0)


def participants(epochs=EPOCHS):
    """Table 5.4 and the two single-participant cases quoted after it.

    For each participant count the input is fixed (seed 900 + n) and every epoch
    draws new arrivals (seed 2000 + n). The failed participant is the last one.
    CPR runs each epoch until its prefix is finalized; there is no window.
    """
    rows = []
    for n in (16, 32, 64, 128):
        summaries = make_blocks(np.random.default_rng(900 + n), n_blocks=n).summaries
        scenarios = [("all present", {}), ("2%", dict(dropout=0.02)),
                     ("5%", dict(dropout=0.05)), ("one missing", dict(dead=n - 1)),
                     ("one 3 s late", dict(late=n - 1, late_delay=3.0))]
        for label, kw in scenarios:
            rng = np.random.default_rng(2000 + n)
            d_done, d_t, c_t = [], [], []
            for _ in range(epochs):
                trace = epoch_trace(rng, n, **kw)
                t = distributed_outcome(trace)
                d_done.append(t is not None)
                if t is not None:
                    d_t.append(t)
                c_t.append(run_cpr(summaries, trace, lookahead=n)["t100"])
            rows.append({"participants": n, "scenario": label,
                         "distributed_completion": float(np.mean(d_done)),
                         "distributed_p95": float(np.percentile(d_t, 95)) if d_t else None,
                         "cpr_completion": float(np.mean([x is not None for x in c_t])),
                         "cpr_p95": float(np.percentile([x for x in c_t if x is not None], 95))})
    return rows


def repair_deadline(reps=REPETITIONS):
    """Table 5.5: one early block's worker is slow, not dead, and reports 2 s late."""
    rows, bitwise_all = [], True
    for deadline in (0.15, 0.30, 0.75, 1.50, 3.00):
        t100, rep = [], []
        for r in range(reps):
            rng = np.random.default_rng(3500 + r)
            blocks = make_blocks(rng)
            trace = make_trace(rng, N_BLOCKS, late=(region_blocks(r)["early"],),
                               late_delay=2.0, churn=0.05)
            run = run_cpr(blocks.summaries, trace, repair_deadline=deadline)
            t100.append(run["t100"]); rep.append(len(run["repairs"]))
            bitwise_all &= check_exact(blocks, run)[0]
        rows.append({"repair_deadline": deadline, "t100": _mean(t100),
                     "repairs": float(np.mean(rep))})
    return {"rows": rows, "bitwise_equal_to_listing_5_5": bitwise_all}


def lookahead(reps=REPETITIONS):
    """Table 5.6: one early block permanently missing, W from 2 to 64."""
    rows, bitwise_all = [], True
    for W in (2, 4, 8, 16, 64):
        t100, pending = [], []
        for r in range(reps):
            rng = np.random.default_rng(3800 + r)
            blocks = make_blocks(rng)
            trace = make_trace(rng, N_BLOCKS, dead=(region_blocks(r)["early"],), churn=0.05)
            run = run_cpr(blocks.summaries, trace, lookahead=W)
            t100.append(run["t100"]); pending.append(run["max_pending"])
            bitwise_all &= check_exact(blocks, run)[0]
        rows.append({"lookahead": W, "t100": _mean(t100), "max_pending": float(np.mean(pending))})
    return {"rows": rows, "bitwise_equal_to_listing_5_5": bitwise_all}


def checkpoint_interval(reps=REPETITIONS, restart_at=1.6):
    """Table 5.7: the coordinator restarts once per run at 1.6 s and recovers with
    listing 5.12. Blocks arrive staggered, so the frontier is advancing when it
    restarts."""
    rows, bitwise_all, same_carries = [], True, True
    for C in (4, 8, 16, 32, 64):
        replayed, refinalized, t100 = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(4200 + r)
            blocks = make_blocks(rng)
            trace = make_trace(rng, N_BLOCKS, stagger=0.04, churn=0.15)
            run = run_cpr(blocks.summaries, trace, checkpoint_interval=C, crash_at=restart_at)
            replayed.append(run["replayed"])
            refinalized.append(sum(1 for v in run["emitted"].values() if len(v) > 1))
            same_carries &= all(len(set(v)) == 1 for v in run["emitted"].values())
            t100.append(run["t100"])
            bitwise_all &= check_exact(blocks, run)[0]
        rows.append({"interval": C, "replayed": float(np.mean(replayed)),
                     "refinalized": float(np.mean(refinalized)), "t100": _mean(t100)})
    return {"rows": rows, "bitwise_equal_to_listing_5_5": bitwise_all,
            "refinalized_with_the_same_carry": same_carries}
