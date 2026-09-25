"""Chapter 4, sections 4.1 to 4.5: the experiment harness behind the chapter's results.

Plumbing for the notebook rather than a listing. It admits contributions with the
chapter's `validate_histogram` (listing 4.8) and merges them with `bounded_merge`
(listing 4.9), over seeded epochs in which completion times and dropout stand in
for a wide-area network.

    distributed_round(...)      the all-worker model: every worker must arrive by
                                the deadline with a valid contribution
    decentralized_epoch(...)    the epoch of listing 4.10 in simulated time: admit
                                in arrival order, close at alpha * t_k, the
                                deadline, or full participation, then merge the
                                frozen accepted set

and one function per result the chapter reports, in chapter order:

    race()                      figures 4.2 and 4.4
    reference()                 table 4.1 and figure 4.5
    variance()                  table 4.2
    statistical_skew()          table 4.3
    system_skew()               table 4.4
    one_adversary()             table 4.5
    coalition()                 table 4.6 and the coalition figure of 4.5.2
    shapes()                    figure 4.6
    one_worker_behaviors()      table 4.7 and the influence-limit costs of 4.5.1
    representation()            table 4.8
    selection_bias()            table 4.9 and figure 4.9

Both models see the same completion times and participation draw in every epoch,
so only the coordination rule changes. Time is in units of the median worker's
completion time.

    evidence class: simulated_decentralized
    The counts, admission checks and merges are real arithmetic, and each worker's
    vector comes from the chapter's kernel. Arrival times, dropout and adversarial
    behavior are seeded, not measured on a network.
"""

from __future__ import annotations

import numpy as np

from bounded_merge import bounded_merge
from contribution import HistogramContribution, validate_histogram
from keystream import HEAVY_K, build_stream

K, ALPHA, DEADLINE, TAU = 12, 1.5, 2.0, 3.0
# Every block that runs the baseline timing uses the same 300 seeded epochs, so
# table 4.2's baseline row, table 4.3's Zipf 1.2 row and the epochs behind table
# 4.5 and figure 4.6 are one run and print the same numbers.
EPOCHS = 300
TIMING_SEED = 20260906
ACTOR_SEED = 20260824
TARGET_RANK = 200          # the target bin is the 201st most frequent


# ------------------------------------------------------------------ metrics

def tvd(a, b):
    """Total-variation distance between two count vectors, after normalizing."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def top_k_recall(truth, hist, k=HEAVY_K):
    """Fraction of the true k heaviest bins still among the result's k heaviest."""
    t = set(np.argsort(truth)[::-1][:k].tolist())
    g = set(np.argsort(hist)[::-1][:k].tolist())
    return len(t & g) / k


def deviation_and_bias(truth, committed):
    """Equation 4.4: the mean distance of each committed histogram from truth, and
    the distance of their mean from truth."""
    p = np.asarray(truth, dtype=np.float64) / truth.sum()
    rows = np.array([h / h.sum() for h in committed], dtype=np.float64)
    deviation = float(np.mean([0.5 * np.abs(p - r).sum() for r in rows]))
    bias = float(0.5 * np.abs(p - rows.mean(axis=0)).sum())
    return deviation, bias


def target_bin(truth):
    return int(np.argsort(truth)[::-1][TARGET_RANK])


# ------------------------------------------------------------------ actors

def byzantine(h, target):
    """A well-formed lie: move every event this worker holds into `target`.

    Events are taken from the worker's other bins, largest first, so the total it
    declared is preserved and every check in listing 4.8 still passes.
    """
    out = np.asarray(h, dtype=np.int64).copy()
    remaining = int(out.sum())
    moved = 0
    for b in np.argsort(out)[::-1]:
        if remaining <= 0:
            break
        if b == target:
            continue
        take = min(int(out[b]), remaining)
        out[b] -= take
        remaining -= take
        moved += take
    out[target] += moved
    return out


def malformed(h):
    """A broken contribution: half the vector is missing, so its shape is wrong."""
    h = np.asarray(h, dtype=np.int64)
    return h[: max(1, h.size // 2)].copy()


def published(stream, target, byzantine_ids=(), malformed_ids=()):
    """Every worker's published vector: honest unless named otherwise."""
    out = {}
    for w in stream.workers:
        h = stream.partial(w.worker_id)
        if w.worker_id in byzantine_ids:
            out[w.worker_id] = byzantine(h, target)
        elif w.worker_id in malformed_ids:
            out[w.worker_id] = malformed(h)
        else:
            out[w.worker_id] = np.asarray(h, dtype=np.int64).copy()
    return out


# ------------------------------------------------------------------ epochs

def timing(rng, loads, spread, dropout=0.0, load_coupled=False):
    """One epoch's completion times (lognormal, median 1.0) and who arrives at all.

    With `load_coupled`, a worker's median scales with its share of the events.
    """
    n = len(loads)
    scale = (loads / loads.mean()) if load_coupled else np.ones(n)
    times = scale * np.exp(rng.normal(0.0, spread, size=n))
    return times, rng.random(n) >= dropout


def distributed_round(stream, vectors, times, present, deadline=DEADLINE):
    """The all-worker model. Returns (histogram or None, rejected contributions)."""
    if not present.all() or float(times.max()) > deadline:
        return None, None
    bad = sum(0 if _valid_shape(vectors[w.worker_id], stream.num_bins) else 1
              for w in stream.workers)
    if bad:
        return None, bad
    total = np.zeros(stream.num_bins, dtype=np.int64)
    for w in stream.workers:
        total += vectors[w.worker_id]
    return total, 0


def _valid_shape(h, num_bins):
    """Listing 4.5's structural check: shape, integer type, no negative count."""
    h = np.asarray(h)
    return (h.shape == (num_bins,) and np.issubdtype(h.dtype, np.integer)
            and not np.any(h < 0))


def decentralized_epoch(stream, vectors, times, present, tau=None,
                        k=K, alpha=ALPHA, deadline=DEADLINE):
    """Listing 4.10's epoch in simulated time.

    Contributions are received in arrival order. Before the k-th valid one the
    epoch closes at the deadline; after it, at alpha * t_k, never later than the
    deadline; and at once if every eligible worker has been accepted. A
    contribution arriving at or after the close is late and never checked.
    """
    counts = {w.worker_id: w.event_count for w in stream.workers}
    n = len(stream.workers)
    accepted, rejected, t_k, close = {}, 0, None, None

    def close_at():
        return deadline if t_k is None else min(alpha * t_k, deadline)

    for i in np.argsort(times):
        if not present[i]:
            continue
        t = float(times[i])
        if close is None and t >= close_at():
            close = close_at()
        if close is not None:
            continue                                   # late
        wid = stream.workers[i].worker_id
        h = vectors[wid]
        c = HistogramContribution(wid, h, int(np.asarray(h).sum()))
        if not validate_histogram(c, stream.num_bins, counts[wid]):
            rejected += 1
            continue
        accepted[wid] = c.histogram
        if t_k is None and len(accepted) >= k:
            t_k = t
        if len(accepted) == n:
            close = t
    if close is None:
        close = close_at()

    merged = bounded_merge(accepted, counts, tau) if len(accepted) >= k else None
    events = sum(counts[w] for w in accepted)
    return {"merged": merged, "close": close, "accepted": sorted(accepted),
            "rejected": rejected,
            "event_coverage": events / stream.total_events()}


def _arm_stats(stream, vectors, draws, arm, truth, target=None, tau=None):
    """One model over a block of epochs, averaged over the epochs that completed."""
    done, cov, tvds, ratios, rejected = 0, [], [], [], []
    for times, present in draws:
        if arm == "distributed":
            hist, bad = distributed_round(stream, vectors, times, present)
            if bad is not None:
                rejected.append(bad)
            ec = 1.0
        else:
            r = decentralized_epoch(stream, vectors, times, present, tau=tau)
            hist, ec = r["merged"], r["event_coverage"]
            rejected.append(r["rejected"])
        if hist is None:
            continue
        done += 1
        cov.append(ec)
        tvds.append(tvd(hist, truth))
        if target is not None:
            ratios.append(hist[target] / max(truth[target], 1))
    return {
        "completed": done / len(draws),
        "event_coverage": float(np.mean(cov)) if cov else None,
        "distance": float(np.mean(tvds)) if tvds else None,
        "target_ratio": float(np.mean(ratios)) if ratios else None,
        "rejected_per_round": float(np.mean(rejected)) if rejected else 0.0,
    }


def _vectors(stream):
    return {w.worker_id: stream.partial(w.worker_id) for w in stream.workers}


def _loads(stream):
    return np.array([w.event_count for w in stream.workers], dtype=float)


# ------------------------------------------------------------------ 4.1

def race(n=1_000_000, seed=42):
    """Figures 4.2 and 4.4: the naive kernel against CPU-exact, then the worker kernel."""
    from partial_histogram import naive_histogram, partial_histogram

    probabilities = np.array([0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.20, 0.60])
    keys = np.random.default_rng(seed).choice(8, size=n, p=probabilities).astype(np.int32)
    exact = np.bincount(keys, minlength=8).astype(np.int64)
    naive = naive_histogram(keys, 8)
    worker = partial_histogram(keys, 8)[1].get().astype(np.int64)
    return {"exact": exact, "naive": naive, "worker": worker}


# ------------------------------------------------------------------ 4.2

def reference():
    """Table 4.1 and figure 4.5."""
    s = build_stream()
    truth = s.true_hist()
    top = np.sort(truth)[::-1]
    nnz = [s.distinct_keys(w.worker_id) for w in s.workers]
    return {
        "stream": s, "truth": truth,
        "workers": len(s.workers), "events": s.total_events(), "bins": s.num_bins,
        "distinct_keys": int(np.count_nonzero(truth)),
        "hottest_share": float(top[0] / truth.sum()),
        "top32_share": float(top[:HEAVY_K].sum() / truth.sum()),
        "batch_min": min(w.event_count for w in s.workers),
        "batch_max": max(w.event_count for w in s.workers),
        "busiest_key_owner_load": float(top[0] / truth.sum() * len(s.workers)),
        "median_distinct_keys": int(np.median(nnz)),
        "computed_by": s.computed_by,
    }


def variance(epochs=EPOCHS):
    """Table 4.2: the same epochs under three timing conditions."""
    s = build_stream()
    truth, vectors, loads = s.true_hist(), _vectors(s), _loads(s)
    rows = []
    for name, spread, dropout in (("baseline timing", 0.35, 0.0),
                                  ("10% dropout", 0.35, 0.10),
                                  ("heavy tail", 0.90, 0.0)):
        rng = np.random.default_rng(TIMING_SEED)
        draws = [timing(rng, loads, spread, dropout) for _ in range(epochs)]
        rows.append({"condition": name,
                     "distributed": _arm_stats(s, vectors, draws, "distributed", truth),
                     "decentralized": _arm_stats(s, vectors, draws, "decentralized", truth)})
    return rows


def statistical_skew(epochs=EPOCHS):
    """Table 4.3: the key distribution concentrates; the shards stay the same."""
    rows = []
    for alpha in (0.6, 1.2, 1.6):
        s = build_stream(alpha=alpha)
        truth, vectors, loads = s.true_hist(), _vectors(s), _loads(s)
        rng = np.random.default_rng(TIMING_SEED)
        draws = [timing(rng, loads, 0.35) for _ in range(epochs)]
        rows.append({"alpha": alpha, "hottest_share": float(truth.max() / truth.sum()),
                     "distributed": _arm_stats(s, vectors, draws, "distributed", truth),
                     "decentralized": _arm_stats(s, vectors, draws, "decentralized", truth)})
    return rows


def system_skew(epochs=EPOCHS):
    """Table 4.4: completion time scales with shard size; some workers run 3x slower."""
    s = build_stream()
    truth, vectors, loads = s.true_hist(), _vectors(s), _loads(s)

    def capacity(slow):
        return np.array([3.0 if w.worker_id in slow else 1.0 for w in s.workers])

    rows = []
    for name, cap in (("Uneven shards", capacity(set())),
                      ("Four slow workers", capacity({12, 13, 14, 15})),
                      ("Eight slow workers", capacity(set(range(8, 16))))):
        rng = np.random.default_rng(TIMING_SEED)
        draws = []
        for _ in range(epochs):
            times, present = timing(rng, loads, 0.35, load_coupled=True)
            draws.append((times * cap, present))
        rows.append({"condition": name,
                     "distributed": _arm_stats(s, vectors, draws, "distributed", truth),
                     "decentralized": _arm_stats(s, vectors, draws, "decentralized", truth)})
    return rows


def one_adversary(epochs=EPOCHS):
    """Table 4.5: worker 3 is malformed or Byzantine. The influence limit is on."""
    s = build_stream()
    truth, loads = s.true_hist(), _loads(s)
    target = target_bin(truth)
    rows = []
    for case, vectors in (("Malformed", published(s, target, malformed_ids={3})),
                          ("Byzantine", published(s, target, byzantine_ids={3}))):
        rng = np.random.default_rng(TIMING_SEED)
        draws = [timing(rng, loads, 0.35) for _ in range(epochs)]
        for arm, tau in (("Distributed", None), ("Decentralized", TAU)):
            st = _arm_stats(s, vectors, draws, arm.lower(), truth, target, tau)
            rows.append({"case": case, "model": arm, **st})
    return {"rows": rows, "target": target, "target_true": int(truth[target]),
            "worker3_events": s.workers[3].event_count,
            "moved": int(published(s, target, byzantine_ids={3})[3][target]
                         - s.partial(3)[target])}


def coalition():
    """Table 4.6 and the coalition figure, one sweep: every worker present and on time."""
    s = build_stream()
    truth = s.true_hist()
    target = target_bin(truth)
    times = np.ones(len(s.workers))
    present = np.ones(len(s.workers), dtype=bool)
    rows = []
    for size in range(17):
        vectors = published(s, target, byzantine_ids=set(range(size)))
        dist, _ = distributed_round(s, vectors, times, present)
        dec = decentralized_epoch(s, vectors, times, present, tau=TAU)["merged"]
        rows.append({"colluding": size,
                     "distributed": dist[target] / truth[target],
                     "decentralized": dec[target] / truth[target]})
    held = max(r["colluding"] for r in rows if r["decentralized"] < 10.0)
    return {"rows": rows, "holds_through": held}


def shapes():
    """Figure 4.6: one epoch of table 4.5's run, bin by bin.

    The first epoch of the same seeded draws in which all sixteen workers arrive by
    the deadline, so both models merge every worker. Table 4.5 averages the target
    bin over every completed epoch; this is one of them.
    """
    s = build_stream()
    truth, loads = s.true_hist(), _loads(s)
    target = target_bin(truth)
    honest = published(s, target)
    rng = np.random.default_rng(TIMING_SEED)
    for _ in range(64):
        times, present = timing(rng, loads, 0.35)
        if distributed_round(s, honest, times, present)[0] is not None:
            break
    bins = sorted(set(np.argsort(truth)[::-1][:8].tolist() + [target]))
    out = {"bins": bins, "target": target, "cpu": [int(truth[b]) for b in bins]}
    for case, vectors in (("honest", honest),
                          ("one Byzantine", published(s, target, byzantine_ids={3}))):
        dist, _ = distributed_round(s, vectors, times, present)
        dec = decentralized_epoch(s, vectors, times, present, tau=TAU)["merged"]
        out[case] = {"distributed": [int(dist[b]) for b in bins],
                     "decentralized": [int(dec[b]) for b in bins]}
    return out


# ------------------------------------------------------------------ 4.5

def _fixed_round(s, vectors, tau):
    """Sixteen workers, all present and on time, arriving at 0.1, 0.2, ..., 1.6."""
    times = 0.1 * np.arange(1, len(s.workers) + 1)
    present = np.ones(len(s.workers), dtype=bool)
    return decentralized_epoch(s, vectors, times, present, tau=tau, deadline=10.0)


def one_worker_behaviors():
    """Table 4.7, and what the influence limit costs an honest fleet (section 4.5.1)."""
    s = build_stream()
    truth = s.true_hist()
    target = target_bin(truth)
    rows = []
    for name, vectors in (("honest", published(s, target)),
                          ("malformed", published(s, target, malformed_ids={3})),
                          ("Byzantine", published(s, target, byzantine_ids={3}))):
        off, on = _fixed_round(s, vectors, None), _fixed_round(s, vectors, TAU)
        rows.append({
            "behavior": name,
            "admission": "accepted" if 3 in off["accepted"] else "rejected",
            "target_off": int(off["merged"][target]), "target_on": int(on["merged"][target]),
            "tvd_off": tvd(off["merged"], truth), "tvd_on": tvd(on["merged"], truth),
            "recall_off": top_k_recall(truth, off["merged"]),
            "recall_on": top_k_recall(truth, on["merged"]),
        })

    honest = published(s, target)
    exact_sum = int(truth.sum())
    removed = {tau: exact_sum - int(_fixed_round(s, honest, tau)["merged"].sum())
               for tau in (TAU, 1000.0)}
    heavy = np.argsort(truth)[::-1][:HEAVY_K]
    at3 = _fixed_round(s, honest, TAU)["merged"]
    rates = np.stack([honest[w.worker_id] / w.event_count for w in s.workers])
    zero_median = np.median(rates, axis=0) == 0
    return {
        "rows": rows, "target": target, "target_true": int(truth[target]),
        "target_share": float(truth[target] / truth.sum()),
        "removed_tau3": removed[TAU],
        "removed_from_heavy_bins": int((truth[heavy] - at3[heavy]).sum()),
        "zero_median_bin_share": float(zero_median.mean()),
        "zero_median_event_share": float(truth[zero_median].sum() / truth.sum()),
        "recovered_tau3_to_1000": removed[TAU] - removed[1000.0],
    }


class MisraGries:
    """A capacity-bounded frequent-items map (section 4.5.3's bounded summary).

    Holds at most `capacity` keys. A new key arriving at a full map pays for its
    place by decrementing every retained counter by the smallest one, and keys
    that reach zero are dropped.
    """

    def __init__(self, capacity):
        self.capacity, self.counts = capacity, {}

    def update(self, key, weight):
        if key in self.counts:
            self.counts[key] += weight
        elif len(self.counts) < self.capacity:
            self.counts[key] = weight
        else:
            paid = min(min(self.counts.values()), weight)
            for k in list(self.counts):
                self.counts[k] -= paid
                if self.counts[k] <= 0:
                    del self.counts[k]
            if weight > paid and len(self.counts) < self.capacity:
                self.counts[key] = weight - paid

    @classmethod
    def from_histogram(cls, h, capacity):
        """Offer a worker's nonzero bins to the map from heaviest to lightest."""
        h = np.asarray(h)
        nz = np.flatnonzero(h)
        mg = cls(capacity)
        for k in nz[np.argsort(-h[nz], kind="stable")].tolist():
            mg.update(int(k), int(h[k]))
        return mg

    def to_dense(self, num_bins):
        out = np.zeros(num_bins, dtype=np.int64)
        for k, c in self.counts.items():
            out[k] = c
        return out


def representation(capacity=256):
    """Table 4.8: dense, sparse and bounded payloads as the key universe grows."""
    rows = []
    for bins in (1024, 4096, 16384, 32768, 65536):
        s = build_stream(num_bins=bins)
        truth = s.true_hist()
        h = s.partial(s.workers[0].worker_id)          # the largest worker
        nnz = int(np.count_nonzero(h))
        merged = sum(MisraGries.from_histogram(s.partial(w.worker_id), capacity)
                     .to_dense(bins) for w in s.workers)
        rows.append({
            "bins": bins, "fill": nnz / bins, "dense": bins * 4, "sparse": nnz * 8,
            "bounded": min(capacity, nnz) * 8,
            "recall": top_k_recall(truth, merged), "tvd": tvd(merged, truth),
            "computed_by": s.computed_by,
        })
    return rows


def selection_bias(trials=400):
    """Table 4.9 and figure 4.9: the same coverage, three different dropout patterns."""
    rows = []
    for name, correlate, p_even, p_odd in (("uniform dropout", False, 0.75, 0.75),
                                           ("differential dropout", False, 0.95, 0.55),
                                           ("differential and correlated", True, 0.95, 0.55)):
        s = build_stream(correlate_keys_with_speed=correlate)
        truth = s.true_hist()
        q = {w.worker_id: (p_even if w.group == 0 else p_odd) for w in s.workers}
        predicted = sum(q[w.worker_id] * s.partial(w.worker_id) for w in s.workers)
        vectors = _vectors(s)
        rng = np.random.default_rng(ACTOR_SEED)
        committed, wcov, ecov = [], [], []
        for _ in range(trials):
            present = np.array([rng.random() < q[w.worker_id] for w in s.workers])
            if present.sum() < K:
                continue
            times = 0.1 * np.arange(1, len(s.workers) + 1)
            r = decentralized_epoch(s, vectors, times, present, deadline=10.0)
            if r["merged"] is None:
                continue
            committed.append(r["merged"])
            wcov.append(len(r["accepted"]) / len(s.workers))
            ecov.append(r["event_coverage"])
        deviation, bias = deviation_and_bias(truth, committed)
        rows.append({"regime": name, "worker_coverage": float(np.mean(wcov)),
                     "event_coverage": float(np.mean(ecov)), "deviation": deviation,
                     "bias": bias, "predicted_bias": tvd(predicted, truth),
                     "committed_rounds": len(committed)})
    return rows
