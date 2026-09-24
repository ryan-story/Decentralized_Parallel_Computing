"""Chapter 3, section 3.2: the experiment harness behind the chapter's results.

Plumbing for the notebook rather than a listing. It drives the chapter's own
`CommitMergeEpoch` (listings 3.9 to 3.12) and `canonical_merge` (listing 3.13)
through thousands of epochs, with seeded arrival times and dropout standing in
for a wide-area network.

    run_scenario(sc)          one cell: `sc.n_epochs` epochs over one population.
                              Every epoch is also read as the listing 3.5
                              coordinator on the SAME draws: it completes only
                              when every worker is present, and then pays the
                              slowest of them.
    pooled(sc, reps=20)       the same cell over 20 arrival seeds, every
                              statistic pooled over all 1,200 epochs
    repeat(sc, reps=20)       the same, averaged per seed (table 3.11 only)

and one function per result the chapter reports, in chapter order:

    dropout_sweep()           table 3.6 and figure 3.4
    window_sweep()            table 3.8 and figures 3.7 and 3.8
    tail_sweep()              figure 3.5, the window sweep's alpha = 1.5 cells
    headline()                tables 3.4 and 3.5, read off the two sweeps

Each condition is simulated once. A number that appears in a table and on a
figure is the same number in both places.
    replay_determinism()      the 200-replay experiment of section 3.4.6
    contribution_state_checks()   table 3.10
    availability()            table 3.11

Time is in units of the median worker's completion time. Every function takes
`reduce`, which computes a worker's (sum, count) from its records. By default it
is an exact float64 host sum; the notebook passes the chapter's CUDA launcher.

    evidence class: simulated_decentralized
    The records, sums, counts and merges are real arithmetic. Arrival times,
    dropout and the network are seeded draws, not measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import commit_merge as cm
from population import arrival_times, make_population

OP_ID, SCHEMA_ID = "mean", "sumcount.v1"
MERGE_PROFILE = "sorted-by-worker-f64-intcount"


def host_reduce(values):
    """The exact (sum, count) of one worker's records."""
    return float(np.sum(values, dtype=np.float64)), int(values.size)


def worker_partials(pop, reduce=None):
    reduce = reduce or host_reduce
    return {w.worker_id: reduce(pop.records[w.worker_id]) for w in pop.workers}


@dataclass
class Scenario:
    """One experimental cell. The all-participant baseline is `k = n`."""

    n: int = 16
    k: int | None = None                 # None means k = n
    deadline: float = 2.0
    dropout: float = 0.0
    tail_sigma: float = 0.1
    dispersion: float = 1.0
    time_scale: float = 1.0              # uniform slowdown of the whole fleet
    correlated: bool = False             # slow workers also hold different data
    alpha: float = 1.0
    n_epochs: int = 60
    seed: int = 0                        # arrivals and dropout only
    pop_seed: int = 0                    # the dataset, fixed across seeds
    duplicate_rate: float = 0.0          # optional fault injection
    revision_rate: float = 0.0
    malformed_rate: float = 0.0

    @property
    def quorum(self):
        return self.n if self.k is None else self.k


def _config(sc, e, counts):
    return cm.EpochConfig(
        epoch_id=e, eligible_workers=tuple(range(sc.n)), base_state_id=f"s{e}",
        op_id=OP_ID, schema_id=SCHEMA_ID, k=sc.quorum, alpha=sc.alpha,
        deadline=sc.deadline, record_counts=dict(enumerate(counts)),
        merge_profile_id=MERGE_PROFILE)


def run_scenario(sc, reduce=None):
    """Drive `sc.n_epochs` epochs over one fixed population. Returns (epochs, summary)."""
    pop = make_population(n_workers=sc.n, seed=sc.pop_seed, dispersion=sc.dispersion,
                          correlated=sc.correlated)
    parts = worker_partials(pop, reduce)
    counts = tuple(w.record_count for w in pop.workers)
    scale = float(np.std(pop.all_values()))           # the population SD
    ref_mean = pop.exact_mean()
    rng = np.random.default_rng(sc.seed + 7919)

    epochs = []
    for e in range(sc.n_epochs):
        present = rng.random(sc.n) >= sc.dropout
        times = arrival_times(pop, rng, tail_sigma=sc.tail_sigma, correlated=sc.correlated)
        if sc.time_scale != 1.0:
            times = {w: t * sc.time_scale for w, t in times.items()}
        epoch = cm.CommitMergeEpoch(_config(sc, e, counts))

        for t, w in sorted((times[w], w) for w in range(sc.n) if present[w]):
            epoch.close_if_due(t)
            s, c = parts[w]
            # The three injection draws happen whatever the rates are, so a cell's
            # arrival stream does not depend on which faults it injects.
            if rng.random() < sc.malformed_rate:
                c = 0
            epoch.receive(cm.Contribution(e, w, 0, f"s{e}", partial_sum=s, partial_count=c), t)
            if rng.random() < sc.duplicate_rate:
                epoch.receive(cm.Contribution(e, w, 0, f"s{e}", partial_sum=s,
                                              partial_count=c), t + 1e-6)
            if rng.random() < sc.revision_rate:
                epoch.receive(cm.Contribution(e, w, 1, f"s{e}", partial_sum=s * 1.01,
                                              partial_count=c), t + 2e-6)

        epoch.close_if_due(sc.deadline)
        manifest = epoch.freeze()
        ap_completed = bool(present.all())            # listing 3.5 on the same draws
        row = {"epoch_id": e, "close_time": epoch.close_time,
               "committed": manifest is not None, "accepted": 0, "mean": None,
               "records": 0, "dev_norm": None, "ap_completed": ap_completed,
               "ap_close": float(max(times.values())) if ap_completed else None}
        if manifest is not None:
            merged_sum, merged_count = cm.canonical_merge(
                [(m.worker_id, m.partial_sum, m.partial_count) for m in manifest],
                MERGE_PROFILE)
            row.update(accepted=len(manifest), mean=float(merged_sum / merged_count),
                       records=merged_count)
            row["dev_norm"] = abs(row["mean"] - ref_mean) / scale
        epochs.append(row)

    done = [r for r in epochs if r["committed"]]
    closes = np.array([r["close_time"] for r in epochs])
    signed = [r["mean"] - ref_mean for r in done]
    summary = {
        "reference_mean": ref_mean, "scale": scale,
        "commit_rate": len(done) / len(epochs),
        "operands": float(np.mean([r["accepted"] for r in done])) if done else 0.0,
        "record_coverage": (float(np.mean([r["records"] / pop.total_records for r in done]))
                            if done else 0.0),
        "mean_deviation_norm": (float(np.mean([abs(x) for x in signed])) / scale
                                if done else 0.0),
        "signed_errors": signed,
        "close_p50": float(np.percentile(closes, 50)),
        "close_p99": float(np.percentile(closes, 99)),
    }
    return epochs, summary


def repeat(sc, reps=20, reduce=None):
    """One cell over `reps` arrival seeds; the dataset stays the same."""
    out = [run_scenario(Scenario(**{**asdict(sc), "seed": sc.seed + r}), reduce)[1]
           for r in range(reps)]
    mean = {k: float(np.mean([o[k] for o in out]))
            for k in ("commit_rate", "operands", "record_coverage",
                      "mean_deviation_norm", "close_p50", "close_p99")}
    # empirical bias: |average signed error| / SD, averaged over the seeds
    mean["mean_bias_norm"] = float(np.mean(
        [abs(np.mean(o["signed_errors"])) / o["scale"] for o in out if o["signed_errors"]]))
    return mean


def pooled(sc, reps=20, reduce=None):
    """One cell over `reps` arrival seeds, every statistic pooled over all epochs.

    Percentiles are taken over the pooled epochs rather than averaged across
    seeds, and both systems are read off the same draws.
    """
    rows = []
    for r in range(reps):
        rows += run_scenario(Scenario(**{**asdict(sc), "seed": sc.seed + r}), reduce)[0]
    done = [e for e in rows if e["committed"]]
    means = [e["mean"] for e in done]
    closes = [e["close_time"] for e in rows]
    ap_closes = [e["ap_close"] for e in rows if e["ap_completed"]]
    return {
        "epochs": len(rows),
        "commit_rate": len(done) / len(rows),
        "close_p50": float(np.percentile(closes, 50)),
        "close_p99": float(np.percentile(closes, 99)),
        "operands": float(np.mean([e["accepted"] for e in done])) if done else 0.0,
        "mean": float(np.mean(means)) if means else None,
        "mean_p05": float(np.percentile(means, 5)) if means else None,
        "mean_p95": float(np.percentile(means, 95)) if means else None,
        "mean_deviation_norm": float(np.mean([e["dev_norm"] for e in done])) if done else None,
        "ap_completion": len(ap_closes) / len(rows),
        "ap_close_p99": float(np.percentile(ap_closes, 99)) if ap_closes else None,
    }


# ------------------------------------------------------------- the results

DROPOUTS = (0.0, 0.05, 0.10, 0.20, 0.30)
SIGMAS = (0.1, 0.3, 0.5, 0.8, 1.2)
ALPHAS = (1.0, 1.25, 1.5, 2.0)
WINDOW_TAILS = (0.1, 0.5, 1.2)       # tight, moderate, severe


def dropout_sweep(reps=20, quorums=(16, 12, 10), alpha=1.5, reduce=None):
    """Table 3.6 and figure 3.4. Arrival tail 0.1, deadline 2.0, alpha 1.5.

    The all-participant column is read off the same epochs as the quorums.
    """
    rows = []
    for p in DROPOUTS:
        cells = {k: pooled(Scenario(k=k, alpha=alpha, dropout=p, tail_sigma=0.1,
                                    deadline=2.0), reps, reduce) for k in quorums}
        row = {"dropout": p, "all_participant": cells[quorums[0]]["ap_completion"],
               "cells": cells}
        row.update({k: c["commit_rate"] for k, c in cells.items()})
        rows.append(row)
    return rows


def window_sweep(reps=20, reduce=None):
    """Table 3.8 and figures 3.5, 3.7, 3.8: k = 12, no dropout, deadline 8.0."""
    cells = {(1.0, s, 1.5) for s in SIGMAS}
    cells |= {(1.0, s, a) for s in WINDOW_TAILS for a in ALPHAS}
    cells |= {(2.0, 0.1, 1.5), (2.0, 1.2, 1.5)}
    out = {}
    for scale, sigma, alpha in sorted(cells):
        out[(scale, sigma, alpha)] = pooled(Scenario(
            k=12, alpha=alpha, tail_sigma=sigma, dropout=0.0, deadline=8.0,
            time_scale=scale), reps, reduce)
    return out


def tail_sweep(window):
    """Figure 3.5: p99 close against arrival-tail severity, no dropout.

    These are the window sweep's alpha = 1.5 cells: k = 12, deadline 8.0, both
    systems on the same 1,200 epochs per point.
    """
    cells = [window[(1.0, s, 1.5)] for s in SIGMAS]
    return {"sigma": list(SIGMAS),
            "all_participant": [c["ap_close_p99"] for c in cells],
            "commit_and_merge": [c["close_p99"] for c in cells]}


def headline(drop, window):
    """Tables 3.4 and 3.5, read off the sweeps rather than simulated again.

        healthy overhead     the dropout sweep at 0 percent  (tail 0.1, deadline 2.0)
        dropout sweep        the dropout sweep at 10 percent
        latency-tail sweep   the tail sweep at sigma 1.2     (deadline 8.0)
    """
    at = {r["dropout"]: r["cells"][12] for r in drop}
    rows = []
    for name, c in (("healthy overhead", at[0.0]), ("dropout sweep", at[0.10]),
                    ("latency-tail sweep", window[(1.0, 1.2, 1.5)])):
        rows.append({
            "experiment": name,
            "ap_completion": c["ap_completion"], "ap_close_p99": c["ap_close_p99"],
            "cm_completion": c["commit_rate"], "cm_close_p99": c["close_p99"],
            "cm_operands": c["operands"], "cm_mean": c["mean"],
            "cm_mean_p05": c["mean_p05"], "cm_mean_p95": c["mean_p95"],
        })
    return rows


def replay_determinism(trials=200, seed=0):
    """Section 3.4.6: merge one frozen manifest under 200 shuffled arrival orders."""
    pop = make_population(seed=seed, dispersion=1.2)
    parts = [(w.worker_id, *pop.partial(w.worker_id)) for w in pop.workers]
    rng = np.random.default_rng(seed)
    canon, arr32, arr64 = set(), set(), set()
    for _ in range(trials):
        order = list(parts)
        rng.shuffle(order)
        s, c = cm.canonical_merge(order)
        canon.add(cm.partial_digest(s, c))
        acc32 = np.float32(0.0)
        acc64 = 0.0
        for _w, ps, _pc in order:
            acc32 = np.float32(acc32 + np.float32(ps))
            acc64 += ps
        arr32.add(cm.partial_digest(float(acc32), 1))
        arr64.add(cm.partial_digest(acc64, 1))
    return {"trials": trials, "canonical_f64": len(canon),
            "arrival_order_f64": len(arr64), "arrival_order_f32": len(arr32)}


def contribution_state_checks():
    """Table 3.10: inject each condition once and record the state it lands in."""
    pop = make_population(seed=0)
    counts = tuple(w.record_count for w in pop.workers)

    def fresh(k=16):
        return cm.CommitMergeEpoch(cm.EpochConfig(
            epoch_id=0, eligible_workers=tuple(range(16)), base_state_id="s0",
            op_id=OP_ID, schema_id=SCHEMA_ID, k=k, alpha=1.0, deadline=5.0,
            record_counts=dict(enumerate(counts)), merge_profile_id=MERGE_PROFILE))

    def c(w, rev=0, **kw):
        s, n = pop.partial(w)
        kw.setdefault("partial_sum", s)
        kw.setdefault("partial_count", n)
        return cm.Contribution(0, w, rev, "s0", **kw)

    def last(epoch, contribution, t):
        epoch.receive(contribution, t)
        return epoch.acks[-1]

    rows = []
    ep = fresh(); ep.receive(c(1), 0.1)
    rows.append(("Exact retransmission", last(ep, c(1), 0.2)))
    ep = fresh(); ep.receive(c(2), 0.1)
    rows.append(("Same revision, different payload", last(ep, c(2, partial_sum=99.0), 0.2)))
    ep = fresh(); ep.receive(c(3), 0.1)
    rows.append(("Higher valid revision", last(ep, c(3, rev=1), 0.2)))
    superseded = [a for a in ep.acks if a.status is cm.Status.SUPERSEDED]
    ep = fresh(); ep.receive(c(4, rev=2), 0.1)
    rows.append(("Stale revision", last(ep, c(4, rev=1), 0.2)))
    rows.append(("Ineligible worker", last(fresh(), cm.Contribution(
        0, 99, 0, "s0", partial_sum=1.0, partial_count=1), 0.1)))
    rows.append(("Non-integer count", last(fresh(), c(14, partial_count=1.5), 0.1)))
    rows.append(("Malformed count", last(fresh(), c(5, partial_count=0), 0.1)))
    rows.append(("Count disagrees with the declaration",
                 last(fresh(), c(9, partial_count=pop.partial(9)[1] + 1), 0.1)))
    rows.append(("Wrong epoch", last(fresh(), cm.Contribution(7, 10, 0, "s0",
                                                             *pop.partial(10)), 0.1)))
    rows.append(("Wrong base state", last(fresh(), cm.Contribution(0, 11, 0, "other",
                                                                  *pop.partial(11)), 0.1)))
    rows.append(("Schema mismatch", last(fresh(), c(12, schema_id="sumcount.v2"), 0.1)))
    rows.append(("Digest disagrees with payload", last(fresh(), c(13, claimed_digest="0" * 64), 0.1)))
    rows.append(("Non-finite sum", last(fresh(), c(6, partial_sum=float("nan")), 0.1)))
    ep = fresh(k=2); ep.receive(c(7), 0.1); ep.receive(c(8), 0.2)
    rows.append(("Post-close arrival", last(ep, c(9), 0.3)))

    out = [{"condition": name, "state": a.status.name, "reason": a.reason} for name, a in rows]
    out[2]["state"] += ", prior candidate " + (superseded[0].status.name if superseded else "?")
    return out


def availability(reps=20, k=12, reduce=None):
    """Table 3.11: independent against content-correlated availability.

    The collection factor is 1.0 here, so every committed manifest holds exactly
    the twelve-worker quorum; the tail is tight and the deadline 8.0.
    """
    rows = []
    for correlated in (False, True):
        m = repeat(Scenario(k=k, alpha=1.0, correlated=correlated, dropout=0.0,
                            deadline=8.0), reps, reduce)
        rows.append({"correlated": correlated, "record_coverage": m["record_coverage"],
                     "mean_deviation_norm": m["mean_deviation_norm"],
                     "mean_bias_norm": m["mean_bias_norm"]})
    return rows
