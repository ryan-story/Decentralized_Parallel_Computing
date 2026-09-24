"""Chapter 3, section 3.2: the working example's workers and their data.

Plumbing for the experiments rather than a listing. It builds the fixed
population the chapter uses throughout, sixteen workers holding unequal batches,
and draws worker completion times.

    make_population(seed=0)    16 workers, 369 to 2,387 records each, 19,965 in
                               total, true mean 11.9818. The data is generated
                               here, deterministically, so nothing is downloaded.

    arrival_times(pop, rng, tail_sigma=...)
                               one completion time per worker, lognormal with a
                               log-mean of zero, so the median worker finishes at
                               1.0 for every sigma. Sigma is the arrival tail of
                               table 3.3: 0.1 tight, 0.5 moderate, 1.2 severe.

`correlated=True` in both makes the slow half of the workers also hold
systematically higher values, which is the content-correlated availability of
table 3.11. The population is otherwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GLOBAL_MEAN = 12.0          # centred well away from zero, so relative error is stable
RECORD_NOISE = 3.0
WORKER_SPREAD = 0.8


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    record_count: int
    group: int                  # 0 fast half, 1 slow half
    mean_shift: float           # this worker's offset from the global mean


@dataclass
class Population:
    workers: tuple
    records: dict = field(repr=False)

    def partial(self, worker_id):
        """One worker's exact (sum, count), in float64 and a Python int."""
        r = self.records[worker_id]
        return float(np.sum(r, dtype=np.float64)), int(r.size)

    @property
    def total_records(self):
        return int(sum(w.record_count for w in self.workers))

    def exact_mean(self):
        return sum(self.partial(w.worker_id)[0] for w in self.workers) / self.total_records

    def all_values(self):
        return np.concatenate([self.records[w.worker_id] for w in self.workers])


def make_population(n_workers=16, seed=0, *, dispersion=1.0, correlated=False,
                    base_records=1000):
    """Build one fixed population. `dispersion` sets how unequal the batches are."""
    rng = np.random.default_rng(seed)
    group = (np.arange(n_workers) >= n_workers // 2).astype(int)

    if dispersion <= 0:
        counts = np.full(n_workers, base_records, dtype=int)
    else:
        factors = np.exp(rng.uniform(-dispersion, dispersion, size=n_workers))
        counts = np.maximum(50, (base_records * factors).astype(int))

    shifts = rng.normal(0.0, WORKER_SPREAD, size=n_workers)
    if correlated:
        shifts = shifts + group * 2.5 * WORKER_SPREAD

    specs, records = [], {}
    for w in range(n_workers):
        specs.append(WorkerSpec(worker_id=w, record_count=int(counts[w]),
                                group=int(group[w]), mean_shift=float(shifts[w])))
        records[w] = rng.normal(GLOBAL_MEAN + shifts[w], RECORD_NOISE,
                                size=int(counts[w])).astype(np.float64)

    return Population(workers=tuple(specs), records=records)


def arrival_times(pop, rng, *, tail_sigma=0.5, correlated=False, slow_factor=2.5):
    """Completion time per worker, lognormal(0, tail_sigma).

    Under `correlated`, the slow half really is slower, by `slow_factor`.
    """
    t = rng.lognormal(mean=0.0, sigma=tail_sigma, size=len(pop.workers))
    return {w.worker_id: float(t[i] * (slow_factor if (correlated and w.group == 1) else 1.0))
            for i, w in enumerate(pop.workers)}
