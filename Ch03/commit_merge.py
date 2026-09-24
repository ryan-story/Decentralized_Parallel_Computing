"""Chapter 3, section 3.4: the commit-and-merge protocol.

One file, in the order the chapter builds it:

    listing 3.7    EpochConfig, _h, partial_digest       what an epoch publishes
    listing 3.8    Contribution, Status, CloseReason,    the records the protocol
                   Ack, ManifestEntry                    processes
    listing 3.9    CommitMergeEpoch                      one epoch's live state
    listing 3.10   CommitMergeEpoch._admit               the admission predicate
    listing 3.11   next_close_time, close_if_due,        arrivals, quorum, the
                   receive                               collection window, deadline
    listing 3.12   CommitMergeEpoch.freeze               the frozen manifest
    listing 3.13   canonical_merge                       the deterministic merge

The epoch holds no numerical result. It decides which operands belong to the
reduction and freezes them; `canonical_merge` does the arithmetic afterwards.

Deviation from the book's listings, and why:

  * Listings 3.10 to 3.12 each show `class CommitMergeEpoch:  # continued from
    listing 3.9` so one rule can be read at a time. Here those methods are
    written inside the one class, which is what the book means by "continued".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math

import numpy as np


# ---------------------------------------------------------------- listing 3.7

@dataclass(frozen=True)
class EpochConfig:
    epoch_id: int
    eligible_workers: tuple[int, ...]
    base_state_id: str
    op_id: str
    schema_id: str
    k: int
    alpha: float
    deadline: float
    record_counts: dict[int, int]
    merge_profile_id: str


def _h(*parts) -> str:
    encoded = json.dumps(
        parts,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def partial_digest(partial_sum, partial_count) -> str:
    return _h(
        float(partial_sum),
        int(partial_count),
    )


# ---------------------------------------------------------------- listing 3.8

@dataclass
class Contribution:
    epoch_id: int
    worker_id: int
    revision: int
    base_state_id: str
    partial_sum: float
    partial_count: int
    op_id: str = "mean"
    schema_id: str = "sumcount.v1"
    claimed_digest: str | None = None

    @property
    def digest(self) -> str:
        return partial_digest(
            self.partial_sum,
            self.partial_count,
        )

    @property
    def contribution_id(self) -> str:
        return _h(
            self.epoch_id,
            self.worker_id,
            self.revision,
            self.base_state_id,
            self.op_id,
            self.schema_id,
            self.digest,
        )


class Status(Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    LATE = "late"
    MISSING = "missing"


class CloseReason(Enum):
    FULL = "full"
    COLLECTION = "collection"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class Ack:
    worker_id: int
    contribution_id: str | None
    status: Status
    reason: str


@dataclass(frozen=True)
class ManifestEntry:
    worker_id: int
    revision: int
    contribution_id: str
    digest: str
    partial_sum: float
    partial_count: int


# ------------------------------------------------------ listings 3.9 to 3.12

class CommitMergeEpoch:

    # listing 3.9
    def __init__(self, cfg):
        self.cfg = cfg

        self.candidate = {}          # the current candidate for each worker
        self.acks = []               # every status event in this epoch

        self.closed = False
        self.close_reason = None
        self.close_time = None

        self.t_k = None              # the quorum time, once the k-th arrives

        self._seen = set()           # contribution ids already processed
        self._eligible = set(cfg.eligible_workers)
        self._rev_digest = {}        # (worker, revision) -> digest

    def _ack(self, c, status, reason):
        a = Ack(c.worker_id, c.contribution_id, status, reason)
        self.acks.append(a)
        return a

    def _mark_superseded(self, previous, by_revision):
        self.acks.append(
            Ack(
                previous.worker_id,
                previous.contribution_id,
                Status.SUPERSEDED,
                f"replaced_by_rev_{by_revision}",
            )
        )

    def _close(self, reason, t):
        self.closed = True
        self.close_reason = reason
        self.close_time = t

    # listing 3.10
    def _admit(self, c):
        if c.worker_id not in self._eligible:
            return False, "not_a_member"

        if c.epoch_id != self.cfg.epoch_id:
            return False, "wrong_epoch"

        if c.base_state_id != self.cfg.base_state_id:
            return False, "wrong_base_state"

        if c.op_id != self.cfg.op_id or c.schema_id != self.cfg.schema_id:
            return False, "schema_mismatch"

        if not isinstance(c.partial_count, int) or isinstance(c.partial_count, bool):
            return False, "count_not_an_integer"

        if c.partial_count <= 0:
            return False, "count_not_positive"

        expected_count = self.cfg.record_counts[c.worker_id]
        if c.partial_count != expected_count:
            return False, "count_mismatch"

        if not math.isfinite(float(c.partial_sum)):
            return False, "sum_not_finite"

        if c.claimed_digest is not None and c.claimed_digest != c.digest:
            return False, "integrity_failed"

        prev_digest = self._rev_digest.get((c.worker_id, c.revision))

        if prev_digest is not None and prev_digest != c.digest:
            return False, "revision_payload_conflict"

        prev = self.candidate.get(c.worker_id)

        if prev is not None and c.revision <= prev.revision:
            return False, "stale_revision"

        return True, "ok"

    # listing 3.11
    def next_close_time(self):

        if self.t_k is None:
            return self.cfg.deadline

        return min(
            self.cfg.alpha * self.t_k,
            self.cfg.deadline,
        )

    def close_if_due(self, now):

        if self.closed:
            return

        close_at = self.next_close_time()

        if now < close_at:
            return

        if self.t_k is None:
            reason = CloseReason.DEADLINE
        elif self.cfg.deadline <= self.cfg.alpha * self.t_k:
            reason = CloseReason.DEADLINE
        else:
            reason = CloseReason.COLLECTION

        self._close(reason, close_at)

    def receive(self, c, arrival_time):

        self.close_if_due(arrival_time)

        if self.closed:
            self._ack(c, Status.LATE, "after_close")
            return

        if c.contribution_id in self._seen:
            self._ack(c, Status.DUPLICATE, "already_processed")
            return

        self._seen.add(c.contribution_id)

        admitted, reason = self._admit(c)

        if not admitted:
            self._ack(c, Status.REJECTED, reason)
            return

        previous = self.candidate.get(c.worker_id)

        if previous is not None:
            self._mark_superseded(previous, c.revision)

        self.candidate[c.worker_id] = c
        self._rev_digest[(c.worker_id, c.revision)] = c.digest
        self._ack(c, Status.ACCEPTED, "ok")

        if self.t_k is None and len(self.candidate) >= self.cfg.k:
            self.t_k = arrival_time

        if len(self.candidate) == len(self._eligible):
            self._close(CloseReason.FULL, arrival_time)

    # listing 3.12
    def freeze(self):

        if not self.closed:
            raise RuntimeError("cannot freeze an open epoch")

        heard_from = {a.worker_id for a in self.acks}

        for wid in sorted(
            self._eligible - set(self.candidate) - heard_from
        ):
            self.acks.append(
                Ack(wid, None, Status.MISSING, "no_contribution_received")
            )

        manifest = tuple(
            ManifestEntry(
                worker_id=wid,
                revision=c.revision,
                contribution_id=c.contribution_id,
                digest=c.digest,
                partial_sum=float(c.partial_sum),
                partial_count=int(c.partial_count),
            )
            for wid, c in sorted(self.candidate.items())
        )

        if len(manifest) < self.cfg.k:
            return None

        return manifest


# --------------------------------------------------------------- listing 3.13

def canonical_merge(items, merge_profile_id="sorted-by-worker-f64-intcount"):
    if merge_profile_id != "sorted-by-worker-f64-intcount":
        raise ValueError(
            f"unknown merge profile {merge_profile_id!r}"
        )

    if not items:
        return None, 0

    ordered = sorted(items, key=lambda t: t[0])

    acc_sum = np.float64(0.0)
    acc_count = 0

    for _wid, s, c in ordered:
        acc_sum += np.float64(s)
        acc_count += int(c)

    return acc_sum, acc_count
