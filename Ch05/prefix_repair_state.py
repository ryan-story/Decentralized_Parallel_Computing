"""Chapter 5, section 5.4.1: the commit and the coordinator state.

    SummaryCommit       what a worker sends: the run, the logical slot
                        (`block_seq`), the attempt, and the block summary.
    PrefixRepairState   the coordinator: five settings fixed before the run, then
                        the frontier, the prefix state through it, the gap clock,
                        the slot under repair, the accepted summaries and carries,
                        the last checkpoint, and the summary log.

Listing 5.7. A commit is identified by its logical slot, not by the worker that
computed it, so a replacement worker fills the same slot as the worker it replaces.
`checkpoint` and `summary_log` are held in memory here; a deployment writes both to
durable storage before acting on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SummaryCommit:
    run_id: str
    block_seq: int
    attempt_id: str
    summary: float


@dataclass
class PrefixRepairState:
    run_id: str
    n_blocks: int
    lookahead: int
    repair_deadline: float
    checkpoint_interval: int
    frontier: int = -1
    prefix_state: float = 0.0
    blocked_since: float = 0.0
    repairing: int = -1
    summaries: dict = field(default_factory=dict)
    carries: dict = field(default_factory=dict)
    checkpoint: tuple = (-1, 0.0)
    summary_log: list = field(default_factory=list)
