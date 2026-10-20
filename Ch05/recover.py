"""Chapter 5, section 5.4.4: recovering the coordinator after a restart.

    recover(crashed, now)    listing 5.12. Build a new coordinator from the run's
                             settings, the last checkpoint and the summary log, and
                             re-finalize what the log makes contiguous.

The checkpoint and log are held in memory, so this models what a restart rebuilds;
it does not survive a real process crash. It rebuilds only the coordinator's state.
The local scans stay with the workers that computed them, and every re-finalized
block receives the carry it received before the restart.
"""

from __future__ import annotations

from cpr_coordinator import advance
from prefix_repair_state import PrefixRepairState


def recover(crashed, now):
    state = PrefixRepairState(crashed.run_id, crashed.n_blocks,
                              crashed.lookahead, crashed.repair_deadline,
                              crashed.checkpoint_interval,
                              blocked_since=now)
    state.frontier, state.prefix_state = crashed.checkpoint
    state.checkpoint = crashed.checkpoint
    for commit in crashed.summary_log:
        state.summaries[commit.block_seq] = commit
        state.summary_log.append(commit)
    return state, advance(state, now)


if __name__ == "__main__":
    from cpr_coordinator import receive
    from prefix_repair_state import SummaryCommit

    state = PrefixRepairState("demo", 8, lookahead=8, repair_deadline=1.0,
                              checkpoint_interval=2)
    for s in (0, 1, 2, 4):
        receive(state, SummaryCommit("demo", s, f"{s}.0", float(s + 1)), now=0.0)
    print("before the restart: frontier", state.frontier, "carries", state.carries)
    print("checkpoint", state.checkpoint, "log", [c.block_seq for c in state.summary_log])

    back, refinalized = recover(state, now=1.0)
    print("after recover    : frontier", back.frontier, "refinalized", refinalized)
    print("same carries     :", all(back.carries[b] == state.carries[b] for b in refinalized))
    print("block 4 still pending:", 4 in back.summaries and back.frontier < 4)
