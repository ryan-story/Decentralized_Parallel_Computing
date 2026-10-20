"""Chapter 5, section 5.4.2: advancing the prefix and repairing gaps.

    advance(state, now)          listing 5.8. Move the frontier through every
                                 contiguous accepted summary, give each block its
                                 carry, checkpoint every `checkpoint_interval`
                                 blocks, and return the blocks finalized.
    receive(state, commit, now)  listing 5.8. Classify an arriving commit as
                                 rejected, duplicate, stale, deferred or accepted,
                                 and advance on acceptance.
    check_for_gap(state, now)    listing 5.9. Report the next required block once
                                 it has been missing for the repair deadline, once
                                 per gap.

A block's carry is the prefix state before its own summary is added: equation 5.3
computed incrementally. One repaired summary finalizes every staged block behind it.
"""

from __future__ import annotations

import math


def advance(state, now):
    finalized = []
    while state.frontier + 1 in state.summaries:
        s = state.frontier + 1
        state.carries[s] = state.prefix_state
        state.prefix_state += state.summaries[s].summary
        state.frontier = s
        state.blocked_since = now
        finalized.append(s)
        if (s + 1) % state.checkpoint_interval == 0:
            state.checkpoint = (s, state.prefix_state)
            state.summary_log = [c for c in state.summary_log
                                 if c.block_seq > s]
    return finalized

def receive(state, commit, now):
    s = commit.block_seq
    if (commit.run_id != state.run_id or not 0 <= s < state.n_blocks
            or not math.isfinite(commit.summary)):
        return "rejected", []
    if s <= state.frontier or s in state.summaries:
        same = state.summaries.get(s) == commit
        return ("duplicate" if same else "stale"), []
    if s > state.frontier + state.lookahead:
        return "deferred", []
    state.summaries[s] = commit
    state.summary_log.append(commit)
    return "accepted", advance(state, now)


def check_for_gap(state, now):
    s = state.frontier + 1
    if (s < state.n_blocks and s not in state.summaries
            and s != state.repairing
            and now - state.blocked_since >= state.repair_deadline):
        state.repairing = s
        return s
    return None
