"""Chapter 6, section 6.4.4: the coordinator's view of a request.

    CompletionState                listing 6.17. The request, the tile count, the
                                   repair deadline, when each tile was last heard,
                                   the tiles sent for repair, and each tile's final
                                   output.
    heard(coord, tile, now)        listing 6.17. Record a report from a tile.
    check_for_missing(coord, now)  listing 6.17. Tiles silent for longer than the
                                   repair deadline, each reported once.
    accept_final(coord, tile, features)
                                   listing 6.17. Keep the first final output for each
                                   tile; True when every tile has one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompletionState:
    request_id: str
    tiles: int
    repair_deadline: float
    last_heard: dict = field(default_factory=dict)   # tile -> time
    reported: set = field(default_factory=set)       # tiles sent for repair
    final: dict = field(default_factory=dict)        # tile -> owned output


def heard(coord, tile, now):
    coord.last_heard[tile] = now


def check_for_missing(coord, now):
    silent = {t for t in range(coord.tiles) if t not in coord.final
              and now - coord.last_heard.setdefault(t, now) > coord.repair_deadline}
    new = silent - coord.reported
    coord.reported |= new
    return sorted(new)


def accept_final(coord, tile, features):
    if tile in coord.final:
        return False
    coord.final[tile] = features
    return len(coord.final) == coord.tiles
