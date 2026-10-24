"""
Draftsman helper utilities and high-level macros for FactorioLLM.
Provides algorithmic building blocks (belt lines, entity rows, underground pairs, power grids)
that drastically reduce code verbosity and eliminate token-limit truncations.

Coordinate convention: every `start`/`end` is an integer TILE position (top-left corner of the
entity footprint), passed to draftsman as `tile_position`. 1x1 and multi-tile entities therefore
share one grid: a 2x2 furnace at (0, 0) occupies tiles (0..1, 0..1), the next one goes to (2, 0).

This module is injected into the Docker sandbox as `draftsman_helpers`, so it must only depend
on draftsman and the standard library.
"""

from __future__ import annotations

import difflib
from typing import Union, Tuple, Dict, Any, Optional

from draftsman.constants import Direction
from draftsman.data import entities as _entity_data
from draftsman.entity import new_entity


_DIRECTION_ALIASES = {
    "north": 0, "up": 0, "n": 0,
    "east": 4, "right": 4, "e": 4,
    "south": 8, "down": 8, "s": 8,
    "west": 12, "left": 12, "w": 12,
}


def resolve_direction(dir_val: Union[int, str, Direction, None]) -> int:
    """Normalizes string or enum directions into Draftsman integer constants (0, 4, 8, 12)."""
    if dir_val is None:
        return 0
    if isinstance(dir_val, Direction):
        return int(dir_val)
    if isinstance(dir_val, int):
        if dir_val % 4 != 0 or not 0 <= dir_val < 16:
            raise ValueError(f"Invalid direction {dir_val}: use 'north', 'east', 'south' or 'west' (0, 4, 8, 12).")
        return dir_val
    if isinstance(dir_val, str):
        s = dir_val.strip().lower()
        if s.isdigit():
            return resolve_direction(int(s))
        if s in _DIRECTION_ALIASES:
            return _DIRECTION_ALIASES[s]
    raise ValueError(f"Invalid direction {dir_val!r}: use 'north', 'east', 'south' or 'west'.")


def _get_direction_offsets(direction: int) -> Tuple[int, int]:
    """Returns (dx, dy) unit tile delta for a given direction."""
    return {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}[direction]


def _resolve_tile(pos: Union[Tuple[int, int], Dict[str, int]]) -> Tuple[int, int]:
    """Normalizes a tile position to an (x, y) int tuple."""
    if isinstance(pos, dict):
        x, y = pos.get("x", 0), pos.get("y", 0)
    else:
        x, y = pos[0], pos[1]
    if float(x) != int(x) or float(y) != int(y):
        raise ValueError(f"Tile positions must be integers, got ({x}, {y}).")
    return int(x), int(y)


def _check_entity_name(name: str) -> None:
    if name not in _entity_data.raw:
        close = difflib.get_close_matches(name, list(_entity_data.raw), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"Unknown entity name '{name}'.{hint}")


def _instantiate_entity(name_or_class: Any, tile: Tuple[int, int], direction: Optional[int] = None, **kwargs) -> Any:
    """Instantiates an entity by prototype name (or draftsman class) at a tile position."""
    call_kwargs = dict(tile_position=tile, **kwargs)
    if direction:
        call_kwargs["direction"] = direction

    if isinstance(name_or_class, type):
        return name_or_class(**call_kwargs)

    name = str(name_or_class).strip()
    _check_entity_name(name)
    return new_entity(name, **call_kwargs)


def add_belt_line(
    bp: Any,
    start: Union[Tuple[int, int], Dict[str, int]],
    length: int,
    direction: Union[int, str, Direction] = "east",
    belt_type: str = "transport-belt",
) -> None:
    """
    Appends a straight line of conveyor belts to the blueprint.
    start: tile of the first belt; the line extends in `direction`.
    length: number of belt tiles to place.
    direction: direction the belts face and flow ('east', 'west', 'north', 'south').
    belt_type: 'transport-belt', 'fast-transport-belt', or 'express-transport-belt'.
    """
    dir_int = resolve_direction(direction)
    dx, dy = _get_direction_offsets(dir_int)
    sx, sy = _resolve_tile(start)

    for i in range(int(length)):
        ent = _instantiate_entity(belt_type, (sx + i * dx, sy + i * dy), direction=dir_int)
        bp.entities.append(ent)


def add_underground_pair(
    bp: Any,
    start: Union[Tuple[int, int], Dict[str, int]],
    end: Union[Tuple[int, int], Dict[str, int]],
    direction: Union[int, str, Direction] = "east",
    belt_type: str = "underground-belt",
) -> None:
    """
    Appends an underground belt pair: the entrance at `start` and the exit at `end` (tiles),
    both facing the flow `direction`.
    """
    dir_int = resolve_direction(direction)
    input_ent = _instantiate_entity(belt_type, _resolve_tile(start), direction=dir_int, io_type="input")
    output_ent = _instantiate_entity(belt_type, _resolve_tile(end), direction=dir_int, io_type="output")
    bp.entities.append(input_ent)
    bp.entities.append(output_ent)


def add_entity_row(
    bp: Any,
    entity: Any,
    start: Union[Tuple[int, int], Dict[str, int]],
    count: int,
    step: Tuple[int, int] = (1, 0),
    direction: Union[int, str, Direction, None] = None,
    **kwargs,
) -> None:
    """
    Appends an evenly spaced row or column of entities (furnaces, assemblers, inserters, poles).
    start: tile of the first entity's top-left corner.
    count: total number of entities to place.
    step: (dx, dy) tile offset between consecutive entities, e.g. (2, 0) for touching 2x2 furnaces.
    Extra keyword arguments (recipe=..., etc.) are passed to every entity.
    """
    dir_int = resolve_direction(direction) if direction is not None else None
    sx, sy = _resolve_tile(start)
    dx, dy = _resolve_tile(step)

    for i in range(int(count)):
        ent = _instantiate_entity(entity, (sx + i * dx, sy + i * dy), direction=dir_int, **kwargs)
        bp.entities.append(ent)


def add_power_poles(
    bp: Any,
    start: Union[Tuple[int, int], Dict[str, int]],
    count: int,
    step: Tuple[int, int] = (7, 0),
    pole_type: str = "small-electric-pole",
) -> None:
    """
    Appends a line of evenly spaced electric poles for powering production arrays.
    """
    add_entity_row(bp, pole_type, start=start, count=count, step=step)
