"""
Draftsman helper utilities and high-level macros for FactorioLLM.
Provides algorithmic building blocks (belt lines, entity rows, underground pairs, power grids)
that drastically reduce code verbosity and eliminate token-limit truncations.
"""

from __future__ import annotations

from typing import Union, Tuple, Dict, Any, Optional
import draftsman.entity as _d_ent
from draftsman.constants import Direction


def resolve_direction(dir_val: Union[int, str, Direction, None]) -> int:
    """Normalizes string or enum directions into Draftsman integer constants (0, 4, 8, 12)."""
    if dir_val is None:
        return 0
    if isinstance(dir_val, int):
        return dir_val
    if isinstance(dir_val, Direction):
        return int(dir_val)
    if isinstance(dir_val, str):
        s = dir_val.strip().lower()
        mapping = {
            "north": 0, "up": 0, "n": 0, "0": 0,
            "east": 4, "right": 4, "e": 4, "4": 4,
            "south": 8, "down": 8, "s": 8, "8": 8,
            "west": 12, "left": 12, "w": 12, "12": 12,
        }
        return mapping.get(s, 0)
    return 0


def _get_direction_offsets(direction: int) -> Tuple[float, float]:
    """Returns (dx, dy) unit delta for a given direction."""
    if direction == 0:  # North
        return (0.0, -1.0)
    elif direction == 4:  # East
        return (1.0, 0.0)
    elif direction == 8:  # South
        return (0.0, 1.0)
    elif direction == 12:  # West
        return (-1.0, 0.0)
    return (1.0, 0.0)


def _resolve_pos(pos: Union[Tuple[float, float], Dict[str, float]]) -> Tuple[float, float]:
    """Normalizes position to (x, y) float tuple."""
    if isinstance(pos, dict):
        return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
    return float(pos[0]), float(pos[1])


def _instantiate_entity(name_or_class: Any, position: Tuple[float, float], direction: Optional[int] = None, **kwargs) -> Any:
    """Instantiates an entity prototype by string name or Draftsman class."""
    pos_dict = {"x": position[0], "y": position[1]}

    # Check if name is already an entity class
    if isinstance(name_or_class, type):
        ctor = name_or_class
        call_kwargs = dict(position=pos_dict, **kwargs)
        if direction is not None:
            call_kwargs["direction"] = direction
        return ctor(**call_kwargs)

    name_str = str(name_or_class).strip()

    # Try resolving via draftsman.entity
    cls = None
    if hasattr(_d_ent, "get_entity_class"):
        try:
            cls = _d_ent.get_entity_class(name_str)
        except Exception:
            pass

    if cls is None and hasattr(_d_ent, name_str):
        cls = getattr(_d_ent, name_str)

    if cls is None:
        # Fallback to general Entity
        from draftsman.classes.entity import Entity
        cls = Entity

    call_kwargs = dict(name=name_str, position=pos_dict, **kwargs)
    if direction is not None:
        call_kwargs["direction"] = direction

    try:
        return cls(**call_kwargs)
    except TypeError:
        # Some entity constructors don't take name if it's specialized
        call_kwargs.pop("name", None)
        return cls(**call_kwargs)


def add_belt_line(
    bp: Any,
    start: Union[Tuple[float, float], Dict[str, float]],
    length: int,
    direction: Union[int, str, Direction] = "east",
    belt_type: str = "transport-belt",
) -> None:
    """
    Appends a straight line of conveyor belts to the blueprint.
    start: starting (x, y) coordinate.
    length: number of belt tiles to place.
    direction: direction the belts face and flow ('east', 'west', 'north', 'south' or 0, 4, 8, 12).
    belt_type: 'transport-belt', 'fast-transport-belt', or 'express-transport-belt'.
    """
    dir_int = resolve_direction(direction)
    dx, dy = _get_direction_offsets(dir_int)
    sx, sy = _resolve_pos(start)

    for i in range(int(length)):
        x = sx + i * dx
        y = sy + i * dy
        ent = _instantiate_entity(belt_type, (x, y), direction=dir_int)
        bp.entities.append(ent)


def add_underground_pair(
    bp: Any,
    start: Union[Tuple[float, float], Dict[str, float]],
    end: Union[Tuple[float, float], Dict[str, float]],
    direction: Union[int, str, Direction] = "east",
    belt_type: str = "underground-belt",
) -> None:
    """
    Appends an underground belt pair (input and output) to the blueprint.
    """
    dir_int = resolve_direction(direction)
    sx, sy = _resolve_pos(start)
    ex, ey = _resolve_pos(end)

    input_ent = _instantiate_entity(belt_type, (sx, sy), direction=dir_int, io_type="input")
    output_ent = _instantiate_entity(belt_type, (ex, ey), direction=dir_int, io_type="output")
    bp.entities.append(input_ent)
    bp.entities.append(output_ent)


def add_entity_row(
    bp: Any,
    entity: Any,
    start: Union[Tuple[float, float], Dict[str, float]],
    count: int,
    step: Tuple[float, float] = (1.0, 0.0),
    direction: Union[int, str, Direction, None] = None,
    **kwargs,
) -> None:
    """
    Appends an evenly spaced row or column of entities (furnaces, assemblers, lamps, chests).
    start: starting (x, y) coordinate of the first entity.
    count: total number of entities to place.
    step: (dx, dy) spacing between consecutive entity centers (e.g. (3.0, 0.0) for 2x2 furnaces with gap).
    """
    dir_int = resolve_direction(direction) if direction is not None else None
    sx, sy = _resolve_pos(start)
    dx, dy = float(step[0]), float(step[1])

    for i in range(int(count)):
        x = sx + i * dx
        y = sy + i * dy
        ent = _instantiate_entity(entity, (x, y), direction=dir_int, **kwargs)
        bp.entities.append(ent)


def add_power_poles(
    bp: Any,
    start: Union[Tuple[float, float], Dict[str, float]],
    count: int,
    step: Tuple[float, float] = (7.0, 0.0),
    pole_type: str = "small-electric-pole",
) -> None:
    """
    Appends a line of evenly spaced electric poles for powering production arrays.
    """
    add_entity_row(bp, pole_type, start=start, count=count, step=step)
