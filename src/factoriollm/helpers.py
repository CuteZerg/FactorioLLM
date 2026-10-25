"""
Draftsman helper utilities and high-level macros for FactorioLLM.
Provides algorithmic building blocks (belt lines, entity rows, underground pairs, power grids)
that drastically reduce code verbosity and eliminate token-limit truncations, plus parametric
modules (smelting_column, assembler_row, lab_array, mining_field, solar_field, steam_power) that
compute the whole tile layout themselves, so a model only chooses what and how many to build.

Coordinate convention: every `start`/`end`/`origin` is an integer TILE position (top-left corner
of the entity footprint), passed to draftsman as `tile_position`. 1x1 and multi-tile entities
therefore share one grid: a 2x2 furnace at (0, 0) occupies tiles (0..1, 0..1), the next one goes
to (2, 0). Every helper refuses to place an entity on an occupied tile and says which one.

This module is injected into the Docker sandbox as `draftsman_helpers`, so it must only depend
on draftsman and the standard library.
"""

from __future__ import annotations

import difflib
import math
from collections import deque
from functools import lru_cache
from typing import Union, Tuple, Dict, Any, Optional, List, NamedTuple

from draftsman.constants import Direction
from draftsman.data import entities as _entity_data
from draftsman.data import recipes as _recipe_data
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


def _entity_tiles(ent: Any) -> List[Tuple[int, int]]:
    x0, y0 = int(ent.tile_position.x), int(ent.tile_position.y)
    return [(x0 + dx, y0 + dy) for dx in range(ent.tile_width) for dy in range(ent.tile_height)]


class _PlacementGuard:
    """Appends entities to a blueprint and refuses any that overlap an already placed one."""

    def __init__(self, bp: Any):
        self.bp = bp
        self.occupied: Dict[Tuple[int, int], Any] = {}
        for ent in bp.entities:
            for tile in _entity_tiles(ent):
                self.occupied[tile] = ent

    def check(self, ent: Any) -> None:
        for tile in _entity_tiles(ent):
            other = self.occupied.get(tile)
            if other is not None:
                pos = (int(ent.tile_position.x), int(ent.tile_position.y))
                other_pos = (int(other.tile_position.x), int(other.tile_position.y))
                raise ValueError(f"Cannot place {ent.name} at {pos}: tile {tile} is already occupied by "
                                 f"{other.name} at {other_pos}.")

    def append(self, ent: Any) -> None:
        self.check(ent)
        self.bp.entities.append(ent)
        for tile in _entity_tiles(ent):
            self.occupied[tile] = ent


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

    guard = _PlacementGuard(bp)
    for i in range(int(length)):
        guard.append(_instantiate_entity(belt_type, (sx + i * dx, sy + i * dy), direction=dir_int))


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
    guard = _PlacementGuard(bp)
    guard.append(input_ent)
    guard.append(output_ent)


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

    guard = _PlacementGuard(bp)
    for i in range(int(count)):
        guard.append(_instantiate_entity(entity, (sx + i * dx, sy + i * dy), direction=dir_int, **kwargs))


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


# --- Parametric modules -------------------------------------------------------------------
#
# A module is laid out in a local frame where items flow east (+u) and v grows southwards, then
# rotated to the requested flow direction and placed with its top-left tile at `origin`.
# Every module returns its BBox so the next one can be placed next to it.

class BBox(NamedTuple):
    """Tiles occupied by a module: top-left (x, y) and size; right/bottom are exclusive."""
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def place_right_of(bbox: BBox, gap: int = 1) -> Tuple[int, int]:
    """Origin tile for the next module to the right of `bbox`, `gap` empty tiles apart."""
    return bbox.right + int(gap), bbox.y


def place_below(bbox: BBox, gap: int = 1) -> Tuple[int, int]:
    """Origin tile for the next module below `bbox`, `gap` empty tiles apart."""
    return bbox.x, bbox.bottom + int(gap)


_FLOW_TURNS = {4: 0, 8: 1, 12: 2, 0: 3}  # quarter turns clockwise from the local east flow


@lru_cache(maxsize=None)
def _footprint(name: str, direction: Optional[int]) -> Tuple[int, int]:
    ent = new_entity(name, tile_position=(0, 0), **({"direction": direction} if direction else {}))
    return ent.tile_width, ent.tile_height


def _is_electric(name: str) -> bool:
    return (_entity_data.raw[name].get("energy_source") or {}).get("type") == "electric"


class _Layout:
    """Entities of one module in its local frame (flow east)."""

    def __init__(self):
        self.items: List[Tuple[str, int, int, Optional[int], Dict[str, Any]]] = []
        self.tiles: Dict[Tuple[int, int], int] = {}
        self.width = 0
        self.height = 0

    def add(self, name: str, u: int, v: int, direction: Optional[int] = None, **kwargs) -> int:
        w, h = _footprint(name, direction)
        idx = len(self.items)
        for du in range(w):
            for dv in range(h):
                tile = (u + du, v + dv)
                if tile in self.tiles:  # a module bug, never a model error
                    raise RuntimeError(f"module layout overlaps at local tile {tile}")
                self.tiles[tile] = idx
        self.items.append((name, u, v, direction, kwargs))
        self.width = max(self.width, u + w)
        self.height = max(self.height, v + h)
        return idx

    def rect(self, idx: int) -> Tuple[int, int, int, int]:
        name, u, v, direction, _ = self.items[idx]
        return (u, v) + _footprint(name, direction)

    def is_free(self, u: int, v: int, w: int = 1, h: int = 1) -> bool:
        return (u >= 0 and v >= 0 and u + w <= self.width and v + h <= self.height
                and all((u + du, v + dv) not in self.tiles for du in range(w) for dv in range(h)))

    def commit(self, bp: Any, origin: Union[Tuple[int, int], Dict[str, int]], direction: Any) -> BBox:
        """Rotates the layout so that items flow in `direction` and appends it at `origin`."""
        turns = _FLOW_TURNS[resolve_direction(direction)]
        ox, oy = _resolve_tile(origin)
        W, H = self.width, self.height
        guard = _PlacementGuard(bp)
        entities = []
        for name, u, v, d, kwargs in self.items:
            w, h = _footprint(name, d)
            if turns == 0:
                x, y = u, v
            elif turns == 1:
                x, y = H - v - h, u
            elif turns == 2:
                x, y = W - u - w, H - v - h
            else:
                x, y = v, W - u - w
            new_dir = (d + 4 * turns) % 16 if d is not None else None
            entities.append(_instantiate_entity(name, (ox + x, oy + y), direction=new_dir, **kwargs))
        for ent in entities:  # check everything first: a failed module leaves the blueprint untouched
            guard.check(ent)
        for ent in entities:
            guard.append(ent)
        size = (W, H) if turns % 2 == 0 else (H, W)
        return BBox(ox, oy, *size)


def _pole_geometry(pole: str) -> Tuple[int, int, float, float]:
    _check_entity_name(pole)
    raw = _entity_data.raw[pole]
    if raw.get("type") != "electric-pole":
        raise ValueError(f"'{pole}' is not an electric pole.")
    w, h = _footprint(pole, None)
    return w, h, float(raw.get("supply_area_distance", 0)), float(raw.get("maximum_wire_distance", 0))


def _add_poles(layout: _Layout, pole: str, powered: List[int]) -> None:
    """
    Places poles on free tiles of the layout so that every entity in `powered` lies in a supply
    area and all poles form one wire-connected network (greedy cover, then bridging).
    """
    if not powered:
        return
    pw, ph, supply, reach = _pole_geometry(pole)
    candidates = [(u, v) for u in range(layout.width) for v in range(layout.height) if layout.is_free(u, v, pw, ph)]

    def center(c):
        return c[0] + pw / 2, c[1] + ph / 2

    def covers(c, rect):
        cx, cy = center(c)
        u, v, w, h = rect
        return u < cx + supply and u + w > cx - supply and v < cy + supply and v + h > cy - supply

    def near(a, b):
        (ax, ay), (bx, by) = center(a), center(b)
        return math.hypot(ax - bx, ay - by) <= reach + 1e-9

    poles: List[Tuple[int, int]] = []

    def place(c):
        poles.append(c)
        layout.add(pole, *c)

    remaining = [layout.rect(i) for i in powered]
    while remaining:
        target = min(remaining, key=lambda r: (r[0], r[1]))
        options = [c for c in candidates if layout.is_free(c[0], c[1], pw, ph) and covers(c, target)]
        if not options:
            raise ValueError(f"No free tile to power {layout.items[powered[0]][0]} with {pole}; "
                             f"use a pole with a larger supply area.")
        best = max(options, key=lambda c: (not poles or any(near(c, p) for p in poles),
                                           sum(covers(c, r) for r in remaining), c[0], -c[1]))
        place(best)
        remaining = [r for r in remaining if not covers(best, r)]

    # bridge separate wire networks through free tiles (BFS over candidate positions)
    while True:
        comp = {poles[0]}
        frontier = [poles[0]]
        while frontier:
            p = frontier.pop()
            for q in poles:
                if q not in comp and near(p, q):
                    comp.add(q)
                    frontier.append(q)
        if len(comp) == len(poles):
            return
        targets = set(poles) - comp
        free = [c for c in candidates if layout.is_free(c[0], c[1], pw, ph)]
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {p: None for p in comp}
        queue = deque(comp)
        found = None
        while queue and found is None:
            p = queue.popleft()
            for q in list(targets) + free:
                if q not in prev and near(p, q):
                    prev[q] = p
                    if q in targets:
                        found = q
                        break
                    queue.append(q)
        if found is None:
            raise ValueError(f"Cannot connect the {pole} network inside the module; use a pole with a longer wire reach.")
        step = prev[found]
        while step is not None and step not in comp:
            place(step)
            step = prev[step]


def _belt_names(belt: str) -> Tuple[str, str, str]:
    """(belt, underground, splitter) names of one belt tier."""
    tiers = {
        "transport-belt": ("transport-belt", "underground-belt", "splitter"),
        "fast-transport-belt": ("fast-transport-belt", "fast-underground-belt", "fast-splitter"),
        "express-transport-belt": ("express-transport-belt", "express-underground-belt", "express-splitter"),
    }
    if belt not in tiers:
        raise ValueError(f"Unknown belt '{belt}': use one of {', '.join(tiers)}.")
    return tiers[belt]


def _check_count(name: str, value: Any, low: int = 1, high: int = 1000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}, got {value!r}.")
    return value


def _check_choice(name: str, value: Any, choices: Tuple[Any, ...]) -> Any:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")
    return value


def _check_type(name: str, allowed_types: Tuple[str, ...]) -> None:
    _check_entity_name(name)
    if _entity_data.raw[name].get("type") not in allowed_types:
        raise ValueError(f"'{name}' cannot be used here: expected a {' or '.join(allowed_types)}.")


def _check_recipe(machine: str, recipe: str) -> None:
    if recipe not in _recipe_data.raw:
        close = difflib.get_close_matches(recipe, list(_recipe_data.raw), n=3)
        raise ValueError(f"Unknown recipe '{recipe}'." + (f" Did you mean: {', '.join(close)}?" if close else ""))
    raw = _recipe_data.raw[recipe]
    if any(part.get("type") == "fluid" for part in (raw.get("ingredients") or []) + (raw.get("results") or [])):
        raise ValueError(f"Recipe '{recipe}' uses fluids; assembler_row only supports item recipes.")
    category = raw.get("category", "crafting")
    if category not in (_entity_data.raw[machine].get("crafting_categories") or []):
        raise ValueError(f"{machine} cannot craft '{recipe}' (category {category}).")


def _production_layout(building: str, count: int, rows: int, inputs: int, output: bool, belt: str,
                       inserter: str, pole: Optional[str], **building_kwargs) -> _Layout:
    """
    Rows of buildings fed by inserters from input belts and emptied onto output belts.
    rows=2 mirrors the row around shared input belts. Local frame, items flow east.
    """
    _check_type(inserter, ("inserter",))
    size = _footprint(building, None)[0]
    rows = min(rows, count)
    per_row = [count - count // rows, count // rows] if rows == 2 else [count]
    length = per_row[0] * size
    layout = _Layout()
    powered: List[int] = []

    def row(v_top: int, n: int, mirrored: bool) -> int:
        # top to bottom when not mirrored: input inserters, building, output inserters, output belt
        ins_dir = 8 if mirrored else 0  # an inserter faces the side it picks up from
        stack_h = 1 + size + (2 if output else 0)
        ins_v = v_top + stack_h - 1 if mirrored else v_top
        bld_v = ins_v - size if mirrored else ins_v + 1
        out_v = bld_v - 1 if mirrored else bld_v + size
        belt_v = out_v - 1 if mirrored else out_v + 1
        for i in range(n):
            u = i * size
            powered.append(layout.add(inserter, u, ins_v, ins_dir))
            if inputs == 2:
                powered.append(layout.add("long-handed-inserter", u + 1, ins_v, ins_dir))
            powered.append(layout.add(building, u, bld_v, **building_kwargs))
            if output:
                powered.append(layout.add(inserter, u + size - 1, out_v, ins_dir))
        if output:
            for u in range(length):
                layout.add(belt, u, belt_v, 4)
        return stack_h

    v = 0
    if rows == 2:
        v += row(v, per_row[1], mirrored=True)
    for k in range(inputs):
        for u in range(length):
            layout.add(belt, u, v + k, 4)
    v += inputs
    row(v, per_row[0], mirrored=False)

    if pole:
        _add_poles(layout, pole, [i for i in powered if _is_electric(layout.items[i][0])])
    return layout


def smelting_column(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 24,
    furnace: str = "stone-furnace",
    belt: str = "transport-belt",
    inserter: str = "inserter",
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    direction: Union[int, str, Direction] = "south",
) -> BBox:
    """
    Smelting array: `count` furnaces in `rows` (1 or 2) lines along the flow `direction`.
    Ore (and coal for stone/steel furnaces, on the other lane) comes on the middle belt, plates
    leave on the outer belt(s); inserters and poles are placed automatically.
    Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_type(furnace, ("furnace",))
    _belt_names(belt)
    layout = _production_layout(furnace, count, rows, 1, True, belt, inserter, pole)
    return layout.commit(bp, origin, direction)


def assembler_row(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 6,
    recipe: str = "iron-gear-wheel",
    machine: str = "assembling-machine-2",
    belt: str = "transport-belt",
    inserter: str = "inserter",
    inputs: int = 1,
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 1,
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Assembly line: `count` machines crafting `recipe` in `rows` (1 or 2) lines along `direction`.
    inputs=1: one input belt (two ingredient lanes); inputs=2: two input belts, the far one is
    reached by long-handed inserters. Products leave on the outer output belt(s).
    Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_choice("inputs", inputs, (1, 2))
    _check_type(machine, ("assembling-machine",))
    _check_recipe(machine, recipe)
    _belt_names(belt)
    layout = _production_layout(machine, count, rows, inputs, True, belt, inserter, pole, recipe=recipe)
    return layout.commit(bp, origin, direction)


def lab_array(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 10,
    belt: str = "transport-belt",
    inserter: str = "inserter",
    inputs: int = 1,
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Research block: `count` labs in `rows` (1 or 2) lines fed with science packs from the middle
    belt(s) (inputs=1: 2 pack types, inputs=2: 4 pack types). Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_choice("inputs", inputs, (1, 2))
    _belt_names(belt)
    layout = _production_layout("lab", count, rows, inputs, False, belt, inserter, pole)
    return layout.commit(bp, origin, direction)


def mining_field(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 16,
    belt: str = "transport-belt",
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Ore field: `count` electric mining drills on one or both sides (`rows`=1|2) of an output belt
    that flows in `direction`. Pole gaps are left between drill pairs. Returns the BBox.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _belt_names(belt)
    per_side = [count - count // rows, count // rows] if rows == 2 else [count]
    layout = _Layout()
    powered: List[int] = []
    belt_v = 3
    width = 0
    for side, n in enumerate(per_side):
        u = 0
        for i in range(n):
            # drills face the belt; a pole gap column follows drills 0, 2, 4, ...
            powered.append(layout.add("electric-mining-drill", u, 0 if side == 0 else belt_v + 1, 8 if side == 0 else 0))
            u += 4 if i % 2 == 0 else 3
        width = max(width, u)
    for u in range(width):
        layout.add(belt, u, belt_v, 4)
    if pole:
        _add_poles(layout, pole, powered)
    return layout.commit(bp, origin, direction)


_PANELS_PER_CELL = 32
_ACCUMULATORS_PER_CELL = 80


def solar_field(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    panels: int = 84,
    accumulators: Optional[int] = None,
) -> BBox:
    """
    Solar field of exactly `panels` solar panels and `accumulators` accumulators (default: the
    usual 0.84 per panel) in 18x18 substation cells: up to 32 panels or 80 accumulators per cell,
    neighbouring substations are wired together. Returns the BBox of the field.
    """
    _check_count("panels", panels, 0, 5000)
    if accumulators is None:
        accumulators = round(panels * 0.84)
    _check_count("accumulators", accumulators, 0, 5000)
    if panels + accumulators == 0:
        raise ValueError("solar_field needs at least one panel or accumulator.")
    cells: List[Tuple[str, int]] = []
    for name, total, per_cell in (("solar-panel", panels, _PANELS_PER_CELL),
                                  ("accumulator", accumulators, _ACCUMULATORS_PER_CELL)):
        while total > 0:
            cells.append((name, min(total, per_cell)))
            total -= per_cell
    cols = math.ceil(math.sqrt(len(cells)))
    layout = _Layout()
    for k, (name, n) in enumerate(cells):
        cu, cv = (k % cols) * 18, (k // cols) * 18
        layout.add("substation", cu + 8, cv + 8)
        size = 3 if name == "solar-panel" else 2
        placed = 0
        for j in range(18 // size):
            for i in range(18 // size):
                su, sv = cu + i * size, cv + j * size
                if placed < n and all((su + du, sv + dv) not in layout.tiles for du in range(size) for dv in range(size)):
                    layout.add(name, su, sv)
                    placed += 1
    return layout.commit(bp, origin, "east")


def steam_power(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    boilers: int = 10,
    engines_per_boiler: int = 2,
    belt: Optional[str] = "transport-belt",
    inserter: str = "burner-inserter",
    pole: Optional[str] = "small-electric-pole",
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Steam power plant: an offshore pump feeds a chain of `boilers` (water flows in `direction`),
    each boiler drives `engines_per_boiler` (1 or 2) steam engines; fuel comes on a belt behind
    the boilers (belt=None: no fuel belt and inserters). One pump feeds up to 20 boilers.
    The offshore pump must end up on a water edge. Returns the BBox of the plant.
    """
    _check_count("boilers", boilers, 1, 20)
    _check_choice("engines_per_boiler", engines_per_boiler, (1, 2))
    if belt is not None:
        _belt_names(belt)
        _check_type(inserter, ("inserter",))
    layout = _Layout()
    boiler_v = 5 * engines_per_boiler
    engines: List[int] = []
    u = 1
    for i in range(boilers):
        layout.add("boiler", u, boiler_v, 0)
        for k in range(engines_per_boiler):
            engines.append(layout.add("steam-engine", u, boiler_v - 5 * (k + 1), 0))
        if belt is not None:
            layout.add(inserter, u + 1, boiler_v + 2, 8)
        u += 3
        if i % 2 == 0:  # pole gap column after boilers 0, 2, 4, ...; water passes through a pipe
            if i < boilers - 1:
                layout.add("pipe", u, boiler_v + 1)
            u += 1
    layout.add("offshore-pump", 0, boiler_v + 1, 12)
    layout.width = max(layout.width, u)
    if belt is not None:
        for bu in range(1, u):
            layout.add(belt, bu, boiler_v + 3, 4)
    if pole:
        _add_poles(layout, pole, engines)
    return layout.commit(bp, origin, direction)


def balancer(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    inputs: int = 4,
    outputs: int = 4,
    belt: str = "transport-belt",
    direction: Union[int, str, Direction] = "north",
) -> BBox:
    """
    Belt balancer that splits every one of `inputs` belts evenly across `outputs` belts, built
    from a verified template. Items flow in `direction`: inputs enter on the opposite side,
    outputs leave on the `direction` side. Returns the BBox of the balancer.
    """
    names = _belt_names(belt)
    tier = ("transport-belt", "fast-transport-belt", "express-transport-belt").index(names[0]) + 1
    options = [(len(t), key) for key, t in _BALANCER_TEMPLATES.items()
               if key[:2] == (inputs, outputs) and key[2] <= tier]
    if not options:
        available = sorted({f"{i}->{o}" for i, o, _ in _BALANCER_TEMPLATES})
        raise ValueError(f"No {inputs}->{outputs} balancer template for {belt}; available: {', '.join(available)}.")
    template = _BALANCER_TEMPLATES[min(options)[1]]
    kind_names = {"belt": names[0], "underground": names[1], "splitter": names[2]}
    layout = _Layout()
    for kind, u, v, d, io in template:
        layout.add(kind_names[kind], u, v, d, **({"io_type": io} if io else {}))
    return layout.commit(bp, origin, direction)


# BEGIN BALANCER TEMPLATES (generated by tools/extract_balancers.py)
# (inputs, outputs, min belt tier 1..3): [(kind, u, v, direction, io_type)], items flow east
_BALANCER_TEMPLATES = {
    (1, 2, 1): [  # canonical_blueprints id=461 EARLY FACTBOOK | 1 to 2
        ('belt', 0, 0, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (1, 3, 1): [  # canonical_blueprints id=462 EARLY FACTBOOK | 1 to 3
        ('belt', 0, 3, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None),
        ('belt', 2, 0, 12, None), ('splitter', 2, 1, 0, None), ('splitter', 2, 2, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 2, 0, None),
        ('belt', 3, 3, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('belt', 5, 1, 4, None),
        ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None),
    ],
    (1, 4, 1): [  # canonical_blueprints id=26 Raynquist's Belt Balancers Fall 2025 | 1-4 TU balancer
        ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None), ('belt', 3, 0, 4, None),
        ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
    ],
    (1, 5, 1): [  # canonical_blueprints id=464 EARLY FACTBOOK | 1 to 5
        ('belt', 0, 4, 4, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 8, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None),
        ('splitter', 2, 0, 12, None), ('splitter', 2, 3, 4, None), ('belt', 3, 0, 12, None), ('belt', 3, 1, 12, None), ('splitter', 3, 2, 0, None),
        ('belt', 3, 3, 0, None), ('belt', 3, 4, 4, None), ('belt', 4, 0, 12, None), ('splitter', 4, 1, 0, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 0, None), ('belt', 5, 0, 4, None), ('splitter', 5, 2, 4, None), ('belt', 6, 0, 4, None), ('splitter', 6, 1, 4, None),
        ('splitter', 6, 3, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None),
        ('belt', 7, 4, 4, None),
    ],
    (1, 6, 1): [  # canonical_blueprints id=6705 juan's basic | 1:6 balancer
        ('belt', 0, 4, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 1, 8, None), ('belt', 2, 2, 4, None),
        ('belt', 2, 3, 4, None), ('belt', 2, 4, 0, None), ('belt', 2, 5, 8, None), ('belt', 2, 6, 4, None), ('belt', 2, 7, 4, None),
        ('belt', 2, 8, 0, None), ('belt', 2, 9, 0, None), ('belt', 3, 0, 12, None), ('splitter', 3, 2, 4, None), ('splitter', 3, 6, 4, None),
        ('belt', 3, 9, 12, None), ('belt', 4, 0, 12, None), ('splitter', 4, 1, 4, None), ('splitter', 4, 3, 4, None), ('splitter', 4, 5, 4, None),
        ('splitter', 4, 7, 4, None), ('belt', 4, 9, 12, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 4, None),
        ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None), ('belt', 5, 7, 4, None),
        ('belt', 5, 8, 8, None), ('belt', 5, 9, 12, None),
    ],
    (1, 7, 1): [  # canonical_blueprints id=466 EARLY FACTBOOK | 1 to 7
        ('belt', 0, 2, 4, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 0, None), ('belt', 1, 5, 0, None),
        ('belt', 1, 6, 0, None), ('belt', 2, 2, 4, None), ('belt', 2, 3, 4, None), ('splitter', 2, 5, 8, None), ('belt', 2, 6, 12, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 8, None), ('belt', 3, 6, 4, None), ('splitter', 4, 1, 4, None), ('belt', 4, 3, 8, None),
        ('belt', 4, 5, 4, None), ('belt', 4, 6, 4, None), ('splitter', 5, 0, 4, None), ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None),
        ('belt', 5, 6, 4, None),
    ],
    (1, 8, 1): [  # canonical_blueprints id=751 Belt Management | 1 to 8 TU
        ('belt', 0, 3, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 8, None), ('belt', 1, 5, 4, None), ('splitter', 2, 1, 4, None),
        ('belt', 2, 3, 8, None), ('belt', 2, 4, 12, None), ('splitter', 2, 5, 4, None), ('splitter', 3, 0, 4, None), ('splitter', 3, 2, 4, None),
        ('splitter', 3, 4, 4, None), ('splitter', 3, 6, 4, None),
    ],
    (2, 1, 1): [  # canonical_blueprints id=468 EARLY FACTBOOK | 2 to 1
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (2, 2, 1): [  # canonical_blueprints id=469 EARLY FACTBOOK | 2 to 2
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (2, 3, 1): [  # canonical_blueprints id=471 EARLY FACTBOOK | 2 to 3 (Wide)
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('splitter', 1, 1, 8, None), ('underground', 1, 2, 8, None), ('belt', 1, 3, 4, None),
        ('belt', 1, 4, 4, None), ('underground', 1, 5, 8, 'output'), ('belt', 1, 6, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 2, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 5, 4, None), ('belt', 2, 6, 0, None), ('belt', 3, 0, 12, None), ('splitter', 3, 2, 4, None),
        ('splitter', 3, 4, 4, None), ('belt', 4, 0, 12, None), ('belt', 4, 1, 0, None), ('belt', 4, 2, 0, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None),
    ],
    (2, 4, 1): [  # canonical_blueprints id=35 Raynquist's Belt Balancers Fall 2025 | 2-4 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
    ],
    (2, 5, 1): [  # canonical_blueprints id=1268 ALL THE BALANCERS | 2_5_balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 0, None),
        ('belt', 1, 5, 0, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('belt', 2, 5, 12, None), ('belt', 3, 1, 4, None),
        ('belt', 3, 2, 4, None), ('splitter', 3, 3, 4, None), ('belt', 3, 5, 12, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None),
        ('underground', 4, 4, 4, None), ('belt', 4, 5, 12, None), ('splitter', 5, 1, 4, None), ('belt', 5, 3, 8, None), ('belt', 5, 4, 8, None),
        ('belt', 5, 5, 12, None), ('splitter', 6, 0, 4, None), ('splitter', 6, 2, 4, None), ('underground', 6, 4, 4, 'output'), ('belt', 7, 0, 4, None),
        ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 4, None),
    ],
    (2, 6, 1): [  # canonical_blueprints id=38 Raynquist's Belt Balancers Fall 2025 | 2-6 balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None),
        ('splitter', 1, 3, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('belt', 1, 7, 0, None), ('belt', 2, 0, 12, None),
        ('splitter', 2, 1, 0, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None), ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 2, 0, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None),
        ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('splitter', 4, 4, 4, None),
        ('belt', 4, 6, 4, None), ('belt', 4, 7, 0, None), ('belt', 5, 1, 4, None), ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None),
        ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None),
    ],
    (2, 7, 1): [  # canonical_blueprints id=129 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 2
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('splitter', 1, 1, 8, None), ('underground', 1, 2, 8, None), ('belt', 1, 3, 4, None),
        ('belt', 1, 4, 4, None), ('underground', 1, 5, 8, 'output'), ('belt', 1, 6, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 2, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 5, 4, None), ('belt', 2, 6, 0, None), ('belt', 3, 0, 12, None), ('splitter', 3, 1, 0, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 2, 0, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 4, None), ('belt', 4, 5, 8, None), ('belt', 4, 6, 4, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 4, None),
        ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None), ('splitter', 5, 6, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 4, None),
    ],
    (2, 8, 1): [  # canonical_blueprints id=2525 8/8 | 1/8
        ('splitter', 0, 3, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 1, 4, None), ('belt', 2, 2, 0, None),
        ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None), ('belt', 2, 5, 8, None), ('belt', 2, 6, 4, None), ('splitter', 3, 0, 4, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('splitter', 3, 6, 4, None),
    ],
    (3, 1, 1): [  # canonical_blueprints id=958 | 3_1_balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('splitter', 1, 2, 4, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('belt', 2, 3, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 4, 0, 8, None), ('splitter', 4, 2, 4, None), ('belt', 5, 0, 12, None),
        ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None), ('belt', 5, 3, 4, None),
    ],
    (3, 2, 1): [  # canonical_blueprints id=1275 ALL THE BALANCERS | 3_2_balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 1, 4, None), ('underground', 1, 3, 4, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('belt', 2, 2, 8, None), ('belt', 2, 3, 8, None), ('belt', 2, 4, 4, None),
        ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None), ('belt', 3, 2, 4, None), ('underground', 3, 3, 4, 'output'), ('belt', 3, 4, 4, None),
        ('belt', 4, 0, 8, None), ('splitter', 4, 2, 4, None), ('belt', 4, 4, 4, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None),
        ('belt', 5, 2, 0, None), ('splitter', 5, 3, 4, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None),
    ],
    (3, 3, 1): [  # canonical_blueprints id=4025 Buildings + Upgrade Plans + Deconstruction plans | 3 to 3
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 1, 0, 4, None), ('belt', 1, 1, 4, None),
        ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 0, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None),
        ('belt', 2, 4, 12, None), ('belt', 3, 0, 4, None), ('splitter', 3, 1, 4, None), ('underground', 3, 3, 4, None), ('belt', 3, 4, 12, None),
        ('belt', 4, 0, 4, None), ('underground', 4, 1, 4, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 8, None), ('belt', 4, 4, 12, None),
        ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 2, 4, None), ('underground', 5, 3, 4, 'output'), ('underground', 6, 1, 4, 'output'),
        ('splitter', 6, 2, 4, None), ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None),
    ],
    (3, 4, 1): [  # canonical_blueprints id=133 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 3
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 0, None), ('belt', 1, 6, 0, None), ('splitter', 2, 1, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 6, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('splitter', 3, 5, 8, None), ('belt', 3, 6, 12, None), ('belt', 4, 0, 8, None),
        ('underground', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('underground', 4, 4, 8, 'output'), ('belt', 4, 6, 4, None),
        ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'), ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None), ('underground', 5, 4, 0, None),
        ('splitter', 5, 5, 0, None), ('belt', 5, 6, 0, None), ('belt', 6, 0, 8, None), ('belt', 6, 1, 4, None), ('splitter', 6, 2, 4, None),
        ('belt', 6, 4, 4, None), ('splitter', 7, 1, 4, None), ('splitter', 7, 3, 4, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 4, None), ('belt', 8, 4, 4, None),
    ],
    (3, 5, 1): [  # canonical_blueprints id=5376 The Essentials | 3 to 5
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 4, None),
        ('splitter', 1, 2, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('belt', 2, 0, 12, None),
        ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('splitter', 2, 4, 4, None), ('underground', 2, 6, 12, 'output'), ('belt', 3, 0, 12, None),
        ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None),
        ('belt', 3, 6, 4, None), ('belt', 4, 0, 12, None), ('underground', 4, 1, 0, 'output'), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('underground', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 2, 4, None),
        ('splitter', 5, 3, 4, None), ('belt', 6, 0, 12, None), ('splitter', 6, 1, 0, None), ('splitter', 6, 2, 4, None), ('belt', 6, 4, 8, None),
        ('belt', 6, 5, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 0, None), ('splitter', 7, 3, 4, None), ('belt', 7, 5, 4, None),
        ('underground', 7, 6, 12, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('underground', 8, 2, 4, 'output'), ('belt', 8, 3, 4, None),
        ('splitter', 8, 4, 4, None), ('belt', 8, 6, 12, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 2, 4, None), ('belt', 9, 4, 4, None),
        ('belt', 9, 5, 8, None), ('belt', 9, 6, 12, None),
    ],
    (3, 5, 2): [  # canonical_blueprints id=2536 8/8 | 3/5
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 4, None),
        ('splitter', 1, 2, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('belt', 2, 0, 12, None),
        ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('splitter', 2, 4, 4, None), ('underground', 2, 6, 12, 'output'), ('belt', 3, 0, 12, None),
        ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None),
        ('belt', 3, 6, 4, None), ('belt', 4, 0, 12, None), ('underground', 4, 1, 0, 'output'), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('underground', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 2, 4, None),
        ('splitter', 5, 3, 4, None), ('belt', 6, 0, 12, None), ('splitter', 6, 1, 0, None), ('splitter', 6, 2, 4, None), ('belt', 6, 4, 8, None),
        ('belt', 6, 5, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 0, None), ('splitter', 7, 3, 4, None), ('belt', 7, 5, 4, None),
        ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('underground', 8, 2, 4, 'output'), ('belt', 8, 3, 4, None), ('splitter', 8, 4, 4, None),
        ('underground', 8, 6, 12, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 2, 4, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 8, None),
        ('belt', 9, 6, 12, None),
    ],
    (3, 6, 1): [  # canonical_blueprints id=135 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 3
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 0, None), ('belt', 1, 6, 0, None), ('splitter', 2, 1, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 6, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('splitter', 3, 5, 8, None), ('belt', 3, 6, 12, None), ('belt', 4, 0, 8, None),
        ('underground', 4, 1, 8, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 8, 'output'), ('belt', 4, 6, 4, None), ('splitter', 5, 1, 4, None),
        ('splitter', 5, 3, 4, None), ('splitter', 5, 5, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None),
        ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None),
    ],
    (3, 7, 1): [  # canonical_blueprints id=136 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 3
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 0, None), ('belt', 1, 6, 0, None), ('belt', 1, 7, 0, None),
        ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None), ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None), ('belt', 3, 0, 4, None),
        ('belt', 3, 1, 0, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('splitter', 3, 5, 8, None),
        ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('underground', 4, 1, 8, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 8, 'output'),
        ('belt', 4, 6, 4, None), ('belt', 4, 7, 4, None), ('splitter', 5, 1, 4, None), ('splitter', 5, 3, 4, None), ('splitter', 5, 5, 4, None),
        ('belt', 5, 7, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None),
        ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 4, None),
    ],
    (3, 8, 1): [  # canonical_blueprints id=1834 The Complete Belt Series | 3 to 8
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None),
        ('belt', 1, 4, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None),
        ('belt', 2, 6, 12, None), ('belt', 2, 7, 0, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 0, None), ('belt', 3, 3, 4, None),
        ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None), ('splitter', 3, 6, 8, None), ('belt', 3, 7, 12, None), ('belt', 4, 1, 8, None),
        ('underground', 4, 2, 8, None), ('splitter', 4, 3, 4, None), ('underground', 4, 5, 8, 'output'), ('belt', 4, 7, 4, None), ('belt', 5, 1, 4, None),
        ('underground', 5, 2, 0, 'output'), ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 0, None), ('splitter', 5, 6, 0, None),
        ('belt', 5, 7, 0, None), ('belt', 6, 1, 8, None), ('belt', 6, 2, 4, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 4, None),
        ('splitter', 7, 2, 4, None), ('splitter', 7, 4, 4, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 0, None), ('belt', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 8, None), ('belt', 8, 6, 4, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 2, 4, None),
        ('splitter', 9, 4, 4, None), ('splitter', 9, 6, 4, None),
    ],
    (4, 1, 1): [  # canonical_blueprints id=966 | 4_1_balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('splitter', 2, 1, 4, None), ('belt', 3, 2, 4, None),
    ],
    (4, 2, 1): [  # canonical_blueprints id=2122 [Tobi] Perfect Belt Balancer Book !... | [font=default-large-bold][color=255,190,2
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('splitter', 2, 1, 4, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None),
    ],
    (4, 3, 1): [  # canonical_blueprints id=2541 8/8 | 4/3
        ('splitter', 0, 2, 4, None), ('splitter', 0, 4, 4, None), ('splitter', 1, 1, 0, None), ('belt', 1, 2, 0, None), ('splitter', 1, 3, 4, None),
        ('belt', 1, 5, 8, None), ('belt', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('underground', 2, 2, 0, 'output'), ('belt', 2, 3, 4, None),
        ('belt', 2, 4, 4, None), ('underground', 2, 5, 0, None), ('belt', 2, 6, 0, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None),
        ('underground', 3, 2, 8, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('underground', 3, 5, 8, 'output'), ('belt', 3, 6, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None),
        ('belt', 4, 6, 0, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None),
        ('belt', 6, 1, 12, None), ('belt', 6, 2, 0, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None),
    ],
    (4, 4, 1): [  # canonical_blueprints id=5382 One of the Blueprint Books Ever V1 | 4x4 Balancer
        ('splitter', 0, 1, 4, None), ('splitter', 0, 3, 4, None), ('underground', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 4, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('belt', 2, 2, 0, None), ('underground', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, 'output'), ('belt', 3, 2, 4, None), ('belt', 3, 3, 0, None), ('belt', 3, 4, 0, None),
        ('belt', 4, 0, 4, None), ('splitter', 4, 1, 4, None), ('underground', 4, 3, 4, 'output'), ('splitter', 5, 0, 4, None), ('splitter', 5, 2, 4, None),
    ],
    (4, 5, 1): [  # canonical_blueprints id=140 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 4
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 1, 0, 8, None),
        ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('belt', 2, 0, 12, None),
        ('splitter', 2, 2, 4, None), ('belt', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('belt', 2, 6, 4, None), ('belt', 3, 0, 12, None),
        ('underground', 3, 1, 0, 'output'), ('underground', 3, 2, 4, None), ('underground', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('underground', 3, 5, 0, None),
        ('belt', 3, 6, 0, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('underground', 4, 5, 4, 'output'), ('splitter', 5, 0, 12, None), ('belt', 5, 3, 4, None), ('splitter', 5, 4, 4, None), ('belt', 6, 0, 12, None),
        ('belt', 6, 1, 12, None), ('splitter', 6, 2, 0, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 4, None), ('belt', 7, 0, 12, None),
        ('belt', 7, 1, 4, None), ('belt', 7, 3, 0, None), ('belt', 7, 4, 4, None), ('belt', 7, 5, 8, None), ('belt', 7, 6, 4, None),
        ('belt', 8, 0, 12, None), ('belt', 8, 1, 4, None), ('underground', 8, 2, 4, 'output'), ('underground', 8, 3, 4, 'output'), ('splitter', 8, 4, 4, None),
        ('belt', 8, 6, 4, None), ('belt', 9, 0, 12, None), ('splitter', 9, 1, 4, None), ('splitter', 9, 3, 4, None), ('splitter', 9, 5, 4, None),
        ('belt', 10, 0, 12, None), ('belt', 10, 1, 0, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None), ('belt', 10, 4, 4, None),
        ('belt', 10, 5, 4, None), ('belt', 10, 6, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None),
        ('belt', 11, 5, 4, None), ('belt', 11, 6, 4, None),
    ],
    (4, 6, 1): [  # canonical_blueprints id=141 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 4
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 1, 0, 8, None),
        ('belt', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 12, None), ('splitter', 2, 1, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 5, 4, None), ('belt', 3, 0, 12, None), ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None),
        ('underground', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('belt', 4, 0, 12, None), ('belt', 4, 1, 0, None),
        ('belt', 4, 2, 0, None), ('belt', 4, 3, 0, None), ('belt', 4, 4, 0, None), ('belt', 4, 5, 0, None), ('belt', 5, 0, 8, None),
        ('belt', 5, 1, 8, None), ('belt', 5, 2, 4, None), ('underground', 5, 3, 4, 'output'), ('underground', 5, 4, 4, 'output'), ('belt', 6, 0, 12, None),
        ('splitter', 6, 1, 0, None), ('splitter', 6, 2, 4, None), ('belt', 6, 4, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 0, None),
        ('belt', 7, 3, 4, None), ('belt', 7, 4, 8, None), ('belt', 7, 5, 4, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'),
        ('underground', 8, 2, 4, 'output'), ('splitter', 8, 3, 4, None), ('belt', 8, 5, 4, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 2, 4, None),
        ('splitter', 9, 4, 4, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 4, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None),
        ('belt', 10, 4, 4, None), ('belt', 10, 5, 4, None),
    ],
    (4, 7, 1): [  # canonical_blueprints id=142 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 4
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 1, 1, 4, None),
        ('underground', 1, 2, 0, 'output'), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('underground', 1, 7, 0, None), ('belt', 1, 8, 0, None),
        ('underground', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('belt', 2, 6, 8, None),
        ('splitter', 2, 7, 8, None), ('belt', 2, 8, 12, None), ('belt', 3, 3, 8, None), ('belt', 3, 4, 8, None), ('belt', 3, 5, 8, None),
        ('belt', 3, 6, 8, None), ('belt', 3, 8, 4, None), ('splitter', 4, 2, 8, None), ('underground', 4, 3, 8, None), ('underground', 4, 6, 8, 'output'),
        ('belt', 4, 7, 4, None), ('belt', 4, 8, 4, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 3, 4, None),
        ('underground', 5, 4, 4, 'output'), ('underground', 5, 5, 4, 'output'), ('belt', 5, 7, 4, None), ('belt', 5, 8, 4, None), ('belt', 6, 0, 12, None),
        ('underground', 6, 1, 4, 'output'), ('splitter', 6, 2, 4, None), ('splitter', 6, 4, 4, None), ('splitter', 6, 6, 4, None), ('belt', 6, 8, 4, None),
        ('belt', 7, 0, 12, None), ('splitter', 7, 1, 4, None), ('splitter', 7, 3, 4, None), ('splitter', 7, 5, 4, None), ('splitter', 7, 7, 4, None),
        ('belt', 8, 0, 12, None), ('belt', 8, 1, 0, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 4, None), ('belt', 8, 4, 4, None),
        ('belt', 8, 5, 4, None), ('belt', 8, 6, 4, None), ('belt', 8, 7, 4, None), ('belt', 8, 8, 4, None), ('belt', 9, 2, 4, None),
        ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 4, None), ('belt', 9, 7, 4, None),
        ('belt', 9, 8, 4, None),
    ],
    (4, 8, 1): [  # canonical_blueprints id=143 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 4
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 1, 0, 4, None),
        ('underground', 1, 1, 0, 'output'), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('underground', 1, 6, 0, None), ('belt', 1, 7, 0, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None), ('belt', 2, 2, 0, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('belt', 2, 5, 8, None), ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 8, None),
        ('underground', 3, 2, 8, None), ('splitter', 3, 3, 4, None), ('underground', 3, 5, 8, 'output'), ('belt', 3, 7, 4, None), ('splitter', 4, 0, 4, None),
        ('splitter', 4, 2, 4, None), ('splitter', 4, 4, 4, None), ('splitter', 4, 6, 4, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 4, None),
        ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None),
        ('belt', 5, 7, 4, None),
    ],
    (5, 4, 1): [  # canonical_blueprints id=145 Belt balancer compendium with nice icons by tuckjohn37 and RedditNamesAreShort. | 5
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 1, 0, 4, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None),
        ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None),
        ('belt', 2, 6, 12, None), ('belt', 3, 0, 4, None), ('splitter', 3, 1, 4, None), ('underground', 3, 3, 4, None), ('underground', 3, 4, 4, None),
        ('belt', 3, 5, 4, None), ('belt', 3, 6, 12, None), ('belt', 4, 0, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None),
        ('splitter', 4, 4, 0, None), ('belt', 4, 5, 0, None), ('belt', 4, 6, 12, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 4, None),
        ('splitter', 5, 2, 4, None), ('belt', 5, 5, 0, None), ('belt', 5, 6, 12, None), ('splitter', 6, 1, 4, None), ('belt', 6, 3, 8, None),
        ('belt', 6, 4, 4, None), ('splitter', 6, 5, 12, None), ('underground', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('underground', 7, 3, 4, 'output'),
        ('belt', 7, 4, 8, None), ('belt', 7, 5, 12, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 0, 'output'), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 4, None), ('underground', 8, 4, 4, 'output'), ('underground', 8, 5, 0, None), ('belt', 8, 6, 0, None), ('belt', 9, 0, 4, None),
        ('underground', 9, 1, 4, 'output'), ('belt', 9, 2, 4, None), ('splitter', 9, 3, 4, None), ('belt', 9, 6, 12, None), ('splitter', 10, 0, 4, None),
        ('splitter', 10, 2, 4, None), ('belt', 10, 4, 8, None), ('belt', 10, 5, 8, None), ('belt', 10, 6, 12, None), ('belt', 11, 0, 4, None),
        ('belt', 11, 1, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None),
    ],
    (5, 6, 1): [  # canonical_blueprints id=2647 Jiro's Master Book | 5 to 6
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None),
        ('belt', 1, 7, 8, None), ('belt', 1, 8, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None),
        ('splitter', 2, 4, 4, None), ('belt', 2, 6, 12, None), ('underground', 2, 7, 12, 'output'), ('belt', 2, 8, 4, None), ('belt', 3, 0, 4, None),
        ('belt', 3, 1, 0, None), ('belt', 3, 2, 4, None), ('underground', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('underground', 3, 5, 4, None),
        ('belt', 3, 6, 12, None), ('splitter', 3, 7, 0, None), ('belt', 3, 8, 0, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None),
        ('belt', 4, 2, 0, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 0, None), ('belt', 4, 5, 0, None), ('splitter', 4, 6, 0, None),
        ('belt', 4, 8, 0, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 8, None), ('underground', 5, 2, 8, None), ('belt', 5, 3, 4, None),
        ('underground', 5, 4, 4, 'output'), ('belt', 5, 5, 4, None), ('underground', 5, 7, 8, 'output'), ('belt', 5, 8, 12, None), ('belt', 6, 0, 8, None),
        ('belt', 6, 1, 8, None), ('underground', 6, 2, 8, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 8, None), ('belt', 6, 6, 4, None),
        ('underground', 6, 7, 8, 'output'), ('belt', 6, 8, 4, None), ('underground', 7, 0, 4, 'output'), ('belt', 7, 1, 4, None), ('belt', 7, 2, 0, None),
        ('belt', 7, 3, 0, None), ('underground', 7, 4, 4, None), ('underground', 7, 5, 4, 'output'), ('belt', 7, 6, 4, None), ('underground', 7, 7, 12, None),
        ('belt', 7, 8, 4, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, None), ('underground', 8, 3, 4, 'output'), ('splitter', 8, 5, 4, None),
        ('belt', 8, 7, 12, None), ('belt', 8, 8, 4, None), ('belt', 9, 0, 8, None), ('belt', 9, 1, 8, None), ('belt', 9, 2, 4, None),
        ('belt', 9, 3, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 8, None), ('belt', 9, 7, 12, None), ('belt', 9, 8, 4, None),
        ('underground', 10, 1, 4, 'output'), ('splitter', 10, 2, 4, None), ('underground', 10, 4, 4, 'output'), ('belt', 10, 5, 4, None), ('belt', 10, 6, 4, None),
        ('belt', 10, 7, 0, None), ('belt', 10, 8, 0, None), ('splitter', 11, 1, 4, None), ('splitter', 11, 3, 4, None), ('splitter', 11, 5, 4, None),
    ],
    (6, 1, 1): [  # canonical_blueprints id=4018 Buildings + Upgrade Plans + Deconstruction plans | 6 to 1
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
        ('belt', 4, 0, 8, None), ('splitter', 4, 2, 4, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None),
        ('belt', 5, 3, 4, None),
    ],
    (6, 6, 1): [  # canonical_blueprints id=3304 Book of common blueprints | BeltBalancer 6 to 6
        ('splitter', 0, 3, 4, None), ('splitter', 0, 5, 4, None), ('splitter', 0, 7, 4, None), ('underground', 1, 3, 4, None), ('underground', 1, 4, 4, None),
        ('belt', 1, 5, 4, None), ('splitter', 1, 6, 4, None), ('belt', 1, 8, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None),
        ('splitter', 2, 2, 0, None), ('belt', 2, 3, 0, None), ('belt', 2, 4, 0, None), ('belt', 2, 5, 0, None), ('underground', 2, 6, 4, None),
        ('underground', 2, 7, 4, None), ('belt', 2, 8, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('underground', 3, 3, 0, 'output'),
        ('underground', 3, 4, 4, 'output'), ('underground', 3, 7, 0, None), ('belt', 3, 8, 0, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 8, None),
        ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 8, None), ('belt', 4, 5, 8, None), ('splitter', 4, 6, 8, None),
        ('belt', 4, 7, 8, None), ('belt', 4, 8, 4, None), ('belt', 5, 0, 8, None), ('splitter', 5, 1, 8, None), ('underground', 5, 2, 8, None),
        ('belt', 5, 3, 8, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 8, 'output'), ('belt', 5, 7, 4, None), ('belt', 5, 8, 4, None),
        ('belt', 6, 0, 8, None), ('belt', 6, 2, 4, None), ('underground', 6, 3, 4, 'output'), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None),
        ('belt', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 6, 8, 4, None), ('belt', 7, 0, 12, None), ('belt', 7, 1, 0, None),
        ('splitter', 7, 2, 4, None), ('belt', 7, 4, 4, None), ('belt', 7, 5, 4, None), ('underground', 7, 6, 4, 'output'), ('underground', 7, 7, 4, 'output'),
        ('belt', 7, 8, 4, None), ('belt', 8, 1, 12, None), ('belt', 8, 2, 0, None), ('splitter', 8, 3, 4, None), ('splitter', 8, 5, 4, None),
        ('splitter', 8, 7, 4, None),
    ],
    (6, 8, 1): [  # canonical_blueprints id=4017 Buildings + Upgrade Plans + Deconstruction plans | 6 to 8
        ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('belt', 0, 8, 4, None),
        ('belt', 0, 9, 4, None), ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('splitter', 1, 8, 4, None), ('belt', 2, 2, 4, None),
        ('belt', 2, 3, 0, None), ('belt', 2, 4, 0, None), ('underground', 2, 5, 4, None), ('belt', 2, 6, 4, None), ('splitter', 2, 7, 4, None),
        ('belt', 2, 9, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 0, 'output'), ('underground', 3, 2, 4, None), ('underground', 3, 4, 0, None),
        ('splitter', 3, 5, 0, None), ('belt', 3, 6, 0, None), ('underground', 3, 7, 4, None), ('underground', 3, 8, 4, None), ('belt', 3, 9, 4, None),
        ('belt', 4, 0, 4, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 4, 8, 0, None), ('belt', 4, 9, 0, None), ('belt', 5, 0, 4, None),
        ('belt', 5, 1, 12, None), ('underground', 5, 2, 4, 'output'), ('splitter', 5, 3, 4, None), ('underground', 5, 5, 4, 'output'), ('underground', 5, 7, 4, 'output'),
        ('underground', 5, 8, 4, 'output'), ('belt', 6, 0, 4, None), ('belt', 6, 1, 12, None), ('splitter', 6, 2, 4, None), ('splitter', 6, 4, 4, None),
        ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None), ('belt', 6, 8, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 1, 12, None),
        ('belt', 7, 2, 0, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 4, None), ('splitter', 7, 5, 4, None), ('belt', 7, 8, 8, None),
        ('belt', 7, 9, 4, None), ('belt', 8, 0, 8, None), ('belt', 8, 1, 8, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('underground', 8, 5, 4, None), ('underground', 8, 6, 4, None), ('belt', 8, 7, 8, None), ('belt', 8, 8, 4, None),
        ('belt', 8, 9, 4, None), ('splitter', 9, 2, 4, None), ('belt', 9, 4, 8, None), ('belt', 9, 5, 8, None), ('belt', 9, 6, 8, None),
        ('belt', 9, 7, 12, None), ('splitter', 9, 8, 4, None), ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('underground', 10, 5, 4, 'output'),
        ('underground', 10, 6, 4, 'output'), ('splitter', 10, 7, 4, None), ('belt', 10, 9, 4, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None),
        ('splitter', 11, 6, 4, None), ('splitter', 11, 8, 4, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None),
        ('belt', 12, 5, 4, None), ('belt', 12, 6, 4, None), ('belt', 12, 7, 4, None), ('belt', 12, 8, 4, None), ('belt', 12, 9, 4, None),
    ],
    (8, 8, 1): [  # canonical_blueprints id=3603 Bored Bob Space Exploration Playthrough Print Book | 8-8 Balancer
        ('splitter', 0, 2, 4, None), ('splitter', 0, 4, 4, None), ('splitter', 0, 6, 4, None), ('splitter', 0, 8, 4, None), ('belt', 1, 2, 4, None),
        ('splitter', 1, 3, 4, None), ('underground', 1, 5, 4, None), ('belt', 1, 6, 4, None), ('splitter', 1, 7, 4, None), ('belt', 1, 9, 4, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('belt', 2, 2, 0, None), ('underground', 2, 3, 4, None), ('underground', 2, 4, 4, None),
        ('belt', 2, 5, 4, None), ('belt', 2, 6, 0, None), ('underground', 2, 7, 4, None), ('underground', 2, 8, 4, None), ('belt', 2, 9, 4, None),
        ('belt', 3, 0, 8, None), ('underground', 3, 1, 8, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 0, None), ('belt', 3, 4, 0, None),
        ('belt', 3, 5, 0, None), ('underground', 3, 6, 8, 'output'), ('belt', 3, 7, 8, None), ('belt', 3, 8, 4, None), ('belt', 3, 9, 8, None),
        ('belt', 3, 10, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('underground', 4, 4, 0, 'output'), ('underground', 4, 5, 4, 'output'),
        ('belt', 4, 6, 8, None), ('belt', 4, 7, 4, None), ('belt', 4, 8, 4, None), ('underground', 4, 9, 0, None), ('belt', 4, 10, 0, None),
        ('splitter', 5, 2, 4, None), ('belt', 5, 5, 8, None), ('belt', 5, 6, 12, None), ('splitter', 5, 7, 4, None), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 8, None), ('belt', 6, 4, 8, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None),
        ('belt', 6, 8, 8, None), ('belt', 6, 9, 4, None), ('belt', 7, 2, 4, None), ('underground', 7, 3, 4, 'output'), ('underground', 7, 4, 4, 'output'),
        ('belt', 7, 5, 4, None), ('belt', 7, 6, 4, None), ('underground', 7, 7, 4, 'output'), ('underground', 7, 8, 4, 'output'), ('belt', 7, 9, 4, None),
        ('splitter', 8, 2, 4, None), ('splitter', 8, 4, 4, None), ('splitter', 8, 6, 4, None), ('splitter', 8, 8, 4, None),
    ],
    (8, 8, 2): [  # canonical_blueprints id=3200 Mega Base build V2.0 | 8-8 balancer [Express]
        ('splitter', 0, 0, 4, None), ('splitter', 0, 2, 4, None), ('splitter', 0, 4, 4, None), ('splitter', 0, 6, 4, None), ('belt', 1, 0, 4, None),
        ('splitter', 1, 1, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('splitter', 1, 5, 4, None), ('belt', 1, 7, 4, None),
        ('belt', 2, 0, 4, None), ('underground', 2, 1, 4, None), ('underground', 2, 2, 4, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('underground', 2, 5, 4, None), ('underground', 2, 6, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 8, None), ('belt', 3, 1, 8, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('belt', 3, 5, 8, None), ('belt', 3, 6, 4, None),
        ('belt', 3, 7, 4, None), ('splitter', 4, 2, 4, None), ('splitter', 4, 6, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'),
        ('belt', 5, 2, 4, None), ('belt', 5, 3, 8, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 0, None), ('belt', 5, 6, 0, None),
        ('belt', 5, 7, 4, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 0, None), ('underground', 6, 4, 4, None),
        ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 4, None), ('belt', 7, 1, 8, None),
        ('underground', 7, 2, 8, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 0, None), ('belt', 7, 5, 0, None), ('underground', 7, 6, 8, 'output'),
        ('belt', 7, 7, 4, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('underground', 8, 2, 4, 'output'), ('belt', 8, 3, 4, None),
        ('underground', 8, 4, 4, 'output'), ('underground', 8, 5, 4, 'output'), ('underground', 8, 6, 4, 'output'), ('belt', 8, 7, 4, None), ('splitter', 9, 0, 4, None),
        ('splitter', 9, 2, 4, None), ('splitter', 9, 4, 4, None), ('splitter', 9, 6, 4, None),
    ],
}
# END BALANCER TEMPLATES
