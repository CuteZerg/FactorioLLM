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


def _apply_modules(ent: Any, modules: Dict[str, int]) -> None:
    """Fills module slots in order, e.g. {'speed-module-3': 2} or {'productivity-module-3': 4}."""
    slots = int(_entity_data.raw[ent.name].get("module_slots") or 0)
    if sum(int(n) for n in modules.values()) > slots:
        raise ValueError(f"{ent.name} has {slots} module slots, got {sum(modules.values())} modules.")
    slot = 0
    for module, count in modules.items():
        ent.request_modules(str(module), range(slot, slot + int(count)))
        slot += int(count)


def _apply_requests(ent: Any, requests: Dict[str, int]) -> None:
    """Logistic requests of a requester / buffer chest (or a storage chest filter): {'iron-plate': 200}."""
    section = ent.add_section()
    section.index = 0  # draftsman 3.3.1 numbers a new section one too high on export
    for i, (item, count) in enumerate(requests.items()):
        section.set_signal(i, str(item), count=int(count))


def _instantiate_entity(name_or_class: Any, tile: Tuple[int, int], direction: Optional[int] = None, **kwargs) -> Any:
    """
    Instantiates an entity by prototype name (or draftsman class) at a tile position.
    modules={'speed-module-3': 2} fills module slots; requests={'iron-plate': 200} sets logistic
    requests; request_from_buffers / trash_not_requested are logistic chest flags.
    """
    modules = kwargs.pop("modules", None)
    requests = kwargs.pop("requests", None)
    flags = {k: kwargs.pop(k) for k in ("request_from_buffers", "trash_not_requested") if k in kwargs}
    call_kwargs = dict(tile_position=tile, **kwargs)
    if direction:
        call_kwargs["direction"] = direction

    if isinstance(name_or_class, type):
        ent = name_or_class(**call_kwargs)
    else:
        name = str(name_or_class).strip()
        _check_entity_name(name)
        ent = new_entity(name, **call_kwargs)
    if modules:
        _apply_modules(ent, modules)
    if requests:
        _apply_requests(ent, requests)
    for key, value in flags.items():
        setattr(ent, key, bool(value))
    return ent


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


def add_entity(
    bp: Any,
    entity: Any,
    tile: Union[Tuple[int, int], Dict[str, int]],
    direction: Union[int, str, Direction, None] = None,
    **kwargs,
) -> None:
    """
    Appends one entity at a tile, e.g. a beacon with modules or a requester chest with requests:
    add_entity(bp, 'beacon', (3, 0), modules={'speed-module-3': 2})
    add_entity(bp, 'requester-chest', (0, 2), requests={'iron-plate': 200, 'copper-cable': 400})
    """
    dir_int = resolve_direction(direction) if direction is not None else None
    _PlacementGuard(bp).append(_instantiate_entity(entity, _resolve_tile(tile), direction=dir_int, **kwargs))


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


def _signal_args(signal: Any) -> Tuple[str, int, Optional[str]]:
    """(name, count, type) from 'iron-ore', ('iron-ore', 1) or ('signal-A', 5, 'virtual')."""
    if isinstance(signal, str):
        return signal, 1, None
    name, count, *rest = signal
    kind = rest[0] if rest else None
    return str(name), int(count), (None if kind in (None, "item") else str(kind))


def add_signal_label(
    bp: Any,
    tile: Union[Tuple[int, int], Dict[str, int]],
    signals: List[Any],
    direction: Union[int, str, Direction, None] = None,
) -> None:
    """
    Constant combinator used as a label, e.g. at the start of a belt to show what it carries.
    signals: item names, (name, count) or (name, count, type) with type 'fluid', 'virtual',
    'entity' or 'recipe', e.g. ['iron-ore', 'coal'] or [('signal-A', 1, 'virtual')].
    """
    ent = _instantiate_entity("constant-combinator", _resolve_tile(tile),
                              direction=resolve_direction(direction) if direction is not None else None)
    for i, signal in enumerate(signals):
        name, count, kind = _signal_args(signal)
        ent.set_signal(i, name, count, type=kind)
    _PlacementGuard(bp).append(ent)


def add_display_panel(
    bp: Any,
    tile: Union[Tuple[int, int], Dict[str, int]],
    text: str = "",
    icon: Optional[Any] = None,
    direction: Union[int, str, Direction, None] = None,
) -> None:
    """Display panel with a text and/or an icon (item name or (name, type)) to mark a block."""
    ent = _instantiate_entity("display-panel", _resolve_tile(tile),
                              direction=resolve_direction(direction) if direction is not None else None)
    if text:
        ent.text = str(text)
    if icon is not None:
        name, kind = (icon, "item") if isinstance(icon, str) else (icon[0], icon[1])
        ent.icon = {"name": str(name), "type": str(kind)}
    _PlacementGuard(bp).append(ent)


def game_string(bp: Any, version: Any = None) -> str:
    """
    Blueprint string as the base game (no quality mod) exports it: draftsman writes a splitter filter
    as {'type': 'item', 'name': ...}, the game writes {'name': ...}. The sandbox uses it for
    Blueprint.to_string().
    """
    from draftsman.utils import JSON_to_string

    def fix(node: Dict[str, Any]) -> None:
        for ent in (node.get("blueprint") or {}).get("entities") or []:
            flt = ent.get("filter")
            if isinstance(flt, dict) and flt.get("name") and flt.get("type", "item") == "item" \
                    and flt.get("quality", "normal") == "normal":
                ent["filter"] = {"name": flt["name"]}
        for child in (node.get("blueprint_book") or {}).get("blueprints") or []:
            fix(child)

    data = bp.to_dict(version=version) if version is not None else bp.to_dict()
    fix(data)
    return JSON_to_string(data)


def connect_power(bp: Any, max_wires: int = 5) -> int:
    """
    Replaces all copper wires with a minimal wire network like the one a player builds: the
    closest pole pairs first (axis-aligned ones on ties), no triangles, at most `max_wires`
    wires per pole, never longer than the shorter reach of the two poles. Pasted blueprint
    poles do not wire themselves to each other in game, so every blueprint needs this.
    The sandbox calls it automatically before Blueprint.to_string(). Returns the wire count.
    """
    poles = [e for e in bp.entities if _entity_data.raw.get(e.name, {}).get("type") == "electric-pole"]
    bp.remove_power_connections()
    if len(poles) < 2:
        return 0
    centers = [(e.tile_position.x + e.tile_width / 2, e.tile_position.y + e.tile_height / 2) for e in poles]
    reach = [float(_entity_data.raw[e.name].get("maximum_wire_distance", 0)) for e in poles]
    order = sorted(range(len(poles)), key=lambda i: centers[i][0])
    longest = max(reach)
    pairs = []
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            dx = centers[j][0] - centers[i][0]
            if dx > longest:
                break
            dy = centers[j][1] - centers[i][1]
            dist = math.hypot(dx, dy)
            if dist <= min(reach[i], reach[j]) + 1e-9:
                pairs.append((round(dist, 6), dx != 0 and dy != 0, min(i, j), max(i, j)))
    pairs.sort()
    parent = list(range(len(poles)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    degree = [0] * len(poles)
    wires = 0
    for capped in (True, False):  # second pass only if the wire cap split the network
        for _, _, i, j in pairs:
            ri, rj = root(i), root(j)
            if ri == rj or (capped and (degree[i] >= max_wires or degree[j] >= max_wires)):
                continue
            bp.add_power_connection(poles[i], poles[j])
            parent[ri] = rj
            degree[i] += 1
            degree[j] += 1
            wires += 1
    return wires


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
        self.pole_tiles: Optional[set] = None  # where poles may stand (None: any free tile)

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

    def translated(self, du: int, dv: int) -> "_Layout":
        """Copy shifted by (du, dv), e.g. to make room for a junction; entity indices are kept."""
        out = _Layout()
        for name, u, v, direction, kwargs in self.items:
            out.add(name, u + du, v + dv, direction, **kwargs)
        if self.pole_tiles is not None:
            out.pole_tiles = {(u + du, v + dv) for u, v in self.pole_tiles}
        return out

    def flipped(self) -> "_Layout":
        """Mirror image across the flow axis (north <-> south), e.g. to put engines on the other side."""
        out = _Layout()
        for name, u, v, direction, kwargs in self.items:
            w, h = _footprint(name, direction)
            new_dir = direction if direction is None or direction in (4, 12) else (direction + 8) % 16
            out.add(name, u, self.height - v - h, new_dir, **kwargs)
        if self.pole_tiles is not None:
            out.pole_tiles = {(u, self.height - 1 - v) for u, v in self.pole_tiles}
        return out

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
        # tight bounds: rows or columns reserved for poles that ended up unused are dropped
        u0, v0 = min(t[0] for t in self.tiles), min(t[1] for t in self.tiles)
        W = max(t[0] for t in self.tiles) + 1 - u0
        H = max(t[1] for t in self.tiles) + 1 - v0
        guard = _PlacementGuard(bp)
        entities = []
        for name, u, v, d, kwargs in self.items:
            u, v = u - u0, v - v0
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
    candidates = [(u, v) for u in range(layout.width) for v in range(layout.height) if layout.is_free(u, v, pw, ph)
                  and (layout.pole_tiles is None or (u, v) in layout.pole_tiles)]

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
                       inserter: str, layout_kind: str = "shared_input",
                       **building_kwargs) -> Tuple[_Layout, List[int], List[int]]:
    """
    Rows of buildings fed by inserters from input belts and emptied onto output belts, in the
    local frame (items flow east). Returns (layout, powered entity indices, input belt rows).
      shared_input: rows=2 are mirrored around shared input belt(s); outputs on the outside.
      outer_input:  rows=2 take input from their own outer belt(s) and output onto one shared
                    middle belt (both lanes filled), the standard for coal-fired furnaces.
    All inserters of a building stand in one column (its middle one); poles go in the gaps.
    """
    _check_type(inserter, ("inserter",))
    size = _footprint(building, None)[0]
    rows = min(rows, count)
    per_row = [count - count // rows, count // rows] if rows == 2 else [count]
    length = per_row[0] * size
    mid = (size - 1) // 2
    layout = _Layout()
    powered: List[int] = []
    input_rows: List[int] = []

    def belt_row(v: int) -> None:
        for u in range(length):
            layout.add(belt, u, v, 4)

    layout.pole_tiles = set()  # poles stand in the gaps of the inserter rows

    def machines(n: int, ins_v: int, bld_v: int, out_v: Optional[int], picks_from_north: bool) -> None:
        ins_dir = 0 if picks_from_north else 8  # an inserter faces the side it picks up from
        out_dir = None if out_v is None else 0 if bld_v < out_v else 8
        for v in (ins_v, out_v):
            if v is not None:
                layout.pole_tiles.update((u, v) for u in range(length))
        for i in range(n):
            u = i * size
            if inputs == 1:
                powered.append(layout.add(inserter, u + mid, ins_v, ins_dir))
            else:  # far belt by the long-handed inserter, near belt by the normal one
                powered.append(layout.add("long-handed-inserter", u, ins_v, ins_dir))
                powered.append(layout.add(inserter, u + size - 1, ins_v, ins_dir))
            powered.append(layout.add(building, u, bld_v, **building_kwargs))
            if out_v is not None:
                powered.append(layout.add(inserter, u + mid, out_v, out_dir))

    if layout_kind == "outer_input" and rows == 2:
        for v in range(inputs):
            belt_row(v)
            input_rows.append(v)
        machines(per_row[1], inputs, inputs + 1, inputs + 1 + size, picks_from_north=True)
        shared = inputs + size + 2
        belt_row(shared)
        machines(per_row[0], shared + 2 + size, shared + 2, shared + 1, picks_from_north=False)
        for k in range(inputs):
            belt_row(shared + 3 + size + k)
            input_rows.append(shared + 3 + size + k)
        return layout, powered, input_rows

    v = 0
    if rows == 2:  # mirrored upper row: output belt, output inserters, buildings, input inserters
        top = 2 if output else 0
        if output:
            belt_row(v)
        machines(per_row[1], v + top + size, v + top, v + 1 if output else None, picks_from_north=False)
        v += top + size + 1
    for k in range(inputs):
        belt_row(v + k)
        input_rows.append(v + k)
    v += inputs
    machines(per_row[0], v, v + 1, v + 1 + size if output else None, picks_from_north=True)
    if output:
        belt_row(v + 2 + size)
    return layout, powered, input_rows


def _power(layout: _Layout, pole: Optional[str], powered: List[int]) -> None:
    if pole:
        _add_poles(layout, pole, [i for i in powered if _is_electric(layout.items[i][0])])


def _add_fuel_feed(layout: _Layout, input_rows: List[int], belt: str) -> None:
    """
    Junction in front of a coal-fired smelting array: one ore belt and one coal belt come in from
    the upstream (west) edge and are side-loaded onto separate lanes of every input belt (ore on
    the inner lanes, coal on the outer ones). Two input rows need 6 free columns, one row needs 3,
    plus one free row above the first and below the last input belt.
    """
    _, underground, splitter = _belt_names(belt)
    c = min(u for name, u, v, d, kw in layout.items if v in input_rows) - 1  # first tile of the input belts
    east, south, north = 4, 8, 0
    for v in input_rows:
        layout.add(belt, c, v, east)  # T-junction tile: fed from both sides, each side fills its near lane
    if len(input_rows) == 1:
        a = input_rows[0]
        for u in (c - 2, c - 1):
            layout.add(belt, u, a - 1, east)  # coal
            layout.add(belt, u, a + 1, east)  # ore
        layout.add(belt, c, a - 1, south)
        layout.add(belt, c, a + 1, north)
        return
    a, b = input_rows
    m = (a + b) // 2
    # coal: in at the top-left, split; one half straight into A, the other down (under the ore) into B
    layout.add(belt, c - 5, a - 1, east)
    layout.add(splitter, c - 4, a - 1, east)
    for u in (c - 3, c - 2, c - 1):
        layout.add(belt, u, a - 1, east)
    layout.add(belt, c, a - 1, south)
    for v in range(a, m - 2):
        layout.add(belt, c - 3, v, south)
    layout.add(underground, c - 3, m - 2, south, io_type="input")
    layout.add(underground, c - 3, m + 2, south, io_type="output")
    for v in range(m + 3, b + 1):
        layout.add(belt, c - 3, v, south)
    for u in (c - 3, c - 2, c - 1):
        layout.add(belt, u, b + 1, east)
    layout.add(belt, c, b + 1, north)
    # ore: in at the middle of the west edge, split; up into A and down into B
    for u in (c - 5, c - 4, c - 3):
        layout.add(belt, u, m, east)
    layout.add(splitter, c - 2, m - 1, east)
    layout.add(belt, c - 1, m - 1, east)
    layout.add(belt, c - 1, m, east)
    for v in range(a + 1, m):
        layout.add(belt, c, v, north)
    for v in range(m, b):
        layout.add(belt, c, v, south)


def smelting_column(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 48,
    furnace: str = "stone-furnace",
    belt: str = "transport-belt",
    inserter: str = "inserter",
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    layout: Optional[str] = None,
    feed: Optional[str] = None,
    direction: Union[int, str, Direction] = "south",
) -> BBox:
    """
    Smelting array: `count` furnaces in `rows` (1 or 2) lines along the flow `direction`.
    layout="outer_input" (default for coal-fired stone/steel furnaces): ore + coal arrive on the
    two outer belts (one lane each), plates leave on the shared middle belt, which both rows fill;
    layout="shared_input" (default for electric furnaces): ore on the middle belt, plates outside.
    feed="split" (default with coal): a junction at the upstream end takes one ore belt (middle of
    the edge) and one coal belt (corner) and puts them on separate lanes; feed="lanes": the input
    belts already carry ore and coal on separate lanes. One full belt of ore feeds 48 stone or
    24 steel furnaces (yellow belt; x2 red, x3 blue). Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_type(furnace, ("furnace",))
    _belt_names(belt)
    burner = (_entity_data.raw[furnace].get("energy_source") or {}).get("type") == "burner"
    layout = _check_choice("layout", layout or ("outer_input" if burner else "shared_input"),
                           ("outer_input", "shared_input"))
    feed = _check_choice("feed", feed or ("split" if burner else "lanes"), ("split", "lanes"))
    plan, powered, input_rows = _production_layout(furnace, count, rows, 1, True, belt, inserter, layout)
    if feed == "split" and burner:
        plan = plan.translated(6 if len(input_rows) == 2 else 3, 1)
        _add_fuel_feed(plan, [v + 1 for v in input_rows], belt)
    _power(plan, pole, powered)
    return plan.commit(bp, origin, direction)


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
    plan, powered, _ = _production_layout(machine, count, rows, inputs, True, belt, inserter, recipe=recipe)
    _power(plan, pole, powered)
    return plan.commit(bp, origin, direction)


def _lab_chains(count: int, rows: int, inputs: int, depth: int, belt: str, inserter: str) -> Tuple[_Layout, List[int]]:
    """
    Labs in columns of up to `depth`, 1-tile gaps between labs: the first lab of a column is fed
    from the belt, inserters in the gaps pass the packs on to the next lab. rows=2: columns grow
    to both sides of the belt(s). Local frame, items flow east.
    """
    _check_type(inserter, ("inserter",))
    per_side = [count - count // rows, count // rows] if rows == 2 else [count]
    columns = math.ceil(per_side[0] / depth)
    length = columns * 4 - 1
    plan = _Layout()
    powered: List[int] = []
    side_h = 1 + depth * 4 - 1  # feed inserter row + labs with gap rows between them

    def side(n: int, v_belt_edge: int, upward: bool) -> None:
        sign = -1 if upward else 1
        ins_dir = 8 if upward else 0  # picks from the belt / the previous lab
        feed_v = v_belt_edge + sign

        def lab_v(r: int) -> int:  # top row of the r-th lab from the belt
            near = feed_v + sign * (1 + 4 * r)
            return near - 2 if upward else near

        for k in range(n):
            c, r = divmod(k, depth)
            u = 4 * c
            if r == 0:
                if inputs == 1:
                    powered.append(plan.add(inserter, u + 1, feed_v, ins_dir))
                else:
                    powered.append(plan.add("long-handed-inserter", u, feed_v, ins_dir))
                    powered.append(plan.add(inserter, u + 2, feed_v, ins_dir))
            else:
                gap_v = lab_v(r) + 3 if upward else lab_v(r) - 1
                powered.append(plan.add(inserter, u + 1, gap_v, ins_dir))
            powered.append(plan.add("lab", u, lab_v(r)))

    v = side_h if rows == 2 else 0
    for k in range(inputs):
        for u in range(length):
            plan.add(belt, u, v + k, 4)
    if rows == 2:
        side(per_side[1], v, upward=True)
    side(per_side[0], v + inputs - 1, upward=False)
    return plan, powered  # every free tile is a gap or a feed row: poles may go anywhere


def lab_array(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 10,
    belt: str = "transport-belt",
    inserter: str = "inserter",
    inputs: int = 1,
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    layout: Optional[str] = None,
    chain_depth: int = 4,
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Research block of `count` labs fed with science packs from the belt(s) (inputs=1: 2 pack
    types, inputs=2: 4 pack types) on `rows` (1 or 2) sides. layout="belt" (default up to 10
    labs): every lab takes packs from the belt; layout="chain" (default above 10): only the first
    lab of each column is fed from the belt and inserters pass packs from lab to lab, columns of
    up to `chain_depth` (1-4) labs. Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_choice("inputs", inputs, (1, 2))
    _check_count("chain_depth", chain_depth, 1, 4)
    _belt_names(belt)
    layout = _check_choice("layout", layout or ("belt" if count <= 10 else "chain"), ("belt", "chain"))
    if layout == "chain":
        plan, powered = _lab_chains(count, min(rows, count), inputs, chain_depth, belt, inserter)
    else:
        plan, powered, _ = _production_layout("lab", count, rows, inputs, False, belt, inserter)
    _power(plan, pole, powered)
    return plan.commit(bp, origin, direction)


def green_circuits(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 4,
    machine: str = "assembling-machine-2",
    belt: str = "transport-belt",
    inserter: str = "inserter",
    link_inserter: Optional[str] = None,
    pole: Optional[str] = "medium-electric-pole",
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Electronic circuits by direct insertion: blocks of cable-circuit-cable-circuit-cable machines
    (the 3:2 ratio), cable goes straight into the circuit machines through `link_inserter`s
    (default: fast for assembling-machine-1, bulk otherwise). One input belt carries copper and
    iron plates on separate lanes; circuits leave on the output belt. `count` circuit machines.
    Returns the BBox of the module.
    """
    _check_count("count", count)
    _check_type(machine, ("assembling-machine",))
    _check_type(inserter, ("inserter",))
    _belt_names(belt)
    link = link_inserter or ("fast-inserter" if machine == "assembling-machine-1" else "bulk-inserter")
    _check_type(link, ("inserter",))
    plan = _Layout()
    powered: List[int] = []
    u = 0
    left = count
    plan.pole_tiles = set()
    while left:
        pairs = min(2, left)  # circuit machines in this block
        left -= pairs
        kinds = ["copper-cable", "electronic-circuit"] * pairs + ["copper-cable"]
        for k, recipe in enumerate(kinds):
            x = u + 4 * k
            powered.append(plan.add(inserter, x + 1, 1, 0))  # copper or iron from the input belt
            powered.append(plan.add(machine, x, 2, recipe=recipe))
            if recipe == "electronic-circuit":
                powered.append(plan.add(link, x - 1, 3, 12))  # cable from the west machine
                powered.append(plan.add(link, x + 3, 3, 4))  # cable from the east machine
                powered.append(plan.add(inserter, x + 1, 5, 0))  # circuits onto the output belt
            plan.pole_tiles |= {(x + 3, 2), (x + 3, 4), (x, 1), (x + 2, 1), (x, 5), (x + 2, 5)}
        u += 4 * len(kinds)
    length = u - 1
    for v in (0, 6):
        for x in range(length):
            plan.add(belt, x, v, 4)
    plan.pole_tiles = {t for t in plan.pole_tiles if t[0] < length}
    _power(plan, pole, powered)
    return plan.commit(bp, origin, direction)


def mining_field(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    count: int = 16,
    belt: str = "transport-belt",
    pole: Optional[str] = "medium-electric-pole",
    rows: int = 2,
    layout: str = "pole_gaps",
    belts: int = 1,
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Ore field: `count` electric mining drills facing `belts` parallel output belts that flow in
    `direction`, on one or both sides of each belt (`rows`=1|2). Layouts:
      pole_gaps:  a 1-tile gap for poles after drills 0, 2, 4, ... (cheap, simple);
      belt_poles: drills touch; poles stand in the belt line, which passes them underground
                  (densest; two undergrounds per pole);
      pole_row:   drills touch; a row of poles above/below every belt pair, shared between
                  neighbouring pairs (medium poles or larger).
    A yellow belt carries the ore of about 30 drills (15 per side; x2 red, x3 blue).
    Returns the BBox of the field.
    """
    _check_count("count", count)
    _check_choice("rows", rows, (1, 2))
    _check_choice("layout", layout, ("pole_gaps", "belt_poles", "pole_row"))
    _check_count("belts", belts, 1, 20)
    _, underground, _ = _belt_names(belt)
    sides = belts * rows
    if count < sides:
        raise ValueError(f"{count} drills cannot fill {belts} belt(s) x {rows} side(s); use fewer belts or rows.")
    per_side = [count // sides + (1 if k < count % sides else 0) for k in range(sides)]
    plan = _Layout()
    powered: List[int] = []
    strip = 1 if layout == "pole_row" else 0  # pole row above every belt pair
    pair_h = (7 if rows == 2 else 4) + strip
    width = 0
    belt_rows = []
    for b in range(belts):
        top = b * pair_h + strip
        belt_v = top + 3
        belt_rows.append(belt_v)
        for side in range(rows):
            u = 0
            for i in range(per_side[b * rows + side]):
                powered.append(plan.add("electric-mining-drill", u, top if side == 0 else belt_v + 1, 8 if side == 0 else 0))
                u += 4 if layout == "pole_gaps" and i % 2 == 0 else 3
            width = max(width, u)
    if layout == "pole_row":
        strips = belts + 1 if rows == 2 else belts  # one-sided pairs need no row below the belt
        plan.pole_tiles = {(u, b * pair_h) for b in range(strips) for u in range(width)}
        if pole and strips > 1 and _pole_geometry(pole)[3] < pair_h:
            # pole rows are too far apart to wire: connector poles in an extra end column
            plan.pole_tiles |= {(width, v) for v in range(belts * pair_h + 1) if v not in belt_rows}
            width += 1
    if layout != "belt_poles":
        for v in belt_rows:
            for u in range(width):
                plan.add(belt, u, v, 4)
        if pole:
            if layout == "pole_row" and rows == 2:
                plan.height = max(plan.height, belts * pair_h + 1)  # bottom pole row
            _add_poles(plan, pole, powered)
        return plan.commit(bp, origin, direction)

    # belt_poles: poles stand in the belt line and the belt passes each one underground
    if not pole:
        raise ValueError("layout='belt_poles' needs a pole type.")
    pw, ph, supply, reach = _pole_geometry(pole)
    if (pw, ph) != (1, 1):
        raise ValueError("layout='belt_poles' needs a 1x1 pole (small or medium electric pole).")
    if width < 6:
        raise ValueError("layout='belt_poles' needs at least 2 drills per side; use layout='pole_gaps'.")
    spans = sorted({plan.items[i][1] for i in powered})  # drill columns (3 wide)
    poles: List[int] = []
    while True:
        uncovered = [s for s in spans if not any(s < p + 0.5 + supply and s + 3 > p + 0.5 - supply for p in poles)]
        if not uncovered:
            break
        s = uncovered[0]
        # not on a drill output column (3j+1), undergrounds of neighbouring poles must not overlap
        options = [p for p in range(1, width - 1) if p % 3 != 1 and s < p + 0.5 + supply and s + 3 > p + 0.5 - supply
                   and (not poles or (p >= poles[-1] + 3 and p - poles[-1] <= reach))]
        if not options:
            raise ValueError(f"{pole} cannot power this belt_poles field; use layout='pole_gaps'.")
        poles.append(max(options))
    for v in belt_rows:
        used = set()
        for p in poles:
            plan.add(underground, p - 1, v, 4, io_type="input")
            plan.add(pole, p, v)
            plan.add(underground, p + 1, v, 4, io_type="output")
            used.update((p - 1, p, p + 1))
        for u in range(width):
            if u not in used:
                plan.add(belt, u, v, 4)
    return plan.commit(bp, origin, direction)


# An 18x18 substation cell without wasted tiles: eight 6x6 blocks (4 panels or 9 accumulators
# each) plus 8 fixed accumulator slots around the central substation. One accumulator block gives
# 28 panels / 17 accumulators, two give 24 / 26; alternating them is close to the ideal 0.84.
_CELL_BLOCKS = [(0, 0), (6, 0), (12, 0), (0, 6), (12, 6), (0, 12), (6, 12), (12, 12)]
_CELL_FIXED_ACCUMULATORS = [(6, 6), (6, 8), (6, 10), (10, 6), (10, 8), (10, 10), (8, 6), (8, 10)]


def _solar_cells(plan: _Layout, panels: int, accumulators: int) -> None:
    def blocks_needed(cells: int) -> int:
        return math.ceil(panels / 4) + math.ceil(max(0, accumulators - 8 * cells) / 9)

    cells = 1
    while blocks_needed(cells) > 8 * cells:
        cells += 1
    cols = math.ceil(math.sqrt(cells))
    origins = [((k % cols) * 18, (k // cols) * 18) for k in range(cells)]
    fixed = min(accumulators, 8 * cells)
    # full blocks of both kinds interleaved evenly, partial blocks last: holes only at the very end
    full = {"solar-panel": panels // 4, "accumulator": (accumulators - fixed) // 9}
    order = sorted([(i / full[k], k) for k in full for i in range(full[k])])
    kinds = [k for _, k in order]
    kinds += ["solar-panel"] * (panels % 4 > 0) + ["accumulator"] * ((accumulators - fixed) % 9 > 0)
    left = {"solar-panel": panels, "accumulator": accumulators - fixed}
    for cu, cv in origins:
        plan.add("substation", cu + 8, cv + 8)
    slots = [(cell, block) for cell in range(cells) for block in range(8)]
    for (cell, block), kind in zip(slots, kinds):
        cu, cv = origins[cell]
        bu, bv = _CELL_BLOCKS[block]
        size = 3 if kind == "solar-panel" else 2
        for j in range(6 // size):
            for i in range(6 // size):
                if left[kind] > 0:
                    plan.add(kind, cu + bu + i * size, cv + bv + j * size)
                    left[kind] -= 1
    for n in range(fixed):
        cu, cv = origins[n % cells]
        du, dv = _CELL_FIXED_ACCUMULATORS[n // cells]
        plan.add("accumulator", cu + du, cv + dv)


def _solar_strips(plan: _Layout, panels: int, accumulators: int, pole: str) -> None:
    """Early-game field without substations: 7-wide strips (3 + pole column + 3)."""
    rows: List[Tuple[str, int]] = []  # (kind, how many side by side) per strip row
    rows += [("solar-panel", 2)] * (panels // 2) + ([("solar-panel", 1)] if panels % 2 else [])
    rows += [("accumulator", 2)] * (accumulators // 2) + ([("accumulator", 1)] if accumulators % 2 else [])
    height = sum(3 if k == "solar-panel" else 2 for k, _ in rows)
    strips = max(1, round(math.sqrt(height / 7)))
    target = math.ceil(height / strips)
    u = v = 0
    powered = []
    for kind, n in rows:
        size = 3 if kind == "solar-panel" else 2
        if v + size > target and v > 0:
            u, v = u + 7, 0
        for side in range(n):
            powered.append(plan.add(kind, u + (0 if side == 0 else 4), v))
        v += size
    plan.pole_tiles = {(su + 3, sv) for su in range(0, u + 7, 7) for sv in range(plan.height)}
    plan.width = max(plan.width, u + 4)
    _add_poles(plan, pole, powered)


def solar_field(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    panels: int = 84,
    accumulators: Optional[int] = None,
    pole: str = "substation",
) -> BBox:
    """
    Solar field of exactly `panels` solar panels and `accumulators` accumulators (default: the
    ideal 0.84 per panel; 23.8 panels per MW of daytime average). pole="substation": dense
    18x18 cells around a substation without gaps; pole="medium-electric-pole" (or small) for
    fields before substations: 7-wide strips with a pole column in the middle.
    Returns the BBox of the field.
    """
    _check_count("panels", panels, 0, 5000)
    if accumulators is None:
        accumulators = round(panels * 0.84)
    _check_count("accumulators", accumulators, 0, 5000)
    if panels + accumulators == 0:
        raise ValueError("solar_field needs at least one panel or accumulator.")
    plan = _Layout()
    if pole == "substation":
        _solar_cells(plan, panels, accumulators)
    else:
        _pole_geometry(pole)
        _solar_strips(plan, panels, accumulators, pole)
    return plan.commit(bp, origin, "east")


def _steam_row(plan: _Layout, boilers: int, engines_per_boiler: int, gaps: bool, fuel: bool, inserter: str,
               v0: int, mirrored: bool) -> Tuple[List[int], int, int]:
    """
    One row of steam units in the local frame (water flows east) starting at row v0: an offshore
    pump at u=0, then boilers, each driving a stack of engines. Unmirrored: engines on top, fuel
    inserters below the boilers. gaps=True leaves a pole column after boilers 0, 2, 4, ... (water
    passes in a pipe); gaps=False packs the boilers and puts a pole row between the two engines of
    a stack (one engine: above it). Returns (engine indices, row width, inserter row).
    """
    two = engines_per_boiler == 2
    top = 1 if not gaps and not two else 0  # pole row above a single engine
    pole_row = 0 if gaps or not two else 1  # pole row between two engines
    boiler_v = top + 5 * engines_per_boiler + pole_row
    height = boiler_v + 3  # + boiler (2) + inserter row

    def at(v: int, h: int = 1) -> int:  # row inside the stack, flipped when mirrored
        return v0 + (height - v - h if mirrored else v)

    up, down = (8, 0) if mirrored else (0, 8)
    engines: List[int] = []
    pole_tiles = {(0, at(v)) for v in range(height) if v != boiler_v + 1}  # pump column: wire connectors
    u = 1
    for i in range(boilers):
        plan.add("boiler", u, at(boiler_v, 2), up)
        engines.append(plan.add("steam-engine", u, at(boiler_v - 5, 5), up))
        if two:
            engines.append(plan.add("steam-engine", u, at(boiler_v - 10 - pole_row, 5), up))
        if pole_row:  # steam passes the pole row in a pipe
            plan.add("pipe", u + 1, at(boiler_v - 6))
            pole_tiles |= {(u, at(boiler_v - 6)), (u + 2, at(boiler_v - 6))}
        if top:
            pole_tiles |= {(u + k, at(0)) for k in range(3)}
        if fuel:
            plan.add(inserter, u + 1, at(boiler_v + 2), down)  # picks from the fuel belt
        u += 3
        if gaps and i % 2 == 0:  # pole gap column; water passes in a pipe
            if i < boilers - 1:
                plan.add("pipe", u, at(boiler_v + 1))
            pole_tiles |= {(u, at(v)) for v in range(boiler_v)}
            u += 1
    plan.add("offshore-pump", 0, at(boiler_v + 1), 12)
    plan.pole_tiles = (plan.pole_tiles or set()) | pole_tiles
    plan.height = max(plan.height, v0 + height)  # the pole row may still be empty
    return engines, u, at(boiler_v + 2)


def steam_power(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    boilers: int = 10,
    engines_per_boiler: int = 2,
    belt: Optional[str] = "transport-belt",
    inserter: str = "burner-inserter",
    pole: Optional[str] = "small-electric-pole",
    layout: str = "gapless_row",
    fuel_flow: str = "forward",
    engines_side: str = "left",
    direction: Union[int, str, Direction] = "east",
) -> BBox:
    """
    Steam power plant: offshore pump(s) feed chains of boilers (water flows in `direction`), each
    boiler drives `engines_per_boiler` (1 or 2) steam engines (1.8 MW per boiler); fuel comes on
    a belt behind the boilers (belt=None: no fuel belt and inserters). A yellow belt of coal
    feeds about 33 boilers. Layouts:
      gapless_row:   boilers side by side, poles in a row between the two engines of each stack;
      gapped_row:    a pole column after every second boiler;
      mirrored_pair: two gapless rows back to back around one shared fuel belt (two pumps).
    fuel_flow="forward"|"backward": the fuel belt flows with or against the water.
    engines_side="left"|"right": engines on the left or right of the water flow.
    The offshore pump(s) must end up on a water edge. Returns the BBox of the plant.
    """
    _check_count("boilers", boilers, 1, 100)
    _check_choice("engines_per_boiler", engines_per_boiler, (1, 2))
    _check_choice("layout", layout, ("gapless_row", "gapped_row", "mirrored_pair"))
    _check_choice("fuel_flow", fuel_flow, ("forward", "backward"))
    _check_choice("engines_side", engines_side, ("left", "right"))
    fuel = belt is not None
    if fuel:
        _belt_names(belt)
        _check_type(inserter, ("inserter",))
    plan = _Layout()
    if layout == "mirrored_pair":
        if boilers < 2:
            raise ValueError("layout='mirrored_pair' needs at least 2 boilers.")
        upper, width_a, ins_v = _steam_row(plan, boilers - boilers // 2, engines_per_boiler, False, fuel, inserter, 0, False)
        belt_v = ins_v + 1
        lower, width_b, _ = _steam_row(plan, boilers // 2, engines_per_boiler, False, fuel, inserter, belt_v + 1, True)
        engines, width = upper + lower, max(width_a, width_b)
    else:
        engines, width, ins_v = _steam_row(plan, boilers, engines_per_boiler, layout == "gapped_row", fuel, inserter, 0, False)
        belt_v = ins_v + 1
    if fuel:  # a backward belt runs past the pump column so that it leaves the module
        for u in range(1 if fuel_flow == "forward" else 0, width):
            plan.add(belt, u, belt_v, 4 if fuel_flow == "forward" else 12)
    if pole:
        _add_poles(plan, pole, engines)
    if engines_side == "right":
        plan = plan.flipped()
    return plan.commit(bp, origin, direction)


def balancer(
    bp: Any,
    origin: Union[Tuple[int, int], Dict[str, int]] = (0, 0),
    inputs: int = 4,
    outputs: int = 4,
    belt: str = "transport-belt",
    direction: Union[int, str, Direction] = "north",
    throughput_unlimited: bool = True,
) -> BBox:
    """
    Belt balancer that splits every one of `inputs` belts evenly across `outputs` belts, built
    from Raynquist's balancer book (the community standard) or another verified design.
    Items flow in `direction`: inputs enter on the opposite side, outputs leave on the
    `direction` side. throughput_unlimited=True prefers the TU design (full throughput with any
    inputs/outputs in use); False takes the smaller regular one. Returns the BBox.
    """
    names = _belt_names(belt)
    tier = ("transport-belt", "fast-transport-belt", "express-transport-belt").index(names[0]) + 1
    # prefer the requested TU-ness, then the design made for this belt tier (not a downgrade), then size
    options = [((tu != throughput_unlimited), -t, len(template), (i, o, t, tu))
               for (i, o, t, tu), template in _BALANCER_TEMPLATES.items() if (i, o) == (inputs, outputs) and t <= tier]
    if not options:
        available = sorted({f"{i}->{o}" for i, o, _, _ in _BALANCER_TEMPLATES}, key=lambda s: tuple(map(int, s.split("->"))))
        raise ValueError(f"No {inputs}->{outputs} balancer template for {belt}; available: {', '.join(available)}.")
    template = _BALANCER_TEMPLATES[min(options)[-1]]
    kind_names = {"belt": names[0], "underground": names[1], "splitter": names[2]}
    layout = _Layout()
    for kind, u, v, d, extra in template:
        settings = {"io_type": extra} if isinstance(extra, str) else dict(extra or {})
        layout.add(kind_names[kind], u, v, d, **settings)
    return layout.commit(bp, origin, direction)


# BEGIN BALANCER TEMPLATES (generated by tools/extract_balancers.py)
# (inputs, outputs, min belt tier 1..3, throughput unlimited): [(kind, u, v, direction, io_type)],
# items flow east. Raynquist's balancer book first, verified corpus designs for the remaining pairs.
_BALANCER_TEMPLATES = {
    (1, 2, 1, False): [  # canonical_blueprints id=461 EARLY FACTBOOK | 1 to 2
        ('belt', 0, 0, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (1, 3, 1, False): [  # canonical_blueprints id=462 EARLY FACTBOOK | 1 to 3
        ('belt', 0, 3, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None),
        ('belt', 2, 0, 12, None), ('splitter', 2, 1, 0, None), ('splitter', 2, 2, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 2, 0, None),
        ('belt', 3, 3, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('belt', 5, 1, 4, None),
        ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None),
    ],
    (1, 4, 1, True): [  # Raynquist book 0/4: 1-4 TU balancer
        ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None), ('belt', 3, 0, 4, None),
        ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
    ],
    (1, 5, 1, False): [  # canonical_blueprints id=464 EARLY FACTBOOK | 1 to 5
        ('belt', 0, 4, 4, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 8, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None),
        ('splitter', 2, 0, 12, None), ('splitter', 2, 3, 4, None), ('belt', 3, 0, 12, None), ('belt', 3, 1, 12, None), ('splitter', 3, 2, 0, None),
        ('belt', 3, 3, 0, None), ('belt', 3, 4, 4, None), ('belt', 4, 0, 12, None), ('splitter', 4, 1, 0, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 0, None), ('belt', 5, 0, 4, None), ('splitter', 5, 2, 4, None), ('belt', 6, 0, 4, None), ('splitter', 6, 1, 4, None),
        ('splitter', 6, 3, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None),
        ('belt', 7, 4, 4, None),
    ],
    (1, 6, 1, False): [  # canonical_blueprints id=6705 juan's basic | 1:6 balancer
        ('belt', 0, 4, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 1, 8, None), ('belt', 2, 2, 4, None),
        ('belt', 2, 3, 4, None), ('belt', 2, 4, 0, None), ('belt', 2, 5, 8, None), ('belt', 2, 6, 4, None), ('belt', 2, 7, 4, None),
        ('belt', 2, 8, 0, None), ('belt', 2, 9, 0, None), ('belt', 3, 0, 12, None), ('splitter', 3, 2, 4, None), ('splitter', 3, 6, 4, None),
        ('belt', 3, 9, 12, None), ('belt', 4, 0, 12, None), ('splitter', 4, 1, 4, None), ('splitter', 4, 3, 4, None), ('splitter', 4, 5, 4, None),
        ('splitter', 4, 7, 4, None), ('belt', 4, 9, 12, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 4, None),
        ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None), ('belt', 5, 7, 4, None),
        ('belt', 5, 8, 8, None), ('belt', 5, 9, 12, None),
    ],
    (1, 7, 1, False): [  # canonical_blueprints id=466 EARLY FACTBOOK | 1 to 7
        ('belt', 0, 2, 4, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 0, None), ('belt', 1, 5, 0, None),
        ('belt', 1, 6, 0, None), ('belt', 2, 2, 4, None), ('belt', 2, 3, 4, None), ('splitter', 2, 5, 8, None), ('belt', 2, 6, 12, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 8, None), ('belt', 3, 6, 4, None), ('splitter', 4, 1, 4, None), ('belt', 4, 3, 8, None),
        ('belt', 4, 5, 4, None), ('belt', 4, 6, 4, None), ('splitter', 5, 0, 4, None), ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None),
        ('belt', 5, 6, 4, None),
    ],
    (1, 8, 1, True): [  # Raynquist book 0/8: 1-8 TU balancer
        ('belt', 0, 3, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 8, None), ('belt', 1, 5, 4, None), ('splitter', 2, 1, 4, None),
        ('belt', 2, 3, 8, None), ('belt', 2, 4, 12, None), ('splitter', 2, 5, 4, None), ('splitter', 3, 0, 4, None), ('splitter', 3, 2, 4, None),
        ('splitter', 3, 4, 4, None), ('splitter', 3, 6, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None), ('belt', 4, 2, 4, None),
        ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 6, 4, None), ('belt', 4, 7, 4, None),
    ],
    (2, 1, 1, False): [  # canonical_blueprints id=468 EARLY FACTBOOK | 2 to 1
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (2, 2, 1, False): [  # canonical_blueprints id=469 EARLY FACTBOOK | 2 to 2
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('splitter', 1, 0, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None),
    ],
    (2, 3, 1, True): [  # Raynquist book 1/2: 2-3 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('splitter', 1, 0, 4, None), ('belt', 1, 2, 4, None), ('belt', 1, 3, 0, None),
        ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, {'input_priority': 'left'}), ('belt', 2, 3, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None),
        ('underground', 3, 2, 4, None), ('belt', 3, 3, 12, None), ('belt', 4, 0, 4, None), ('splitter', 4, 1, 4, None), ('belt', 4, 3, 12, None),
        ('splitter', 5, 0, 4, {'input_priority': 'left'}), ('belt', 5, 2, 8, None), ('belt', 5, 3, 12, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 4, None),
        ('underground', 6, 2, 4, 'output'), ('belt', 7, 0, 4, None), ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None),
    ],
    (2, 4, 1, False): [  # Raynquist book 1/3: 2-4 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 2, 0, 4, None), ('splitter', 2, 2, 4, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
    ],
    (2, 4, 1, True): [  # Raynquist book 1/4: 2-4 TU balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 2, 0, 4, {'output_priority': 'left'}), ('splitter', 2, 2, 4, {'output_priority': 'right'}),
        ('belt', 3, 0, 4, None), ('splitter', 3, 1, 4, None), ('belt', 3, 3, 4, None), ('splitter', 4, 0, 4, None), ('splitter', 4, 2, 4, None),
        ('belt', 5, 0, 4, None), ('belt', 5, 1, 4, None), ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None),
    ],
    (2, 5, 1, False): [  # canonical_blueprints id=1268 ALL THE BALANCERS | 2_5_balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 1, 4, None), ('belt', 1, 3, 4, None), ('belt', 1, 4, 0, None),
        ('belt', 1, 5, 0, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('belt', 2, 5, 12, None), ('belt', 3, 1, 4, None),
        ('belt', 3, 2, 4, None), ('splitter', 3, 3, 4, None), ('belt', 3, 5, 12, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None),
        ('underground', 4, 4, 4, None), ('belt', 4, 5, 12, None), ('splitter', 5, 1, 4, None), ('belt', 5, 3, 8, None), ('belt', 5, 4, 8, None),
        ('belt', 5, 5, 12, None), ('splitter', 6, 0, 4, None), ('splitter', 6, 2, 4, None), ('underground', 6, 4, 4, 'output'), ('belt', 7, 0, 4, None),
        ('belt', 7, 1, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 4, None),
    ],
    (2, 6, 1, False): [  # Raynquist book 1/6: 2-6 balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None),
        ('splitter', 1, 3, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None), ('belt', 1, 7, 0, None), ('belt', 2, 0, 12, None),
        ('splitter', 2, 1, 0, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None), ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 2, 0, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None),
        ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('splitter', 4, 4, 4, None),
        ('belt', 4, 6, 4, None), ('belt', 4, 7, 0, None), ('belt', 5, 1, 4, None), ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None),
        ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None),
    ],
    (2, 7, 1, False): [  # canonical_blueprints id=129 Belt balancer compendium with nice icons by tuckjohn37 and Red
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('splitter', 1, 1, 8, None), ('underground', 1, 2, 8, None), ('belt', 1, 3, 4, None),
        ('belt', 1, 4, 4, None), ('underground', 1, 5, 8, 'output'), ('belt', 1, 6, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 2, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 5, 4, None), ('belt', 2, 6, 0, None), ('belt', 3, 0, 12, None), ('splitter', 3, 1, 0, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 2, 0, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 4, None), ('belt', 4, 5, 8, None), ('belt', 4, 6, 4, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 4, None),
        ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None), ('splitter', 5, 6, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 4, None),
    ],
    (2, 8, 1, False): [  # Raynquist book 1/8: 2-8 balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 8, None), ('belt', 1, 5, 4, None),
        ('splitter', 2, 1, 4, None), ('belt', 2, 3, 8, None), ('belt', 2, 4, 12, None), ('splitter', 2, 5, 4, None), ('splitter', 3, 0, 4, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('splitter', 3, 6, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None),
        ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 6, 4, None),
        ('belt', 4, 7, 4, None),
    ],
    (3, 1, 1, True): [  # Raynquist book 2/0: 3-1 TU balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('splitter', 1, 2, 4, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('belt', 2, 3, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 4, 0, 8, None), ('splitter', 4, 2, 4, {'output_priority': 'right'}), ('belt', 5, 0, 12, None),
        ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None), ('belt', 5, 3, 4, None),
    ],
    (3, 2, 1, True): [  # Raynquist book 2/1: 3-2 TU balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('underground', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 1, 3, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, {'output_priority': 'right'}), ('belt', 3, 0, 12, None),
        ('splitter', 3, 1, 4, None), ('belt', 3, 3, 4, None), ('belt', 4, 0, 12, None), ('underground', 4, 1, 4, 'output'), ('belt', 4, 2, 4, None),
        ('belt', 4, 3, 4, None), ('belt', 5, 0, 12, None), ('splitter', 5, 1, 4, {'output_priority': 'right'}), ('belt', 5, 3, 4, None), ('belt', 6, 0, 12, None),
        ('belt', 6, 1, 0, None), ('splitter', 6, 2, 4, None), ('belt', 7, 2, 4, None), ('belt', 7, 3, 4, None),
    ],
    (3, 3, 1, True): [  # Raynquist book 2/3: 3-3 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 0, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 0, None), ('splitter', 3, 0, 4, None),
        ('splitter', 3, 2, 4, None), ('belt', 3, 4, 12, None), ('belt', 4, 0, 4, None), ('splitter', 4, 1, 4, None), ('underground', 4, 3, 4, None),
        ('belt', 4, 4, 12, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 4, None), ('belt', 5, 2, 8, None), ('belt', 5, 3, 8, None),
        ('belt', 5, 4, 12, None), ('belt', 6, 0, 8, None), ('belt', 6, 1, 8, None), ('belt', 6, 2, 4, None), ('underground', 6, 3, 4, 'output'),
        ('underground', 7, 1, 4, 'output'), ('splitter', 7, 2, 4, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 4, None),
    ],
    (3, 4, 1, False): [  # canonical_blueprints id=133 Belt balancer compendium with nice icons by tuckjohn37 and Red
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
    (3, 5, 1, False): [  # canonical_blueprints id=5376 The Essentials | 3 to 5
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
    (3, 5, 2, False): [  # canonical_blueprints id=2536 8/8 | 3/5
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
    (3, 6, 1, False): [  # canonical_blueprints id=135 Belt balancer compendium with nice icons by tuckjohn37 and Red
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 0, None), ('belt', 1, 6, 0, None), ('splitter', 2, 1, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 2, 6, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('splitter', 3, 5, 8, None), ('belt', 3, 6, 12, None), ('belt', 4, 0, 8, None),
        ('underground', 4, 1, 8, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 8, 'output'), ('belt', 4, 6, 4, None), ('splitter', 5, 1, 4, None),
        ('splitter', 5, 3, 4, None), ('splitter', 5, 5, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None),
        ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 4, None),
    ],
    (3, 7, 1, False): [  # Raynquist book 2/9: 3-7 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('splitter', 1, 0, 4, None), ('belt', 1, 2, 8, None),
        ('belt', 1, 3, 4, None), ('belt', 1, 4, 4, None), ('belt', 1, 5, 0, None), ('belt', 1, 6, 0, None), ('belt', 1, 7, 0, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 8, None), ('underground', 2, 2, 8, None), ('splitter', 2, 3, 4, None), ('underground', 2, 5, 8, 'output'),
        ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None), ('belt', 3, 0, 8, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 0, None), ('belt', 3, 4, 8, None), ('belt', 3, 5, 8, None), ('belt', 3, 7, 4, None), ('belt', 4, 1, 4, None),
        ('belt', 4, 2, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None), ('splitter', 4, 6, 0, None), ('belt', 4, 7, 0, None),
        ('splitter', 5, 1, 4, None), ('splitter', 5, 3, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 7, 0, None), ('belt', 6, 1, 4, None),
        ('splitter', 6, 2, 4, None), ('underground', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 7, 12, None), ('splitter', 7, 1, 4, None),
        ('belt', 7, 3, 8, None), ('underground', 7, 4, 8, None), ('belt', 7, 5, 4, None), ('underground', 7, 6, 8, 'output'), ('belt', 7, 7, 12, None),
        ('splitter', 8, 0, 4, None), ('splitter', 8, 2, 4, None), ('underground', 8, 4, 4, 'output'), ('splitter', 8, 5, 4, None), ('belt', 9, 0, 4, None),
        ('belt', 9, 1, 4, None), ('belt', 9, 2, 4, None), ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 4, None),
        ('belt', 9, 6, 4, None),
    ],
    (3, 8, 1, False): [  # canonical_blueprints id=1834 The Complete Belt Series | 3 to 8
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
    (4, 1, 1, True): [  # Raynquist book 3/0: 4-1 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('splitter', 2, 1, 4, None), ('belt', 3, 2, 4, None),
    ],
    (4, 2, 1, False): [  # Raynquist book 3/1: 4-2 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('splitter', 2, 1, 4, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None),
    ],
    (4, 2, 1, True): [  # Raynquist book 3/2: 4-2 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('splitter', 3, 0, 4, {'input_priority': 'left'}),
        ('splitter', 3, 2, 4, {'input_priority': 'right'}), ('splitter', 4, 1, 4, None), ('belt', 5, 1, 4, None), ('belt', 5, 2, 4, None),
    ],
    (4, 3, 1, True): [  # Raynquist book 3/3: 4-3 TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('splitter', 1, 2, 4, {'output_priority': 'right'}),
        ('splitter', 1, 4, 4, {'output_priority': 'left'}), ('splitter', 2, 1, 0, None), ('belt', 2, 2, 0, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('belt', 2, 5, 8, None), ('belt', 2, 6, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 2, 0, 'output'), ('belt', 3, 3, 4, None),
        ('belt', 3, 4, 4, None), ('underground', 3, 5, 0, None), ('belt', 3, 6, 0, None), ('belt', 4, 0, 8, None), ('splitter', 4, 1, 8, {'input_priority': 'right'}),
        ('underground', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('underground', 4, 5, 8, 'output'), ('belt', 4, 6, 4, None),
        ('belt', 5, 0, 8, None), ('belt', 5, 2, 4, None), ('splitter', 5, 3, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 0, None),
        ('belt', 6, 0, 12, None), ('belt', 6, 1, 0, None), ('splitter', 6, 2, 4, {'input_priority': 'left'}), ('splitter', 6, 4, 4, {'input_priority': 'right'}), ('belt', 7, 1, 12, None),
        ('belt', 7, 2, 0, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 4, None), ('belt', 7, 5, 4, None),
    ],
    (4, 4, 1, True): [  # Raynquist book 3/7: 4-4 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('splitter', 1, 0, 4, None),
        ('splitter', 1, 2, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('belt', 3, 0, 4, None),
        ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None),
        ('belt', 4, 2, 4, None), ('belt', 4, 3, 0, None), ('splitter', 5, 1, 4, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 0, None),
        ('belt', 6, 2, 8, None), ('belt', 6, 3, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 4, 'output'), ('underground', 7, 2, 4, 'output'),
        ('belt', 7, 3, 4, None), ('splitter', 8, 0, 4, None), ('splitter', 8, 2, 4, None), ('belt', 9, 0, 4, None), ('belt', 9, 1, 4, None),
        ('belt', 9, 2, 4, None), ('belt', 9, 3, 4, None),
    ],
    (4, 5, 1, False): [  # Raynquist book 3/11: 4-5 balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('splitter', 1, 3, 4, None),
        ('splitter', 1, 5, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('splitter', 2, 2, 0, None), ('belt', 2, 3, 0, None),
        ('splitter', 2, 4, 4, None), ('belt', 2, 6, 8, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None),
        ('underground', 3, 3, 0, 'output'), ('underground', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('underground', 3, 6, 0, None), ('belt', 3, 7, 0, None),
        ('belt', 4, 0, 4, None), ('underground', 4, 1, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None), ('belt', 4, 6, 0, None),
        ('belt', 4, 7, 0, None), ('belt', 5, 0, 8, None), ('splitter', 5, 1, 8, None), ('underground', 5, 2, 8, None), ('splitter', 5, 3, 4, None),
        ('underground', 5, 5, 8, 'output'), ('splitter', 5, 6, 8, None), ('belt', 5, 7, 12, None), ('belt', 6, 0, 8, None), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 4, None), ('belt', 6, 4, 8, None), ('belt', 6, 5, 8, None), ('belt', 6, 7, 4, None), ('belt', 7, 0, 12, None),
        ('underground', 7, 1, 4, 'output'), ('splitter', 7, 2, 4, None), ('underground', 7, 4, 4, 'output'), ('underground', 7, 5, 4, 'output'), ('belt', 7, 6, 4, None),
        ('belt', 7, 7, 0, None), ('belt', 8, 0, 12, None), ('splitter', 8, 1, 4, None), ('splitter', 8, 3, 4, None), ('splitter', 8, 5, 4, None),
        ('belt', 9, 0, 12, None), ('belt', 9, 1, 0, None), ('belt', 9, 2, 4, None), ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None),
        ('belt', 9, 5, 4, None), ('belt', 9, 6, 4, None),
    ],
    (4, 6, 1, False): [  # Raynquist book 3/12: 4-6 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('splitter', 1, 1, 4, None),
        ('splitter', 1, 3, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None), ('splitter', 2, 2, 4, None), ('underground', 2, 4, 4, None),
        ('belt', 3, 0, 8, None), ('underground', 3, 1, 8, None), ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('underground', 3, 4, 8, 'output'),
        ('belt', 3, 5, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None), ('belt', 4, 2, 0, None), ('belt', 4, 3, 0, None),
        ('underground', 4, 4, 4, 'output'), ('belt', 4, 5, 4, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 8, None), ('belt', 5, 2, 8, None),
        ('belt', 5, 3, 4, None), ('splitter', 5, 4, 4, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 12, None), ('splitter', 6, 2, 0, None),
        ('splitter', 6, 3, 4, None), ('belt', 6, 5, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 1, 4, None), ('belt', 7, 3, 0, None),
        ('belt', 7, 4, 4, None), ('belt', 7, 5, 4, None), ('belt', 8, 0, 4, None), ('belt', 8, 1, 4, None), ('underground', 8, 2, 4, 'output'),
        ('splitter', 8, 3, 4, None), ('belt', 8, 5, 4, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 2, 4, None), ('splitter', 9, 4, 4, None),
        ('belt', 10, 0, 4, None), ('belt', 10, 1, 4, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None), ('belt', 10, 4, 4, None),
        ('belt', 10, 5, 4, None),
    ],
    (4, 6, 1, True): [  # Raynquist book 3/13: 4-6 TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 1, 0, 4, None),
        ('underground', 1, 1, 0, 'output'), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('underground', 1, 6, 0, None), ('belt', 1, 7, 0, None),
        ('belt', 2, 0, 8, None), ('belt', 2, 1, 4, None), ('belt', 2, 2, 4, None), ('splitter', 2, 3, 4, None), ('belt', 2, 5, 8, None),
        ('belt', 2, 6, 8, None), ('belt', 2, 7, 12, None), ('splitter', 3, 1, 4, None), ('underground', 3, 3, 4, None), ('belt', 3, 4, 8, None),
        ('belt', 3, 5, 8, None), ('belt', 3, 6, 4, None), ('underground', 4, 1, 4, None), ('belt', 4, 2, 8, None), ('splitter', 4, 3, 8, {'input_priority': 'right'}),
        ('belt', 4, 4, 8, None), ('belt', 4, 5, 4, None), ('underground', 4, 6, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'),
        ('belt', 5, 2, 8, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 4, None), ('underground', 5, 6, 0, None), ('belt', 5, 7, 0, None),
        ('belt', 6, 0, 4, None), ('belt', 6, 2, 12, None), ('splitter', 6, 3, 0, None), ('belt', 6, 4, 0, None), ('belt', 6, 5, 4, None),
        ('belt', 6, 6, 0, None), ('belt', 6, 7, 12, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 4, None), ('splitter', 7, 4, 4, None),
        ('splitter', 7, 6, 12, {'input_priority': 'left'}), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('belt', 8, 2, 4, None), ('underground', 8, 3, 4, 'output'),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 8, None), ('belt', 8, 6, 12, None), ('belt', 8, 7, 12, None), ('belt', 9, 0, 4, None),
        ('splitter', 9, 1, 4, {'input_priority': 'left'}), ('splitter', 9, 3, 4, {'input_priority': 'left'}), ('underground', 9, 5, 4, 'output'), ('underground', 9, 6, 4, 'output'), ('belt', 9, 7, 12, None),
        ('splitter', 10, 0, 4, None), ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('belt', 10, 6, 8, None), ('belt', 10, 7, 12, None),
        ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None),
        ('belt', 11, 5, 4, None),
    ],
    (4, 7, 1, False): [  # Raynquist book 3/14: 4-7 balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 1, 1, 8, None),
        ('underground', 1, 2, 8, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('underground', 1, 7, 8, 'output'), ('belt', 1, 8, 4, None),
        ('belt', 2, 1, 12, None), ('belt', 2, 2, 0, None), ('belt', 2, 3, 0, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 8, None),
        ('belt', 2, 7, 4, None), ('belt', 2, 8, 4, None), ('belt', 3, 0, 8, None), ('belt', 3, 1, 8, None), ('belt', 3, 2, 8, None),
        ('belt', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('splitter', 3, 7, 4, None), ('belt', 4, 0, 12, None),
        ('belt', 4, 1, 4, None), ('underground', 4, 2, 0, 'output'), ('splitter', 4, 3, 4, None), ('underground', 4, 6, 0, None), ('belt', 4, 7, 0, None),
        ('belt', 4, 8, 4, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 4, None), ('splitter', 5, 2, 4, None), ('belt', 5, 4, 8, None),
        ('belt', 5, 5, 8, None), ('belt', 5, 6, 4, None), ('belt', 5, 8, 4, None), ('belt', 6, 0, 12, None), ('splitter', 6, 1, 4, None),
        ('belt', 6, 3, 4, None), ('underground', 6, 4, 4, 'output'), ('underground', 6, 5, 4, 'output'), ('splitter', 6, 6, 4, None), ('belt', 6, 8, 4, None),
        ('belt', 7, 0, 12, None), ('belt', 7, 1, 0, None), ('belt', 7, 2, 4, None), ('splitter', 7, 3, 4, None), ('splitter', 7, 5, 4, None),
        ('splitter', 7, 7, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 4, None), ('belt', 8, 4, 4, None), ('belt', 8, 5, 4, None),
        ('belt', 8, 6, 4, None), ('belt', 8, 7, 4, None), ('belt', 8, 8, 4, None),
    ],
    (4, 8, 1, False): [  # Raynquist book 3/15: 4-8 balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 1, 0, 4, None),
        ('underground', 1, 1, 0, 'output'), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('underground', 1, 6, 0, None), ('belt', 1, 7, 0, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None), ('belt', 2, 2, 0, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('belt', 2, 5, 8, None), ('splitter', 2, 6, 8, None), ('belt', 2, 7, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 8, None),
        ('underground', 3, 2, 8, None), ('splitter', 3, 3, 4, None), ('underground', 3, 5, 8, 'output'), ('belt', 3, 7, 4, None), ('splitter', 4, 0, 4, None),
        ('splitter', 4, 2, 4, None), ('splitter', 4, 4, 4, None), ('splitter', 4, 6, 4, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 4, None),
        ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 6, 4, None),
        ('belt', 5, 7, 4, None),
    ],
    (4, 8, 1, True): [  # Raynquist book 10/0: 4-8 yellow TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None), ('underground', 2, 1, 0, 'output'), ('underground', 2, 2, 4, None), ('belt', 2, 3, 4, None),
        ('belt', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('underground', 2, 6, 0, None), ('belt', 2, 7, 0, None), ('underground', 3, 0, 4, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('belt', 3, 7, 12, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None),
        ('belt', 4, 2, 0, None), ('splitter', 4, 3, 4, None), ('belt', 4, 5, 8, None), ('splitter', 4, 6, 8, None), ('belt', 4, 7, 12, None),
        ('belt', 5, 0, 8, None), ('underground', 5, 1, 8, None), ('underground', 5, 2, 4, 'output'), ('underground', 5, 3, 4, None), ('underground', 5, 4, 4, None),
        ('underground', 5, 5, 8, 'output'), ('belt', 5, 7, 4, None), ('splitter', 6, 1, 4, None), ('underground', 6, 5, 4, 'output'), ('underground', 6, 7, 4, None),
        ('belt', 7, 1, 4, None), ('belt', 7, 2, 8, None), ('underground', 7, 3, 8, None), ('splitter', 7, 4, 4, None), ('underground', 7, 6, 8, 'output'),
        ('belt', 7, 7, 4, None), ('underground', 8, 0, 4, 'output'), ('belt', 8, 1, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 0, None),
        ('belt', 8, 4, 0, None), ('belt', 8, 5, 4, None), ('belt', 8, 6, 4, None), ('belt', 8, 7, 0, None), ('belt', 9, 0, 4, None),
        ('splitter', 9, 1, 4, None), ('underground', 9, 3, 4, 'output'), ('underground', 9, 4, 4, 'output'), ('splitter', 9, 5, 4, None), ('underground', 9, 7, 4, 'output'),
        ('splitter', 10, 0, 4, None), ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('splitter', 10, 6, 4, None), ('belt', 11, 0, 4, None),
        ('belt', 11, 1, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None),
        ('belt', 11, 6, 4, None), ('belt', 11, 7, 4, None),
    ],
    (4, 8, 2, True): [  # Raynquist book 3/16: 4-8 TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None), ('underground', 2, 1, 0, 'output'), ('underground', 2, 2, 4, None), ('belt', 2, 3, 4, None),
        ('belt', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('underground', 2, 6, 0, None), ('belt', 2, 7, 0, None), ('underground', 3, 0, 4, None),
        ('splitter', 3, 2, 4, None), ('splitter', 3, 4, 4, None), ('belt', 3, 7, 12, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None),
        ('belt', 4, 2, 0, None), ('splitter', 4, 3, 4, None), ('belt', 4, 5, 8, None), ('splitter', 4, 6, 8, None), ('belt', 4, 7, 12, None),
        ('belt', 5, 0, 8, None), ('underground', 5, 1, 8, None), ('underground', 5, 2, 4, 'output'), ('underground', 5, 3, 4, None), ('underground', 5, 4, 4, None),
        ('underground', 5, 5, 8, 'output'), ('belt', 5, 7, 4, None), ('splitter', 6, 1, 4, None), ('underground', 6, 5, 4, 'output'), ('underground', 6, 7, 4, None),
        ('belt', 7, 1, 4, None), ('belt', 7, 2, 8, None), ('underground', 7, 3, 8, None), ('splitter', 7, 4, 4, None), ('underground', 7, 6, 8, 'output'),
        ('belt', 7, 7, 4, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 0, None), ('belt', 8, 4, 0, None),
        ('belt', 8, 5, 4, None), ('belt', 8, 6, 4, None), ('belt', 8, 7, 0, None), ('underground', 9, 0, 4, 'output'), ('splitter', 9, 1, 4, None),
        ('underground', 9, 3, 4, 'output'), ('underground', 9, 4, 4, 'output'), ('splitter', 9, 5, 4, None), ('underground', 9, 7, 4, 'output'), ('splitter', 10, 0, 4, None),
        ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('splitter', 10, 6, 4, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None),
        ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None), ('belt', 11, 6, 4, None),
        ('belt', 11, 7, 4, None),
    ],
    (5, 1, 1, True): [  # Raynquist book 4/0: 5-1 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('belt', 1, 4, 8, None), ('belt', 1, 5, 4, None), ('splitter', 2, 1, 4, None),
        ('splitter', 2, 4, 0, None), ('belt', 2, 5, 0, None), ('belt', 3, 1, 4, None), ('splitter', 3, 3, 0, None), ('belt', 3, 5, 0, None),
        ('belt', 4, 1, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 4, 0, None), ('belt', 4, 5, 12, None), ('splitter', 5, 1, 4, {'output_priority': 'left'}),
        ('splitter', 5, 4, 12, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 8, None), ('belt', 6, 3, 8, None), ('belt', 6, 4, 12, None),
    ],
    (5, 2, 1, False): [  # Raynquist book 4/1: 5-2 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 1, 0, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 2, 0, 4, None), ('underground', 2, 1, 4, None),
        ('underground', 2, 2, 4, None), ('belt', 2, 4, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 3, 2, 8, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 4, 0, 8, None), ('underground', 4, 2, 4, 'output'), ('splitter', 4, 3, 4, None),
        ('belt', 5, 0, 12, None), ('underground', 5, 1, 4, 'output'), ('splitter', 5, 2, 4, None), ('belt', 5, 4, 4, None), ('belt', 6, 0, 12, None),
        ('splitter', 6, 1, 4, None), ('splitter', 6, 3, 4, None), ('belt', 7, 0, 12, None), ('belt', 7, 1, 0, None), ('belt', 7, 3, 4, None),
        ('belt', 7, 4, 4, None),
    ],
    (5, 3, 1, False): [  # Raynquist book 4/2: 5-3 balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 1, 2, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 1, 4, None),
        ('belt', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 4, None), ('belt', 3, 0, 12, None),
        ('splitter', 3, 1, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 6, 8, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 12, None), ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None),
        ('splitter', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 12, None), ('splitter', 5, 1, 4, None), ('underground', 5, 3, 4, 'output'),
        ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('belt', 5, 7, 0, None), ('belt', 6, 0, 12, None), ('belt', 6, 1, 0, None),
        ('splitter', 6, 2, 4, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 8, None), ('splitter', 6, 6, 8, None), ('belt', 6, 7, 12, None),
        ('belt', 7, 0, 8, None), ('underground', 7, 1, 8, None), ('belt', 7, 2, 4, None), ('splitter', 7, 3, 4, None), ('underground', 7, 5, 8, 'output'),
        ('belt', 7, 7, 4, None), ('belt', 8, 0, 12, None), ('belt', 8, 1, 0, None), ('belt', 8, 2, 0, None), ('belt', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 4, None), ('belt', 8, 6, 0, None), ('belt', 8, 7, 0, None), ('belt', 9, 3, 4, None),
        ('belt', 9, 4, 4, None), ('belt', 9, 5, 4, None),
    ],
    (5, 4, 1, False): [  # canonical_blueprints id=145 Belt balancer compendium with nice icons by tuckjohn37 and Red
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
    (5, 5, 1, False): [  # Raynquist book 10/2: 5-5 yellow balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None),
        ('underground', 1, 3, 4, None), ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 1, 4, None), ('belt', 2, 2, 0, None),
        ('splitter', 2, 3, 0, None), ('belt', 2, 4, 0, None), ('underground', 2, 5, 4, None), ('underground', 2, 6, 4, None), ('belt', 2, 7, 4, None),
        ('underground', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('belt', 3, 6, 0, None),
        ('belt', 3, 7, 0, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None), ('belt', 4, 2, 0, None), ('underground', 4, 3, 4, 'output'),
        ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 8, None),
        ('splitter', 5, 1, 8, None), ('underground', 5, 2, 8, None), ('splitter', 5, 3, 4, None), ('underground', 5, 5, 8, 'output'), ('splitter', 5, 6, 8, None),
        ('belt', 5, 7, 12, None), ('belt', 6, 0, 8, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 8, None),
        ('belt', 6, 5, 8, None), ('belt', 6, 7, 4, None), ('belt', 7, 0, 12, None), ('underground', 7, 1, 4, 'output'), ('splitter', 7, 2, 4, None),
        ('underground', 7, 5, 4, 'output'), ('underground', 7, 6, 4, 'output'), ('belt', 7, 7, 4, None), ('belt', 8, 0, 12, None), ('splitter', 8, 1, 4, None),
        ('belt', 8, 3, 8, None), ('belt', 8, 4, 4, None), ('splitter', 8, 5, 4, None), ('belt', 8, 7, 4, None), ('belt', 9, 0, 12, None),
        ('belt', 9, 1, 0, None), ('belt', 9, 2, 8, None), ('belt', 9, 3, 4, None), ('splitter', 9, 4, 4, None), ('splitter', 9, 6, 4, None),
        ('belt', 10, 3, 4, None), ('belt', 10, 4, 4, None), ('belt', 10, 5, 4, None), ('belt', 10, 6, 4, None), ('belt', 10, 7, 4, None),
    ],
    (5, 5, 2, False): [  # Raynquist book 4/4: 5-5 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 8, None), ('belt', 1, 6, 4, None), ('underground', 2, 1, 4, None),
        ('splitter', 2, 2, 4, None), ('belt', 2, 4, 8, None), ('belt', 2, 5, 4, None), ('belt', 2, 6, 4, None), ('belt', 3, 0, 4, None),
        ('underground', 3, 1, 0, 'output'), ('belt', 3, 2, 4, None), ('underground', 3, 3, 4, None), ('underground', 3, 4, 0, None), ('belt', 3, 5, 0, None),
        ('belt', 3, 6, 4, None), ('belt', 4, 0, 4, None), ('underground', 4, 1, 4, 'output'), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 0, None), ('splitter', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('splitter', 5, 0, 4, None), ('splitter', 5, 2, 4, None),
        ('belt', 5, 4, 4, None), ('belt', 5, 6, 0, None), ('belt', 6, 0, 4, None), ('splitter', 6, 1, 4, None), ('splitter', 6, 3, 4, None),
        ('belt', 6, 5, 8, None), ('belt', 6, 6, 12, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 4, None), ('splitter', 7, 2, 4, None),
        ('belt', 7, 4, 8, None), ('belt', 7, 5, 12, None), ('belt', 8, 0, 8, None), ('underground', 8, 1, 8, None), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 8, None), ('belt', 8, 4, 4, None), ('underground', 8, 5, 8, 'output'), ('belt', 8, 6, 4, None), ('belt', 9, 2, 4, None),
        ('underground', 9, 3, 4, 'output'), ('belt', 9, 4, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 0, None), ('underground', 10, 1, 4, 'output'),
        ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('belt', 11, 1, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None),
        ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None),
    ],
    (5, 6, 1, False): [  # Raynquist book 10/3: 5-6 yellow balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 8, None), ('belt', 1, 6, 4, None), ('belt', 1, 7, 4, None),
        ('belt', 1, 8, 0, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('belt', 2, 4, 8, None), ('belt', 2, 5, 4, None),
        ('splitter', 2, 6, 4, None), ('belt', 2, 8, 12, None), ('belt', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('underground', 3, 3, 4, None),
        ('splitter', 3, 5, 4, None), ('underground', 3, 7, 4, None), ('belt', 3, 8, 12, None), ('belt', 4, 1, 8, None), ('splitter', 4, 2, 8, None),
        ('belt', 4, 3, 8, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 6, 8, None), ('splitter', 4, 7, 8, None),
        ('belt', 4, 8, 12, None), ('belt', 5, 1, 8, None), ('underground', 5, 3, 8, None), ('splitter', 5, 4, 4, None), ('underground', 5, 6, 8, 'output'),
        ('belt', 5, 8, 4, None), ('belt', 6, 1, 12, None), ('belt', 6, 2, 0, None), ('belt', 6, 3, 0, None), ('belt', 6, 4, 0, None),
        ('underground', 6, 5, 4, None), ('underground', 6, 7, 4, 'output'), ('belt', 6, 8, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'),
        ('underground', 7, 2, 4, 'output'), ('underground', 7, 3, 4, 'output'), ('underground', 7, 5, 0, None), ('splitter', 7, 6, 0, None), ('belt', 7, 7, 0, None),
        ('belt', 7, 8, 4, None), ('belt', 8, 0, 4, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 0, None), ('underground', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 0, None), ('belt', 8, 7, 0, None), ('belt', 8, 8, 0, None), ('belt', 9, 0, 4, None),
        ('belt', 9, 1, 4, None), ('belt', 9, 2, 4, None), ('underground', 9, 3, 0, 'output'), ('belt', 9, 4, 4, None), ('underground', 9, 6, 0, None),
        ('splitter', 9, 7, 0, None), ('splitter', 10, 0, 4, None), ('belt', 10, 2, 4, None), ('underground', 10, 3, 4, 'output'), ('belt', 10, 4, 4, None),
        ('underground', 10, 5, 4, 'output'), ('belt', 10, 6, 4, None), ('belt', 10, 8, 0, None), ('belt', 11, 0, 4, None), ('splitter', 11, 1, 4, None),
        ('splitter', 11, 3, 4, None), ('splitter', 11, 5, 4, None), ('belt', 11, 7, 8, None), ('belt', 11, 8, 12, None), ('splitter', 12, 0, 4, None),
        ('splitter', 12, 2, 4, None), ('splitter', 12, 4, 4, None), ('belt', 12, 6, 8, None), ('belt', 12, 7, 12, None), ('belt', 13, 0, 4, None),
        ('belt', 13, 1, 4, None), ('belt', 13, 2, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None),
    ],
    (5, 6, 2, False): [  # Raynquist book 4/5: 5-6 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 1, 0, None),
        ('belt', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('belt', 2, 5, 4, None), ('belt', 3, 0, 8, None),
        ('underground', 3, 1, 8, None), ('belt', 3, 2, 8, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 0, None),
        ('underground', 3, 6, 8, 'output'), ('belt', 3, 7, 4, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('splitter', 4, 3, 4, None),
        ('belt', 4, 5, 4, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 12, None),
        ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None), ('belt', 6, 0, 12, None), ('splitter', 6, 1, 0, None), ('belt', 6, 2, 0, None),
        ('splitter', 6, 3, 4, None), ('belt', 6, 5, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 0, None), ('belt', 7, 3, 0, None),
        ('splitter', 7, 4, 4, None), ('belt', 7, 6, 4, None), ('belt', 7, 7, 0, None), ('underground', 8, 0, 4, None), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 0, None), ('belt', 8, 4, 0, None), ('underground', 8, 5, 4, None), ('underground', 8, 6, 4, None), ('belt', 8, 7, 12, None),
        ('belt', 9, 0, 4, None), ('underground', 9, 1, 0, 'output'), ('belt', 9, 2, 4, None), ('underground', 9, 3, 4, 'output'), ('underground', 9, 4, 4, 'output'),
        ('underground', 9, 5, 0, None), ('belt', 9, 6, 0, None), ('belt', 9, 7, 12, None), ('belt', 10, 0, 8, None), ('belt', 10, 1, 4, None),
        ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('underground', 10, 5, 4, 'output'), ('splitter', 10, 6, 12, None), ('belt', 11, 1, 4, None),
        ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('underground', 11, 6, 4, 'output'), ('belt', 11, 7, 12, None), ('underground', 12, 0, 4, 'output'),
        ('splitter', 12, 1, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('splitter', 12, 5, 4, None), ('belt', 12, 7, 12, None),
        ('splitter', 13, 0, 4, None), ('splitter', 13, 2, 4, None), ('splitter', 13, 4, 4, None), ('belt', 13, 6, 8, None), ('belt', 13, 7, 12, None),
        ('belt', 14, 0, 4, None), ('belt', 14, 1, 4, None), ('belt', 14, 2, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None),
        ('belt', 14, 5, 4, None),
    ],
    (5, 7, 1, False): [  # Raynquist book 10/4: 5-7 yellow balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 8, None), ('belt', 1, 6, 4, None), ('belt', 1, 7, 4, None),
        ('belt', 1, 8, 0, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('belt', 2, 4, 8, None), ('belt', 2, 5, 4, None),
        ('splitter', 2, 6, 4, None), ('belt', 2, 8, 12, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('underground', 3, 2, 4, None),
        ('underground', 3, 3, 4, None), ('splitter', 3, 5, 4, None), ('underground', 3, 7, 4, None), ('belt', 3, 8, 12, None), ('belt', 4, 0, 8, None),
        ('splitter', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None),
        ('belt', 4, 6, 8, None), ('splitter', 4, 7, 8, None), ('belt', 4, 8, 12, None), ('belt', 5, 0, 8, None), ('underground', 5, 2, 8, None),
        ('splitter', 5, 3, 4, None), ('underground', 5, 6, 8, 'output'), ('belt', 5, 8, 4, None), ('belt', 6, 0, 12, None), ('belt', 6, 1, 0, None),
        ('belt', 6, 2, 0, None), ('belt', 6, 3, 0, None), ('belt', 6, 4, 4, None), ('underground', 6, 7, 4, 'output'), ('belt', 6, 8, 4, None),
        ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('underground', 7, 2, 4, 'output'), ('underground', 7, 4, 4, None), ('underground', 7, 5, 0, None),
        ('splitter', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('belt', 7, 8, 4, None), ('belt', 8, 0, 4, None), ('belt', 8, 1, 4, None),
        ('belt', 8, 2, 0, None), ('underground', 8, 3, 4, 'output'), ('belt', 8, 4, 4, None), ('belt', 8, 5, 0, None), ('belt', 8, 7, 0, None),
        ('belt', 8, 8, 0, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 3, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 0, None),
        ('belt', 9, 7, 0, None), ('belt', 9, 8, 0, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 8, None), ('underground', 10, 2, 8, None),
        ('underground', 10, 3, 4, None), ('splitter', 10, 4, 4, None), ('underground', 10, 6, 8, 'output'), ('belt', 10, 7, 4, None), ('belt', 10, 8, 12, None),
        ('belt', 11, 0, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 0, None), ('belt', 11, 4, 0, None), ('splitter', 11, 5, 4, None),
        ('belt', 11, 7, 4, None), ('belt', 11, 8, 12, None), ('belt', 12, 0, 4, None), ('splitter', 12, 1, 4, None), ('underground', 12, 3, 4, 'output'),
        ('underground', 12, 4, 4, 'output'), ('belt', 12, 5, 4, None), ('splitter', 12, 6, 4, None), ('belt', 12, 8, 12, None), ('splitter', 13, 0, 4, None),
        ('splitter', 13, 2, 4, None), ('splitter', 13, 4, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 8, None), ('belt', 13, 8, 12, None),
        ('belt', 14, 0, 4, None), ('belt', 14, 1, 4, None), ('belt', 14, 2, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None),
        ('belt', 14, 5, 4, None), ('belt', 14, 6, 4, None),
    ],
    (5, 7, 2, False): [  # Raynquist book 4/6: 5-7 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 4, None), ('belt', 1, 7, 0, None),
        ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('belt', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('belt', 2, 7, 12, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('underground', 3, 2, 4, None), ('underground', 3, 3, 4, None), ('splitter', 3, 4, 4, None),
        ('underground', 3, 6, 4, None), ('belt', 3, 7, 12, None), ('belt', 4, 0, 8, None), ('splitter', 4, 1, 8, None), ('belt', 4, 2, 8, None),
        ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 8, None), ('splitter', 4, 6, 8, None), ('belt', 4, 7, 12, None),
        ('belt', 5, 0, 8, None), ('underground', 5, 2, 8, None), ('splitter', 5, 3, 4, None), ('underground', 5, 5, 8, 'output'), ('belt', 5, 7, 4, None),
        ('belt', 6, 0, 12, None), ('belt', 6, 1, 0, None), ('belt', 6, 2, 0, None), ('belt', 6, 3, 0, None), ('underground', 6, 4, 4, None),
        ('underground', 6, 6, 4, 'output'), ('belt', 6, 7, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('underground', 7, 2, 4, 'output'),
        ('underground', 7, 4, 0, None), ('splitter', 7, 5, 0, None), ('belt', 7, 6, 0, None), ('belt', 7, 7, 4, None), ('belt', 8, 0, 4, None),
        ('belt', 8, 1, 4, None), ('belt', 8, 2, 0, None), ('underground', 8, 3, 4, 'output'), ('belt', 8, 4, 4, None), ('belt', 8, 6, 0, None),
        ('belt', 8, 7, 0, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 3, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 0, None),
        ('belt', 9, 7, 0, None), ('belt', 9, 8, 0, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 8, None), ('underground', 10, 2, 8, None),
        ('underground', 10, 3, 4, None), ('splitter', 10, 4, 4, None), ('underground', 10, 6, 8, 'output'), ('belt', 10, 7, 4, None), ('belt', 10, 8, 12, None),
        ('belt', 11, 0, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 0, None), ('belt', 11, 4, 0, None), ('splitter', 11, 5, 4, None),
        ('belt', 11, 7, 4, None), ('belt', 11, 8, 12, None), ('belt', 12, 0, 4, None), ('splitter', 12, 1, 4, None), ('underground', 12, 3, 4, 'output'),
        ('underground', 12, 4, 4, 'output'), ('belt', 12, 5, 4, None), ('splitter', 12, 6, 4, None), ('belt', 12, 8, 12, None), ('splitter', 13, 0, 4, None),
        ('splitter', 13, 2, 4, None), ('splitter', 13, 4, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 8, None), ('belt', 13, 8, 12, None),
        ('belt', 14, 0, 4, None), ('belt', 14, 1, 4, None), ('belt', 14, 2, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None),
        ('belt', 14, 5, 4, None), ('belt', 14, 6, 4, None),
    ],
    (5, 8, 1, False): [  # Raynquist book 10/5: 5-8 yellow balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 1, 0, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 12, None),
        ('belt', 2, 7, 0, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None),
        ('underground', 3, 4, 4, None), ('belt', 3, 5, 8, None), ('splitter', 3, 6, 8, None), ('belt', 3, 7, 12, None), ('belt', 4, 0, 4, None),
        ('belt', 4, 1, 8, None), ('underground', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('underground', 4, 4, 8, 'output'), ('splitter', 4, 5, 8, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 8, None), ('splitter', 5, 1, 8, None), ('underground', 5, 2, 8, None), ('underground', 5, 3, 4, None),
        ('underground', 5, 4, 8, 'output'), ('belt', 5, 6, 4, None), ('belt', 5, 7, 4, None), ('belt', 6, 0, 8, None), ('belt', 6, 2, 8, None),
        ('belt', 6, 3, 4, None), ('underground', 6, 4, 4, 'output'), ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None), ('underground', 6, 7, 4, None),
        ('belt', 7, 0, 12, None), ('underground', 7, 2, 4, 'output'), ('belt', 7, 3, 4, None), ('splitter', 7, 4, 4, None), ('belt', 8, 0, 12, None),
        ('splitter', 8, 1, 0, None), ('belt', 8, 2, 0, None), ('splitter', 8, 3, 4, None), ('splitter', 8, 5, 4, None), ('belt', 9, 0, 4, None),
        ('belt', 9, 2, 0, None), ('belt', 9, 3, 0, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 4, None), ('belt', 9, 6, 8, None),
        ('belt', 9, 7, 4, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 4, None), ('underground', 10, 2, 0, 'output'), ('underground', 10, 3, 4, 'output'),
        ('belt', 10, 4, 4, None), ('belt', 10, 5, 4, None), ('underground', 10, 6, 0, None), ('belt', 10, 7, 0, None), ('belt', 11, 0, 4, None),
        ('splitter', 11, 1, 4, None), ('splitter', 11, 3, 4, None), ('splitter', 11, 5, 4, None), ('underground', 11, 7, 4, 'output'), ('splitter', 12, 0, 4, None),
        ('splitter', 12, 2, 4, None), ('splitter', 12, 4, 4, None), ('splitter', 12, 6, 4, None), ('belt', 13, 0, 4, None), ('belt', 13, 1, 4, None),
        ('belt', 13, 2, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None),
        ('belt', 13, 7, 4, None),
    ],
    (5, 8, 2, False): [  # Raynquist book 4/7: 5-8 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 1, 0, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('belt', 1, 5, 4, None), ('belt', 1, 6, 0, None),
        ('belt', 2, 0, 4, None), ('belt', 2, 1, 4, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 12, None),
        ('belt', 2, 7, 0, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('underground', 3, 2, 4, None), ('underground', 3, 3, 4, None),
        ('underground', 3, 4, 4, None), ('belt', 3, 5, 8, None), ('splitter', 3, 6, 8, None), ('belt', 3, 7, 12, None), ('belt', 4, 0, 4, None),
        ('belt', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('splitter', 4, 3, 8, None), ('belt', 4, 4, 8, None), ('belt', 4, 5, 8, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('splitter', 5, 2, 8, None), ('belt', 5, 4, 8, None),
        ('belt', 5, 5, 4, None), ('underground', 5, 7, 4, None), ('belt', 6, 1, 8, None), ('belt', 6, 3, 4, None), ('underground', 6, 4, 4, 'output'),
        ('belt', 6, 5, 4, None), ('belt', 7, 0, 8, None), ('belt', 7, 1, 12, None), ('underground', 7, 2, 4, 'output'), ('belt', 7, 3, 4, None),
        ('splitter', 7, 4, 4, None), ('belt', 8, 0, 12, None), ('splitter', 8, 1, 0, None), ('belt', 8, 2, 0, None), ('splitter', 8, 3, 4, None),
        ('splitter', 8, 5, 4, None), ('belt', 9, 0, 4, None), ('belt', 9, 2, 0, None), ('belt', 9, 3, 0, None), ('belt', 9, 4, 4, None),
        ('belt', 9, 5, 4, None), ('belt', 9, 6, 8, None), ('belt', 9, 7, 4, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 4, None),
        ('underground', 10, 2, 0, 'output'), ('underground', 10, 3, 4, 'output'), ('belt', 10, 4, 4, None), ('belt', 10, 5, 4, None), ('underground', 10, 6, 0, None),
        ('belt', 10, 7, 0, None), ('belt', 11, 0, 4, None), ('splitter', 11, 1, 4, None), ('splitter', 11, 3, 4, None), ('splitter', 11, 5, 4, None),
        ('underground', 11, 7, 4, 'output'), ('splitter', 12, 0, 4, None), ('splitter', 12, 2, 4, None), ('splitter', 12, 4, 4, None), ('splitter', 12, 6, 4, None),
        ('belt', 13, 0, 4, None), ('belt', 13, 1, 4, None), ('belt', 13, 2, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None),
        ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 4, None),
    ],
    (6, 1, 1, True): [  # Raynquist book 5/0: 6-1 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None),
        ('splitter', 2, 3, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None),
        ('belt', 4, 0, 8, None), ('splitter', 4, 2, 4, {'output_priority': 'right'}), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None),
        ('belt', 5, 3, 4, None),
    ],
    (6, 2, 1, False): [  # Raynquist book 5/1: 6-2 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('splitter', 1, 4, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}), ('belt', 2, 0, 8, None),
        ('splitter', 2, 1, 8, None), ('underground', 2, 2, 8, None), ('splitter', 2, 3, 4, None), ('underground', 2, 5, 8, 'output'), ('belt', 2, 6, 4, None),
        ('belt', 3, 0, 8, None), ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('belt', 3, 5, 4, None),
        ('belt', 3, 6, 4, None), ('belt', 4, 0, 12, None), ('splitter', 4, 2, 4, None), ('splitter', 4, 5, 4, None), ('belt', 5, 0, 12, None),
        ('underground', 5, 1, 0, 'output'), ('belt', 5, 2, 4, None), ('belt', 5, 3, 4, None), ('underground', 5, 4, 0, None), ('belt', 5, 5, 0, None),
    ],
    (6, 3, 1, False): [  # Raynquist book 5/2: 6-3 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('splitter', 1, 4, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}), ('belt', 2, 0, 8, None),
        ('splitter', 2, 1, 8, None), ('underground', 2, 2, 8, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None), ('underground', 2, 5, 8, 'output'),
        ('belt', 2, 6, 4, None), ('belt', 3, 0, 8, None), ('belt', 3, 2, 4, None), ('splitter', 3, 3, 4, None), ('belt', 3, 5, 4, None),
        ('belt', 3, 6, 0, None), ('belt', 4, 0, 12, None), ('belt', 4, 1, 0, None), ('splitter', 4, 2, 4, None), ('splitter', 4, 4, 4, None),
        ('belt', 5, 1, 12, None), ('belt', 5, 2, 0, None), ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 4, None),
    ],
    (6, 3, 2, True): [  # Raynquist book 5/3: 6-3 TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('belt', 1, 0, 8, None), ('underground', 1, 1, 8, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('splitter', 1, 6, 4, None), ('underground', 1, 8, 8, 'output'), ('belt', 1, 9, 4, None), ('belt', 2, 0, 12, None), ('splitter', 2, 1, 0, None),
        ('belt', 2, 2, 0, None), ('splitter', 2, 3, 4, None), ('underground', 2, 5, 4, None), ('underground', 2, 6, 4, None), ('belt', 2, 7, 8, None),
        ('belt', 2, 8, 4, None), ('belt', 2, 9, 4, None), ('belt', 3, 0, 4, None), ('splitter', 3, 2, 0, None), ('belt', 3, 3, 0, None),
        ('belt', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('belt', 3, 6, 0, None), ('splitter', 3, 7, 0, None), ('belt', 3, 8, 0, None),
        ('belt', 3, 9, 4, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None), ('underground', 4, 3, 0, 'output'), ('splitter', 4, 4, 4, None),
        ('underground', 4, 6, 0, None), ('belt', 4, 8, 0, None), ('belt', 4, 9, 4, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 8, None),
        ('underground', 5, 2, 8, None), ('underground', 5, 4, 4, None), ('underground', 5, 5, 4, 'output'), ('underground', 5, 6, 4, 'output'), ('underground', 5, 7, 8, 'output'),
        ('belt', 5, 8, 12, None), ('belt', 5, 9, 4, None), ('belt', 6, 0, 8, None), ('belt', 6, 1, 8, None), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 4, None), ('belt', 6, 4, 0, None), ('splitter', 6, 5, 4, None), ('belt', 6, 7, 8, None), ('belt', 6, 8, 4, None),
        ('belt', 6, 9, 4, None), ('splitter', 7, 2, 4, None), ('belt', 7, 4, 12, None), ('belt', 7, 5, 0, None), ('belt', 7, 6, 8, None),
        ('belt', 7, 7, 12, None), ('splitter', 7, 8, 4, None), ('belt', 8, 3, 4, None), ('underground', 8, 4, 4, 'output'), ('belt', 8, 5, 4, None),
        ('belt', 8, 6, 0, None), ('splitter', 8, 7, 0, None), ('belt', 8, 8, 0, None), ('belt', 9, 1, 4, None), ('underground', 9, 2, 0, 'output'),
        ('belt', 9, 3, 4, None), ('splitter', 9, 4, 4, None), ('underground', 9, 6, 0, None), ('belt', 9, 8, 0, None), ('belt', 10, 1, 8, None),
        ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('belt', 10, 5, 8, None), ('belt', 10, 6, 8, None), ('belt', 10, 7, 8, None),
        ('belt', 10, 8, 12, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None),
    ],
    (6, 4, 1, False): [  # Raynquist book 5/4: 6-4 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None),
        ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('belt', 2, 5, 4, None), ('belt', 3, 0, 4, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('splitter', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('belt', 4, 0, 8, None),
        ('belt', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('belt', 4, 5, 0, None), ('underground', 5, 1, 4, None), ('underground', 5, 2, 4, None),
        ('belt', 5, 3, 8, None), ('belt', 5, 4, 8, None), ('belt', 5, 5, 12, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 0, None),
        ('belt', 6, 2, 0, None), ('belt', 6, 3, 0, None), ('belt', 6, 4, 0, None), ('belt', 6, 5, 0, None), ('belt', 7, 0, 4, None),
        ('underground', 7, 1, 4, 'output'), ('underground', 7, 2, 4, 'output'), ('underground', 7, 3, 4, 'output'), ('underground', 7, 4, 4, 'output'), ('belt', 7, 5, 12, None),
        ('belt', 8, 0, 4, None), ('splitter', 8, 1, 4, None), ('splitter', 8, 3, 4, None), ('belt', 8, 5, 12, None), ('splitter', 9, 0, 4, None),
        ('splitter', 9, 2, 4, None), ('belt', 9, 4, 8, None), ('belt', 9, 5, 12, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 4, None),
        ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None),
    ],
    (6, 4, 1, True): [  # Raynquist book 10/6: 6-4 yellow TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 2, 4, None),
        ('splitter', 2, 3, 4, {'output_priority': 'right'}), ('splitter', 2, 5, 4, {'output_priority': 'right'}), ('belt', 2, 7, 4, None), ('belt', 3, 1, 8, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 12, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 4, None), ('belt', 4, 5, 4, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 12, None), ('splitter', 5, 1, 0, {'output_priority': 'right'}), ('belt', 5, 2, 0, None), ('splitter', 5, 4, 0, None),
        ('belt', 5, 5, 0, None), ('belt', 5, 7, 4, None), ('belt', 6, 0, 4, None), ('underground', 6, 2, 0, 'output'), ('belt', 6, 3, 4, None),
        ('belt', 6, 5, 0, None), ('underground', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 8, None), ('belt', 7, 1, 4, None),
        ('underground', 7, 2, 4, 'output'), ('belt', 7, 3, 8, None), ('splitter', 7, 4, 8, {'output_priority': 'left'}), ('belt', 7, 5, 12, None), ('belt', 8, 1, 4, None),
        ('belt', 8, 2, 8, None), ('belt', 8, 3, 8, None), ('belt', 8, 5, 4, None), ('underground', 8, 6, 4, 'output'), ('belt', 9, 1, 8, None),
        ('belt', 9, 2, 8, None), ('belt', 9, 3, 4, None), ('underground', 9, 4, 4, 'output'), ('splitter', 9, 5, 4, None), ('belt', 10, 0, 8, None),
        ('belt', 10, 1, 8, None), ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('belt', 10, 5, 4, None), ('belt', 10, 6, 8, None),
        ('belt', 10, 7, 4, None), ('belt', 11, 0, 12, None), ('underground', 11, 1, 0, 'output'), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None),
        ('underground', 11, 6, 0, None), ('belt', 11, 7, 0, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None),
        ('belt', 12, 5, 4, None),
    ],
    (6, 4, 2, True): [  # Raynquist book 5/5: 6-4 TU balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 2, 4, None),
        ('splitter', 2, 3, 4, {'output_priority': 'right'}), ('splitter', 2, 5, 4, {'output_priority': 'right'}), ('belt', 2, 7, 4, None), ('belt', 3, 1, 8, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 12, None), ('splitter', 4, 2, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 7, 4, None),
        ('belt', 5, 0, 12, None), ('splitter', 5, 1, 0, {'output_priority': 'right'}), ('belt', 5, 2, 0, None), ('splitter', 5, 4, 0, None), ('belt', 5, 5, 0, None),
        ('belt', 5, 7, 4, None), ('belt', 6, 0, 4, None), ('underground', 6, 2, 0, 'output'), ('belt', 6, 3, 4, None), ('belt', 6, 5, 0, None),
        ('underground', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 8, None), ('belt', 7, 1, 4, None), ('underground', 7, 2, 4, 'output'),
        ('belt', 7, 3, 8, None), ('splitter', 7, 4, 8, {'output_priority': 'left'}), ('belt', 7, 5, 12, None), ('belt', 8, 1, 4, None), ('belt', 8, 2, 8, None),
        ('belt', 8, 3, 8, None), ('belt', 8, 5, 4, None), ('underground', 8, 6, 4, 'output'), ('belt', 9, 1, 8, None), ('belt', 9, 2, 8, None),
        ('belt', 9, 3, 4, None), ('underground', 9, 4, 4, 'output'), ('splitter', 9, 5, 4, None), ('belt', 10, 0, 8, None), ('belt', 10, 1, 8, None),
        ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('belt', 10, 5, 4, None), ('belt', 10, 6, 8, None), ('belt', 10, 7, 4, None),
        ('belt', 11, 0, 12, None), ('underground', 11, 1, 0, 'output'), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('underground', 11, 6, 0, None),
        ('belt', 11, 7, 0, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('belt', 12, 5, 4, None),
    ],
    (6, 5, 1, False): [  # Raynquist book 10/7: 6-5 yellow balancer
        ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None),
        ('belt', 0, 8, 4, None), ('belt', 1, 1, 8, None), ('belt', 1, 2, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None),
        ('splitter', 1, 7, 4, None), ('belt', 2, 0, 8, None), ('belt', 2, 1, 12, None), ('splitter', 2, 2, 4, None), ('splitter', 2, 4, 4, None),
        ('splitter', 2, 6, 4, None), ('belt', 2, 8, 4, None), ('belt', 3, 0, 12, None), ('splitter', 3, 1, 0, None), ('belt', 3, 2, 0, None),
        ('underground', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('belt', 3, 6, 4, None), ('splitter', 3, 7, 4, None),
        ('underground', 4, 2, 0, 'output'), ('belt', 4, 4, 4, None), ('underground', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 4, None),
        ('belt', 4, 8, 4, None), ('belt', 5, 0, 4, None), ('belt', 5, 1, 0, None), ('splitter', 5, 2, 0, None), ('belt', 5, 3, 0, None),
        ('belt', 5, 4, 0, None), ('underground', 5, 5, 4, 'output'), ('belt', 5, 6, 4, None), ('belt', 5, 7, 0, None), ('belt', 5, 8, 4, None),
        ('belt', 6, 0, 4, None), ('belt', 6, 1, 4, None), ('underground', 6, 3, 0, 'output'), ('underground', 6, 5, 4, None), ('underground', 6, 6, 4, None),
        ('underground', 6, 7, 0, None), ('belt', 6, 8, 0, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 4, None), ('underground', 7, 3, 4, 'output'),
        ('belt', 7, 4, 4, None), ('belt', 7, 5, 0, None), ('belt', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('belt', 8, 0, 8, None),
        ('splitter', 8, 1, 8, None), ('underground', 8, 2, 8, None), ('splitter', 8, 3, 4, None), ('underground', 8, 5, 8, 'output'), ('splitter', 8, 6, 8, None),
        ('belt', 8, 7, 12, None), ('belt', 9, 0, 8, None), ('belt', 9, 2, 4, None), ('belt', 9, 3, 4, None), ('belt', 9, 4, 8, None),
        ('belt', 9, 5, 8, None), ('belt', 9, 7, 4, None), ('belt', 10, 0, 12, None), ('underground', 10, 1, 4, 'output'), ('splitter', 10, 2, 4, None),
        ('underground', 10, 5, 4, 'output'), ('underground', 10, 6, 4, 'output'), ('belt', 10, 7, 4, None), ('belt', 11, 0, 12, None), ('splitter', 11, 1, 4, None),
        ('belt', 11, 3, 8, None), ('belt', 11, 4, 4, None), ('splitter', 11, 5, 4, None), ('belt', 11, 7, 4, None), ('belt', 12, 0, 12, None),
        ('belt', 12, 1, 0, None), ('belt', 12, 2, 8, None), ('belt', 12, 3, 4, None), ('splitter', 12, 4, 4, None), ('splitter', 12, 6, 4, None),
        ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 4, None),
    ],
    (6, 5, 2, False): [  # Raynquist book 5/6: 6-5 balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('splitter', 1, 6, 4, None), ('belt', 2, 0, 12, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('splitter', 2, 5, 4, None), ('underground', 2, 7, 4, None), ('belt', 3, 0, 12, None), ('underground', 3, 1, 4, None), ('splitter', 3, 2, 4, None),
        ('splitter', 3, 4, 4, None), ('belt', 3, 6, 4, None), ('splitter', 4, 0, 12, None), ('underground', 4, 2, 4, None), ('splitter', 4, 3, 4, None),
        ('belt', 4, 5, 4, None), ('belt', 4, 6, 8, None), ('belt', 4, 7, 4, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 12, None),
        ('underground', 5, 2, 0, 'output'), ('underground', 5, 3, 4, None), ('underground', 5, 4, 4, None), ('belt', 5, 5, 4, None), ('underground', 5, 6, 0, None),
        ('belt', 5, 7, 0, None), ('belt', 6, 0, 12, None), ('underground', 6, 1, 4, 'output'), ('underground', 6, 2, 4, 'output'), ('belt', 6, 3, 4, None),
        ('belt', 6, 4, 0, None), ('belt', 6, 5, 0, None), ('underground', 6, 7, 4, 'output'), ('belt', 7, 0, 12, None), ('belt', 7, 1, 0, None),
        ('splitter', 7, 2, 4, None), ('belt', 7, 4, 4, None), ('belt', 7, 5, 0, None), ('splitter', 7, 6, 0, None), ('belt', 7, 7, 0, None),
        ('belt', 8, 2, 4, None), ('splitter', 8, 3, 4, None), ('belt', 8, 5, 4, None), ('belt', 8, 7, 0, None), ('splitter', 9, 2, 4, None),
        ('splitter', 9, 4, 4, None), ('belt', 9, 6, 8, None), ('belt', 9, 7, 12, None), ('belt', 10, 0, 4, None), ('belt', 10, 1, 0, None),
        ('belt', 10, 2, 0, None), ('splitter', 10, 3, 4, None), ('belt', 10, 5, 8, None), ('belt', 10, 6, 12, None), ('belt', 11, 0, 8, None),
        ('underground', 11, 1, 8, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 0, None), ('belt', 11, 4, 8, None), ('belt', 11, 5, 4, None),
        ('underground', 11, 6, 8, 'output'), ('belt', 11, 7, 4, None), ('belt', 12, 2, 4, None), ('underground', 12, 3, 4, 'output'), ('underground', 12, 4, 4, 'output'),
        ('belt', 12, 5, 4, None), ('belt', 12, 6, 4, None), ('belt', 12, 7, 0, None), ('splitter', 13, 2, 4, None), ('splitter', 13, 4, 4, None),
        ('belt', 13, 6, 4, None), ('belt', 14, 2, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None), ('belt', 14, 5, 4, None),
        ('belt', 14, 6, 4, None),
    ],
    (6, 6, 1, False): [  # Raynquist book 10/8: 6-6 yellow balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 0, 6, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('belt', 2, 1, 4, None),
        ('underground', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 8, None), ('belt', 2, 7, 4, None),
        ('belt', 3, 1, 4, None), ('belt', 3, 2, 4, None), ('underground', 3, 3, 0, 'output'), ('underground', 3, 4, 4, None), ('underground', 3, 5, 4, None),
        ('underground', 3, 6, 0, None), ('belt', 3, 7, 0, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None), ('belt', 4, 2, 4, None),
        ('underground', 4, 3, 4, 'output'), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None),
        ('belt', 5, 0, 8, None), ('underground', 5, 1, 8, None), ('splitter', 5, 2, 4, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 8, 'output'),
        ('splitter', 5, 6, 8, None), ('belt', 5, 7, 12, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 0, None), ('splitter', 6, 3, 4, None),
        ('belt', 6, 5, 8, None), ('belt', 6, 7, 4, None), ('belt', 7, 1, 4, None), ('underground', 7, 2, 4, 'output'), ('belt', 7, 3, 4, None),
        ('belt', 7, 4, 8, None), ('belt', 7, 5, 12, None), ('belt', 7, 6, 4, None), ('belt', 7, 7, 0, None), ('belt', 8, 1, 4, None),
        ('splitter', 8, 2, 4, None), ('underground', 8, 4, 4, 'output'), ('underground', 8, 5, 4, 'output'), ('belt', 8, 6, 4, None), ('splitter', 9, 1, 4, None),
        ('splitter', 9, 3, 4, None), ('splitter', 9, 5, 4, None), ('belt', 10, 1, 4, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None),
        ('belt', 10, 4, 4, None), ('belt', 10, 5, 4, None), ('belt', 10, 6, 4, None),
    ],
    (6, 6, 2, False): [  # Raynquist book 5/7: 6-6 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None),
        ('underground', 2, 1, 4, None), ('underground', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('belt', 2, 5, 4, None),
        ('belt', 3, 0, 8, None), ('belt', 3, 1, 8, None), ('belt', 3, 2, 8, None), ('belt', 3, 3, 8, None), ('belt', 3, 4, 4, None),
        ('belt', 3, 5, 4, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('splitter', 4, 4, 4, None),
        ('belt', 5, 0, 8, None), ('belt', 5, 1, 12, None), ('underground', 5, 2, 4, 'output'), ('splitter', 5, 3, 4, None), ('underground', 5, 5, 4, None),
        ('belt', 6, 0, 12, None), ('splitter', 6, 1, 0, None), ('belt', 6, 2, 0, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 8, None),
        ('belt', 6, 5, 4, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 0, None), ('belt', 7, 3, 0, None), ('underground', 7, 4, 4, 'output'),
        ('belt', 7, 5, 4, None), ('belt', 8, 0, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 0, None), ('belt', 8, 4, 0, None),
        ('belt', 8, 5, 4, None), ('belt', 9, 0, 4, None), ('underground', 9, 1, 4, 'output'), ('belt', 9, 2, 4, None), ('underground', 9, 3, 4, 'output'),
        ('belt', 9, 4, 4, None), ('belt', 9, 5, 0, None), ('belt', 10, 0, 4, None), ('splitter', 10, 1, 4, None), ('splitter', 10, 3, 4, None),
        ('underground', 10, 5, 4, 'output'), ('splitter', 11, 0, 4, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('belt', 12, 0, 4, None),
        ('belt', 12, 1, 4, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('belt', 12, 5, 4, None),
    ],
    (6, 7, 1, False): [  # Raynquist book 5/10: 6-7 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 0, 6, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('belt', 2, 0, 4, None),
        ('belt', 2, 1, 0, None), ('underground', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('splitter', 2, 4, 4, None), ('belt', 2, 6, 8, None),
        ('belt', 2, 7, 8, None), ('belt', 2, 8, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None), ('belt', 3, 2, 8, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None), ('belt', 3, 6, 8, None), ('belt', 3, 7, 4, None),
        ('belt', 3, 8, 4, None), ('belt', 4, 0, 8, None), ('underground', 4, 2, 8, None), ('splitter', 4, 3, 4, None), ('underground', 4, 5, 8, 'output'),
        ('belt', 4, 6, 4, None), ('belt', 4, 7, 4, None), ('belt', 4, 8, 4, None), ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None),
        ('belt', 5, 2, 0, None), ('belt', 5, 3, 0, None), ('belt', 5, 4, 8, None), ('belt', 5, 5, 4, None), ('splitter', 5, 6, 4, None),
        ('belt', 5, 8, 4, None), ('underground', 6, 2, 4, 'output'), ('underground', 6, 3, 4, 'output'), ('underground', 6, 5, 4, None), ('belt', 6, 6, 4, None),
        ('underground', 6, 7, 4, None), ('belt', 6, 8, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('splitter', 7, 2, 4, None),
        ('underground', 7, 4, 0, None), ('splitter', 7, 5, 0, None), ('belt', 7, 6, 0, None), ('belt', 7, 8, 4, None), ('belt', 8, 0, 4, None),
        ('belt', 8, 1, 4, None), ('belt', 8, 2, 0, None), ('underground', 8, 3, 4, None), ('belt', 8, 4, 4, None), ('belt', 8, 6, 0, None),
        ('belt', 8, 7, 0, None), ('belt', 8, 8, 0, None), ('belt', 9, 0, 4, None), ('belt', 9, 1, 4, None), ('belt', 9, 2, 4, None),
        ('underground', 9, 3, 0, 'output'), ('belt', 9, 4, 4, None), ('underground', 9, 6, 0, None), ('splitter', 9, 7, 0, None), ('splitter', 10, 0, 4, None),
        ('belt', 10, 2, 4, None), ('underground', 10, 3, 4, 'output'), ('belt', 10, 4, 4, None), ('underground', 10, 5, 4, 'output'), ('belt', 10, 6, 4, None),
        ('belt', 10, 8, 0, None), ('belt', 11, 0, 4, None), ('splitter', 11, 1, 4, None), ('splitter', 11, 3, 4, None), ('splitter', 11, 5, 4, None),
        ('underground', 11, 7, 4, 'output'), ('belt', 11, 8, 12, None), ('splitter', 12, 0, 4, None), ('splitter', 12, 2, 4, None), ('splitter', 12, 4, 4, None),
        ('splitter', 12, 6, 4, None), ('belt', 12, 8, 12, None), ('belt', 13, 0, 4, None), ('belt', 13, 1, 4, None), ('belt', 13, 2, 4, None),
        ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 8, None),
        ('belt', 13, 8, 12, None),
    ],
    (6, 8, 1, False): [  # Raynquist book 10/9: 6-8 yellow balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None), ('belt', 2, 0, 4, None),
        ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None), ('underground', 2, 5, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, None),
        ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 8, None), ('belt', 3, 5, 8, None), ('belt', 3, 6, 8, None),
        ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('splitter', 4, 1, 8, None), ('underground', 4, 2, 8, None), ('belt', 4, 3, 8, None),
        ('belt', 4, 4, 4, None), ('underground', 4, 5, 8, 'output'), ('belt', 4, 6, 4, None), ('belt', 4, 7, 4, None), ('belt', 5, 0, 8, None),
        ('belt', 5, 2, 8, None), ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('splitter', 5, 6, 4, None), ('belt', 6, 0, 12, None),
        ('belt', 6, 1, 0, None), ('belt', 6, 2, 0, None), ('splitter', 6, 3, 4, None), ('underground', 6, 5, 4, 'output'), ('underground', 6, 6, 4, None),
        ('underground', 6, 7, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('belt', 7, 2, 12, None), ('belt', 7, 3, 0, None),
        ('splitter', 7, 4, 4, None), ('underground', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'),
        ('underground', 8, 2, 4, 'output'), ('belt', 8, 4, 4, None), ('belt', 8, 5, 8, None), ('belt', 8, 6, 8, None), ('belt', 8, 7, 12, None),
        ('splitter', 9, 0, 4, None), ('belt', 9, 2, 8, None), ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None), ('underground', 9, 6, 4, 'output'),
        ('belt', 10, 0, 4, None), ('splitter', 10, 1, 4, None), ('splitter', 10, 3, 4, None), ('splitter', 10, 5, 4, None), ('underground', 10, 7, 4, 'output'),
        ('splitter', 11, 0, 4, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('splitter', 11, 6, 4, None), ('belt', 12, 0, 4, None),
        ('belt', 12, 1, 4, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('belt', 12, 5, 4, None),
        ('belt', 12, 6, 4, None), ('belt', 12, 7, 4, None),
    ],
    (6, 8, 2, False): [  # Raynquist book 5/11: 6-8 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 0, 6, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('underground', 2, 1, 4, None),
        ('splitter', 2, 2, 4, None), ('underground', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('belt', 2, 6, 8, None), ('belt', 2, 7, 4, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('belt', 3, 2, 0, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None),
        ('belt', 3, 5, 0, None), ('splitter', 3, 6, 0, None), ('belt', 3, 7, 0, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None),
        ('underground', 4, 2, 0, 'output'), ('splitter', 4, 3, 4, None), ('underground', 4, 5, 0, None), ('belt', 4, 7, 0, None), ('splitter', 5, 0, 4, None),
        ('underground', 5, 3, 4, None), ('belt', 5, 4, 8, None), ('belt', 5, 5, 8, None), ('belt', 5, 6, 8, None), ('belt', 5, 7, 12, None),
        ('underground', 6, 0, 4, None), ('belt', 6, 1, 8, None), ('splitter', 6, 2, 8, None), ('underground', 6, 3, 8, None), ('underground', 6, 4, 4, 'output'),
        ('underground', 6, 5, 4, 'output'), ('underground', 6, 6, 8, 'output'), ('belt', 6, 7, 4, None), ('belt', 7, 0, 8, None), ('belt', 7, 1, 8, None),
        ('belt', 7, 3, 4, None), ('splitter', 7, 4, 4, None), ('belt', 7, 7, 4, None), ('belt', 8, 0, 12, None), ('underground', 8, 1, 4, 'output'),
        ('splitter', 8, 3, 4, None), ('belt', 8, 5, 8, None), ('belt', 8, 6, 4, None), ('belt', 8, 7, 4, None), ('belt', 9, 0, 12, None),
        ('belt', 9, 1, 0, None), ('belt', 9, 2, 4, None), ('belt', 9, 3, 0, None), ('belt', 9, 4, 4, None), ('splitter', 9, 6, 4, None),
        ('underground', 10, 0, 4, 'output'), ('splitter', 10, 1, 4, None), ('underground', 10, 3, 4, 'output'), ('belt', 10, 4, 4, None), ('splitter', 10, 5, 4, None),
        ('belt', 10, 7, 4, None), ('splitter', 11, 0, 4, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('splitter', 11, 6, 4, None),
        ('belt', 12, 0, 4, None), ('belt', 12, 1, 4, None), ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None),
        ('belt', 12, 5, 4, None), ('belt', 12, 6, 4, None), ('belt', 12, 7, 4, None),
    ],
    (7, 1, 1, True): [  # Raynquist book 6/0: 7-1 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('belt', 1, 6, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 4, 0, None), ('belt', 2, 5, 0, None), ('belt', 2, 6, 4, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('splitter', 3, 5, 0, None), ('belt', 3, 6, 0, None), ('splitter', 4, 2, 4, {'output_priority': 'left'}),
        ('belt', 4, 6, 0, None), ('belt', 5, 2, 4, None), ('belt', 5, 3, 8, None), ('belt', 5, 4, 8, None), ('belt', 5, 5, 8, None),
        ('belt', 5, 6, 12, None),
    ],
    (7, 2, 1, False): [  # Raynquist book 6/1: 7-2 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 1, 0, 4, None), ('splitter', 1, 1, 4, None), ('splitter', 1, 3, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}),
        ('splitter', 1, 5, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}), ('belt', 2, 0, 4, None), ('belt', 2, 1, 8, None), ('splitter', 2, 2, 8, None), ('underground', 2, 3, 8, None),
        ('splitter', 2, 4, 4, None), ('underground', 2, 6, 8, 'output'), ('belt', 2, 7, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 8, None), ('belt', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('splitter', 4, 3, 4, None), ('splitter', 4, 6, 4, None), ('belt', 5, 0, 12, None), ('underground', 5, 1, 0, 'output'),
        ('belt', 5, 3, 4, None), ('belt', 5, 4, 4, None), ('underground', 5, 5, 0, None), ('belt', 5, 6, 0, None),
    ],
    (7, 4, 1, False): [  # Raynquist book 6/3: 7-4 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('belt', 1, 6, 4, None), ('belt', 1, 7, 4, None), ('belt', 1, 8, 0, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None),
        ('underground', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('belt', 2, 5, 4, None), ('splitter', 2, 6, 4, None), ('belt', 2, 8, 12, None),
        ('belt', 3, 0, 4, None), ('belt', 3, 2, 8, None), ('belt', 3, 3, 8, None), ('belt', 3, 4, 4, None), ('splitter', 3, 5, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}),
        ('belt', 3, 7, 4, None), ('belt', 3, 8, 12, None), ('belt', 4, 0, 4, None), ('belt', 4, 1, 4, None), ('underground', 4, 2, 0, 'output'),
        ('splitter', 4, 4, 4, None), ('underground', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 4, 8, 12, None), ('splitter', 5, 0, 4, None),
        ('underground', 5, 3, 4, 'output'), ('underground', 5, 4, 4, 'output'), ('belt', 5, 5, 8, None), ('belt', 5, 6, 8, None), ('belt', 5, 7, 8, None),
        ('belt', 5, 8, 12, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 8, None), ('belt', 6, 2, 4, None), ('splitter', 6, 3, 4, None),
        ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 8, None), ('underground', 7, 1, 8, None),
        ('splitter', 7, 2, 4, None), ('splitter', 7, 4, 4, None), ('underground', 7, 6, 8, 'output'), ('belt', 7, 7, 12, None), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 4, None), ('belt', 8, 4, 4, None), ('belt', 8, 5, 4, None),
    ],
    (7, 5, 1, False): [  # Raynquist book 10/11: 7-5 yellow balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('belt', 0, 8, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('splitter', 1, 7, 4, None), ('belt', 2, 0, 12, None), ('splitter', 2, 1, 4, None),
        ('belt', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('splitter', 2, 6, 4, None), ('belt', 2, 8, 4, None),
        ('belt', 3, 0, 12, None), ('belt', 3, 1, 4, None), ('splitter', 3, 2, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 3, 4, 4, None), ('belt', 3, 5, 0, None),
        ('belt', 3, 6, 0, None), ('belt', 3, 8, 4, None), ('belt', 4, 0, 12, None), ('belt', 4, 1, 8, None), ('underground', 4, 2, 8, None),
        ('splitter', 4, 3, 4, None), ('underground', 4, 5, 4, 'output'), ('underground', 4, 6, 8, 'output'), ('belt', 4, 7, 4, None), ('belt', 4, 8, 4, None),
        ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None), ('belt', 5, 3, 0, None), ('splitter', 5, 4, 4, None),
        ('splitter', 5, 7, 4, None), ('belt', 6, 0, 4, None), ('belt', 6, 1, 0, None), ('splitter', 6, 2, 0, None), ('belt', 6, 3, 0, None),
        ('belt', 6, 4, 0, None), ('underground', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None), ('belt', 6, 8, 4, None),
        ('belt', 7, 0, 4, None), ('belt', 7, 1, 4, None), ('underground', 7, 3, 0, 'output'), ('underground', 7, 4, 4, 'output'), ('underground', 7, 6, 4, None),
        ('underground', 7, 7, 0, None), ('belt', 7, 8, 0, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, None), ('belt', 8, 4, 4, None),
        ('belt', 8, 5, 4, None), ('belt', 8, 6, 0, None), ('belt', 8, 7, 0, None), ('belt', 8, 8, 0, None), ('belt', 9, 0, 8, None),
        ('splitter', 9, 1, 8, None), ('underground', 9, 2, 8, None), ('splitter', 9, 4, 4, None), ('underground', 9, 6, 8, 'output'), ('splitter', 9, 7, 8, None),
        ('belt', 9, 8, 12, None), ('belt', 10, 0, 8, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 4, None), ('belt', 10, 4, 0, None),
        ('belt', 10, 5, 8, None), ('belt', 10, 6, 8, None), ('belt', 10, 8, 4, None), ('belt', 11, 0, 12, None), ('underground', 11, 1, 4, 'output'),
        ('splitter', 11, 2, 4, None), ('underground', 11, 5, 4, 'output'), ('underground', 11, 6, 4, 'output'), ('belt', 11, 7, 4, None), ('belt', 11, 8, 0, None),
        ('belt', 12, 0, 12, None), ('splitter', 12, 1, 4, None), ('belt', 12, 3, 8, None), ('belt', 12, 4, 4, None), ('splitter', 12, 5, 4, None),
        ('belt', 12, 7, 4, None), ('belt', 13, 0, 12, None), ('belt', 13, 1, 0, None), ('belt', 13, 2, 8, None), ('belt', 13, 3, 4, None),
        ('splitter', 13, 4, 4, None), ('splitter', 13, 6, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None), ('belt', 14, 5, 4, None),
        ('belt', 14, 6, 4, None), ('belt', 14, 7, 4, None),
    ],
    (7, 5, 2, False): [  # Raynquist book 6/4: 7-5 balancer
        ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None),
        ('belt', 0, 7, 4, None), ('belt', 0, 8, 4, None), ('belt', 1, 0, 8, None), ('belt', 1, 1, 4, None), ('belt', 1, 2, 4, None),
        ('splitter', 1, 3, 4, None), ('splitter', 1, 5, 4, None), ('splitter', 1, 7, 4, None), ('belt', 2, 0, 12, None), ('splitter', 2, 1, 4, None),
        ('belt', 2, 3, 4, None), ('underground', 2, 4, 4, None), ('underground', 2, 5, 4, None), ('splitter', 2, 6, 4, None), ('belt', 2, 8, 4, None),
        ('belt', 3, 0, 12, None), ('belt', 3, 1, 4, None), ('splitter', 3, 2, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 3, 4, 4, None), ('belt', 3, 5, 0, None),
        ('belt', 3, 6, 0, None), ('belt', 3, 8, 4, None), ('belt', 4, 0, 12, None), ('belt', 4, 1, 8, None), ('underground', 4, 2, 8, None),
        ('splitter', 4, 3, 4, None), ('underground', 4, 5, 4, 'output'), ('underground', 4, 6, 8, 'output'), ('belt', 4, 7, 4, None), ('belt', 4, 8, 4, None),
        ('belt', 5, 0, 12, None), ('belt', 5, 1, 0, None), ('belt', 5, 2, 0, None), ('belt', 5, 3, 0, None), ('splitter', 5, 4, 4, None),
        ('splitter', 5, 7, 4, None), ('belt', 6, 1, 4, None), ('belt', 6, 2, 0, None), ('splitter', 6, 3, 0, None), ('belt', 6, 4, 0, None),
        ('underground', 6, 5, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None), ('belt', 6, 8, 4, None), ('belt', 7, 1, 4, None),
        ('belt', 7, 2, 4, None), ('underground', 7, 4, 0, 'output'), ('underground', 7, 6, 4, None), ('underground', 7, 7, 0, None), ('belt', 7, 8, 0, None),
        ('belt', 8, 1, 4, None), ('underground', 8, 2, 4, None), ('underground', 8, 4, 4, 'output'), ('belt', 8, 5, 4, None), ('belt', 8, 6, 0, None),
        ('belt', 8, 7, 0, None), ('belt', 8, 8, 0, None), ('belt', 9, 1, 8, None), ('splitter', 9, 2, 8, None), ('underground', 9, 3, 8, None),
        ('splitter', 9, 4, 4, None), ('underground', 9, 6, 8, 'output'), ('splitter', 9, 7, 8, None), ('belt', 9, 8, 12, None), ('belt', 10, 1, 8, None),
        ('belt', 10, 3, 4, None), ('belt', 10, 4, 4, None), ('belt', 10, 5, 8, None), ('belt', 10, 6, 8, None), ('belt', 10, 8, 4, None),
        ('belt', 11, 1, 12, None), ('underground', 11, 2, 4, 'output'), ('splitter', 11, 3, 4, None), ('underground', 11, 5, 4, 'output'), ('underground', 11, 6, 4, 'output'),
        ('belt', 11, 7, 4, None), ('belt', 11, 8, 0, None), ('belt', 12, 1, 12, None), ('splitter', 12, 2, 4, None), ('belt', 12, 4, 4, None),
        ('splitter', 12, 5, 4, None), ('belt', 12, 7, 4, None), ('belt', 13, 1, 12, None), ('belt', 13, 2, 0, None), ('belt', 13, 3, 4, None),
        ('splitter', 13, 4, 4, None), ('splitter', 13, 6, 4, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None), ('belt', 14, 5, 4, None),
        ('belt', 14, 6, 4, None), ('belt', 14, 7, 4, None),
    ],
    (7, 7, 3, False): [  # Raynquist book 6/6: 7-7 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None), ('underground', 1, 4, 4, None),
        ('underground', 1, 5, 4, None), ('belt', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None),
        ('belt', 2, 4, 4, None), ('belt', 2, 5, 0, None), ('belt', 2, 6, 0, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, None),
        ('belt', 3, 2, 8, None), ('underground', 3, 3, 8, None), ('belt', 3, 4, 4, None), ('underground', 3, 5, 8, 'output'), ('belt', 3, 6, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('underground', 4, 3, 4, 'output'), ('belt', 4, 4, 4, None),
        ('underground', 4, 6, 4, None), ('belt', 5, 0, 8, None), ('belt', 5, 1, 4, None), ('splitter', 5, 2, 4, None), ('belt', 5, 4, 8, None),
        ('belt', 5, 5, 8, None), ('belt', 5, 6, 4, None), ('belt', 6, 0, 12, None), ('splitter', 6, 1, 4, None), ('underground', 6, 3, 4, None),
        ('underground', 6, 5, 4, 'output'), ('belt', 6, 6, 4, None), ('belt', 7, 0, 12, None), ('belt', 7, 1, 4, None), ('underground', 7, 2, 4, None),
        ('splitter', 7, 5, 4, None), ('belt', 8, 0, 12, None), ('belt', 8, 1, 8, None), ('belt', 8, 2, 8, None), ('belt', 8, 3, 4, None),
        ('underground', 8, 4, 4, 'output'), ('underground', 8, 5, 4, None), ('belt', 8, 6, 4, None), ('belt', 9, 0, 12, None), ('splitter', 9, 1, 0, None),
        ('underground', 9, 2, 0, 'output'), ('splitter', 9, 3, 4, None), ('underground', 9, 5, 0, None), ('belt', 9, 6, 0, None), ('belt', 10, 0, 4, None),
        ('belt', 10, 2, 0, None), ('belt', 10, 3, 0, None), ('belt', 10, 4, 4, None), ('underground', 10, 5, 4, 'output'), ('belt', 11, 0, 4, None),
        ('underground', 11, 1, 4, 'output'), ('underground', 11, 3, 4, 'output'), ('splitter', 11, 4, 4, None), ('underground', 11, 6, 4, 'output'), ('splitter', 12, 0, 4, None),
        ('underground', 12, 2, 4, 'output'), ('splitter', 12, 3, 4, None), ('splitter', 12, 5, 4, None), ('belt', 13, 0, 4, None), ('belt', 13, 1, 4, None),
        ('belt', 13, 2, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None),
    ],
    (7, 8, 1, False): [  # Raynquist book 10/13: 7-8 yellow balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('underground', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('splitter', 1, 6, 4, None), ('belt', 2, 2, 4, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None), ('splitter', 2, 5, 4, None),
        ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None), ('belt', 3, 2, 0, None), ('underground', 3, 3, 4, None),
        ('belt', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None),
        ('splitter', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None),
        ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 8, None), ('underground', 5, 2, 8, None), ('belt', 5, 3, 4, None),
        ('splitter', 5, 4, 4, None), ('underground', 5, 6, 8, 'output'), ('belt', 5, 7, 4, None), ('belt', 6, 0, 12, None), ('underground', 6, 1, 4, 'output'),
        ('splitter', 6, 3, 4, None), ('belt', 6, 5, 4, None), ('underground', 6, 7, 4, None), ('belt', 7, 0, 12, None), ('belt', 7, 1, 4, None),
        ('belt', 7, 2, 4, None), ('belt', 7, 3, 0, None), ('belt', 7, 4, 4, None), ('belt', 7, 5, 8, None), ('belt', 7, 6, 8, None),
        ('belt', 7, 7, 4, None), ('belt', 8, 0, 12, None), ('splitter', 8, 1, 4, None), ('underground', 8, 3, 4, 'output'), ('underground', 8, 4, 4, None),
        ('underground', 8, 5, 4, 'output'), ('underground', 8, 6, 4, 'output'), ('belt', 8, 7, 4, None), ('belt', 9, 0, 12, None), ('belt', 9, 1, 0, None),
        ('splitter', 9, 2, 4, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 0, None), ('underground', 9, 6, 4, None), ('belt', 9, 7, 4, None),
        ('belt', 10, 0, 4, None), ('underground', 10, 1, 0, 'output'), ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('underground', 10, 6, 0, None),
        ('belt', 10, 7, 0, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None), ('belt', 11, 2, 0, None), ('underground', 11, 3, 4, None),
        ('splitter', 11, 4, 4, None), ('underground', 11, 6, 4, 'output'), ('underground', 11, 7, 4, 'output'), ('splitter', 12, 0, 4, None), ('belt', 12, 2, 4, None),
        ('belt', 12, 3, 0, None), ('belt', 12, 4, 0, None), ('belt', 12, 5, 4, None), ('splitter', 12, 6, 4, None), ('belt', 13, 0, 4, None),
        ('splitter', 13, 1, 4, None), ('underground', 13, 3, 4, 'output'), ('underground', 13, 4, 4, 'output'), ('splitter', 13, 5, 4, None), ('belt', 13, 7, 4, None),
        ('splitter', 14, 0, 4, None), ('splitter', 14, 2, 4, None), ('splitter', 14, 4, 4, None), ('splitter', 14, 6, 4, None), ('belt', 15, 0, 4, None),
        ('belt', 15, 1, 4, None), ('belt', 15, 2, 4, None), ('belt', 15, 3, 4, None), ('belt', 15, 4, 4, None), ('belt', 15, 5, 4, None),
        ('belt', 15, 6, 4, None), ('belt', 15, 7, 4, None),
    ],
    (7, 8, 2, False): [  # Raynquist book 6/7: 7-8 balancer
        ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None), ('belt', 0, 5, 4, None),
        ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('underground', 1, 1, 4, None), ('splitter', 1, 2, 4, None), ('splitter', 1, 4, 4, None),
        ('splitter', 1, 6, 4, None), ('belt', 2, 2, 4, None), ('underground', 2, 3, 4, None), ('belt', 2, 4, 4, None), ('underground', 2, 5, 4, None),
        ('underground', 2, 6, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 1, 4, None), ('belt', 3, 2, 0, None), ('belt', 3, 4, 4, None),
        ('belt', 3, 5, 4, None), ('belt', 3, 6, 0, None), ('belt', 3, 7, 0, None), ('belt', 4, 1, 8, None), ('splitter', 4, 2, 8, None),
        ('underground', 4, 3, 8, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('underground', 4, 6, 8, 'output'), ('belt', 4, 7, 4, None),
        ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 3, 4, None), ('splitter', 5, 4, 4, None), ('underground', 5, 7, 4, None),
        ('belt', 6, 0, 12, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 8, None), ('belt', 6, 6, 8, None), ('belt', 6, 7, 4, None),
        ('belt', 7, 0, 12, None), ('underground', 7, 1, 4, 'output'), ('belt', 7, 2, 4, None), ('belt', 7, 3, 0, None), ('underground', 7, 4, 4, None),
        ('underground', 7, 5, 4, 'output'), ('underground', 7, 6, 4, 'output'), ('belt', 7, 7, 4, None), ('belt', 8, 0, 12, None), ('splitter', 8, 1, 4, None),
        ('underground', 8, 3, 4, 'output'), ('splitter', 8, 5, 4, None), ('belt', 8, 7, 4, None), ('belt', 9, 0, 12, None), ('belt', 9, 1, 0, None),
        ('splitter', 9, 2, 4, None), ('belt', 9, 4, 4, None), ('belt', 9, 5, 0, None), ('underground', 9, 6, 4, None), ('belt', 9, 7, 4, None),
        ('belt', 10, 0, 4, None), ('underground', 10, 1, 0, 'output'), ('belt', 10, 2, 4, None), ('splitter', 10, 3, 4, None), ('underground', 10, 6, 0, None),
        ('belt', 10, 7, 0, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None), ('belt', 11, 2, 0, None), ('underground', 11, 3, 4, None),
        ('splitter', 11, 4, 4, None), ('underground', 11, 6, 4, 'output'), ('underground', 11, 7, 4, 'output'), ('splitter', 12, 0, 4, None), ('belt', 12, 2, 4, None),
        ('belt', 12, 3, 0, None), ('belt', 12, 4, 0, None), ('belt', 12, 5, 4, None), ('splitter', 12, 6, 4, None), ('belt', 13, 0, 4, None),
        ('splitter', 13, 1, 4, None), ('underground', 13, 3, 4, 'output'), ('underground', 13, 4, 4, 'output'), ('splitter', 13, 5, 4, None), ('belt', 13, 7, 4, None),
        ('splitter', 14, 0, 4, None), ('splitter', 14, 2, 4, None), ('splitter', 14, 4, 4, None), ('splitter', 14, 6, 4, None), ('belt', 15, 0, 4, None),
        ('belt', 15, 1, 4, None), ('belt', 15, 2, 4, None), ('belt', 15, 3, 4, None), ('belt', 15, 4, 4, None), ('belt', 15, 5, 4, None),
        ('belt', 15, 6, 4, None), ('belt', 15, 7, 4, None),
    ],
    (8, 1, 1, True): [  # Raynquist book 7/0: 8-1 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 5, 4, None), ('belt', 3, 2, 8, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 0, None), ('splitter', 4, 3, 4, None), ('belt', 5, 4, 4, None),
    ],
    (8, 2, 1, False): [  # Raynquist book 7/1: 8-2 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 5, 4, None), ('belt', 3, 2, 8, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 0, None), ('splitter', 4, 3, 4, None), ('belt', 5, 3, 4, None),
        ('belt', 5, 4, 4, None),
    ],
    (8, 2, 1, True): [  # Raynquist book 7/2: 8-2 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 2, 4, None), ('underground', 3, 3, 4, None),
        ('underground', 3, 5, 4, None), ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('underground', 4, 1, 8, None), ('belt', 4, 2, 8, None),
        ('belt', 4, 3, 8, None), ('belt', 4, 4, 4, None), ('underground', 4, 5, 8, 'output'), ('belt', 4, 6, 4, None), ('belt', 4, 7, 4, None),
        ('belt', 5, 0, 8, None), ('belt', 5, 1, 8, None), ('belt', 5, 2, 4, None), ('underground', 5, 3, 4, 'output'), ('belt', 5, 4, 4, None),
        ('underground', 5, 5, 4, 'output'), ('splitter', 5, 6, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 6, 0, 12, None), ('underground', 6, 1, 0, 'output'), ('splitter', 6, 2, 4, None),
        ('splitter', 6, 4, 4, None), ('underground', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 4, None),
    ],
    (8, 3, 1, False): [  # Raynquist book 7/3: 8-3 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 5, 4, None), ('belt', 3, 1, 4, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('belt', 3, 6, 4, None),
        ('belt', 4, 0, 4, None), ('belt', 4, 1, 0, None), ('splitter', 4, 2, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 4, 4, 4, None), ('belt', 4, 5, 0, None),
        ('belt', 4, 6, 0, None), ('belt', 5, 0, 8, None), ('splitter', 5, 1, 8, None), ('underground', 5, 2, 8, None), ('splitter', 5, 3, 4, None),
        ('underground', 5, 5, 8, 'output'), ('belt', 5, 6, 4, None), ('belt', 6, 0, 8, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None),
        ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None), ('belt', 7, 0, 12, None), ('belt', 7, 1, 0, None),
        ('splitter', 7, 2, 4, None), ('splitter', 7, 4, 4, None), ('belt', 8, 1, 12, None), ('belt', 8, 2, 0, None), ('belt', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 4, None),
    ],
    (8, 4, 1, False): [  # Raynquist book 7/4: 8-4 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('belt', 2, 3, 4, None), ('belt', 2, 4, 4, None),
        ('belt', 2, 7, 4, None), ('belt', 3, 0, 8, None), ('splitter', 3, 1, 8, None), ('underground', 3, 2, 8, None), ('splitter', 3, 3, 4, None),
        ('underground', 3, 5, 8, 'output'), ('belt', 3, 6, 4, None), ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 2, 4, None),
        ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 4, None),
        ('belt', 5, 0, 12, None), ('underground', 5, 1, 0, 'output'), ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None), ('underground', 5, 6, 0, None),
        ('belt', 5, 7, 0, None), ('belt', 6, 2, 4, None), ('belt', 6, 3, 4, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None),
    ],
    (8, 4, 1, True): [  # Raynquist book 10/14: 8-4 yellow TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None),
        ('underground', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('belt', 3, 6, 4, None),
        ('underground', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('underground', 4, 1, 8, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 8, 'output'),
        ('belt', 4, 5, 4, None), ('belt', 4, 6, 4, None), ('underground', 5, 0, 4, 'output'), ('underground', 5, 2, 4, None), ('splitter', 5, 5, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}),
        ('belt', 6, 0, 8, None), ('splitter', 6, 1, 8, None), ('underground', 6, 2, 8, None), ('underground', 6, 3, 4, 'output'), ('underground', 6, 4, 4, 'output'),
        ('underground', 6, 5, 4, None), ('underground', 6, 6, 8, 'output'), ('belt', 6, 7, 4, None), ('belt', 7, 0, 8, None), ('belt', 7, 2, 4, None),
        ('splitter', 7, 3, 4, None), ('belt', 7, 5, 4, None), ('belt', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('belt', 8, 0, 12, None),
        ('splitter', 8, 2, 4, None), ('splitter', 8, 4, 4, None), ('underground', 8, 7, 4, 'output'), ('belt', 9, 0, 12, None), ('underground', 9, 1, 0, 'output'),
        ('underground', 9, 2, 4, 'output'), ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None), ('underground', 9, 5, 4, 'output'), ('underground', 9, 6, 0, None),
        ('belt', 9, 7, 0, None), ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None),
        ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None),
    ],
    (8, 4, 2, True): [  # Raynquist book 7/5: 8-4 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None),
        ('underground', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('underground', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 0, None),
        ('belt', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('belt', 3, 6, 4, None),
        ('belt', 4, 0, 8, None), ('underground', 4, 1, 8, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 8, 'output'), ('belt', 4, 5, 4, None),
        ('belt', 4, 6, 4, None), ('underground', 5, 0, 4, 'output'), ('underground', 5, 2, 4, None), ('splitter', 5, 5, 4, {'output_priority': 'right', 'filter': 'deconstruction-planner'}), ('belt', 6, 0, 8, None),
        ('splitter', 6, 1, 8, None), ('underground', 6, 2, 8, None), ('underground', 6, 3, 4, 'output'), ('underground', 6, 4, 4, 'output'), ('underground', 6, 5, 4, None),
        ('underground', 6, 6, 8, 'output'), ('belt', 6, 7, 4, None), ('belt', 7, 0, 8, None), ('belt', 7, 2, 4, None), ('splitter', 7, 3, 4, None),
        ('belt', 7, 5, 4, None), ('belt', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('belt', 8, 0, 12, None), ('splitter', 8, 2, 4, None),
        ('splitter', 8, 4, 4, None), ('underground', 8, 7, 4, 'output'), ('belt', 9, 0, 12, None), ('underground', 9, 1, 0, 'output'), ('underground', 9, 2, 4, 'output'),
        ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None), ('underground', 9, 5, 4, 'output'), ('underground', 9, 6, 0, None), ('belt', 9, 7, 0, None),
        ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None),
        ('belt', 11, 5, 4, None),
    ],
    (8, 5, 1, False): [  # Raynquist book 10/15: 8-5 yellow balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 0, 'output'), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('underground', 3, 5, 0, None), ('belt', 3, 6, 0, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('belt', 4, 5, 0, None), ('splitter', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('splitter', 5, 1, 4, None), ('splitter', 5, 3, 4, None),
        ('belt', 5, 5, 4, None), ('belt', 5, 7, 0, None), ('splitter', 6, 2, 4, None), ('belt', 6, 4, 4, None), ('underground', 6, 5, 4, None),
        ('belt', 6, 7, 12, None), ('underground', 7, 0, 4, 'output'), ('belt', 7, 1, 4, None), ('belt', 7, 2, 0, None), ('underground', 7, 3, 4, None),
        ('belt', 7, 4, 8, None), ('belt', 7, 5, 8, None), ('splitter', 7, 6, 8, None), ('belt', 7, 7, 12, None), ('belt', 8, 0, 4, None),
        ('belt', 8, 1, 8, None), ('splitter', 8, 2, 8, None), ('underground', 8, 3, 8, None), ('underground', 8, 4, 4, 'output'), ('underground', 8, 5, 8, 'output'),
        ('belt', 8, 7, 4, None), ('belt', 9, 0, 8, None), ('splitter', 9, 1, 8, None), ('underground', 9, 3, 8, None), ('belt', 9, 4, 4, None),
        ('underground', 9, 5, 8, 'output'), ('belt', 9, 6, 4, None), ('belt', 9, 7, 4, None), ('belt', 10, 0, 8, None), ('belt', 10, 2, 4, None),
        ('underground', 10, 3, 4, 'output'), ('belt', 10, 4, 4, None), ('underground', 10, 5, 4, 'output'), ('belt', 10, 6, 4, None), ('belt', 10, 7, 4, None),
        ('belt', 11, 0, 12, None), ('belt', 11, 1, 0, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('belt', 11, 6, 4, None),
        ('belt', 11, 7, 4, None), ('belt', 12, 1, 12, None), ('belt', 12, 2, 0, None), ('splitter', 12, 3, 4, None), ('splitter', 12, 5, 4, None),
        ('belt', 12, 7, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None), ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None),
        ('belt', 13, 7, 4, None),
    ],
    (8, 5, 2, False): [  # Raynquist book 7/6: 8-5 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 0, 'output'), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('underground', 3, 5, 0, None), ('belt', 3, 6, 0, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None), ('belt', 4, 4, 4, None),
        ('belt', 4, 5, 0, None), ('splitter', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('splitter', 5, 1, 4, None), ('splitter', 5, 3, 4, None),
        ('belt', 5, 5, 4, None), ('belt', 5, 7, 0, None), ('splitter', 6, 2, 4, None), ('belt', 6, 4, 4, None), ('underground', 6, 5, 4, None),
        ('belt', 6, 6, 8, None), ('belt', 6, 7, 12, None), ('belt', 7, 2, 4, None), ('underground', 7, 3, 4, None), ('belt', 7, 4, 8, None),
        ('splitter', 7, 5, 8, None), ('belt', 7, 6, 12, None), ('underground', 8, 0, 4, 'output'), ('belt', 8, 2, 8, None), ('belt', 8, 3, 8, None),
        ('splitter', 8, 4, 8, None), ('belt', 8, 6, 8, None), ('belt', 8, 7, 4, None), ('belt', 9, 0, 8, None), ('splitter', 9, 1, 8, None),
        ('belt', 9, 2, 8, None), ('belt', 9, 3, 8, None), ('belt', 9, 5, 8, None), ('belt', 9, 6, 4, None), ('belt', 9, 7, 4, None),
        ('belt', 10, 0, 8, None), ('belt', 10, 2, 4, None), ('underground', 10, 3, 4, 'output'), ('underground', 10, 4, 4, 'output'), ('underground', 10, 5, 4, 'output'),
        ('belt', 10, 6, 4, None), ('belt', 10, 7, 4, None), ('belt', 11, 0, 12, None), ('belt', 11, 1, 0, None), ('splitter', 11, 2, 4, None),
        ('splitter', 11, 4, 4, None), ('belt', 11, 6, 4, None), ('belt', 11, 7, 4, None), ('belt', 12, 1, 12, None), ('belt', 12, 2, 0, None),
        ('splitter', 12, 3, 4, None), ('splitter', 12, 5, 4, None), ('belt', 12, 7, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None),
        ('belt', 13, 5, 4, None), ('belt', 13, 6, 4, None), ('belt', 13, 7, 4, None),
    ],
    (8, 6, 1, False): [  # Raynquist book 7/7: 8-6 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('underground', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, {'output_priority': 'left', 'filter': 'deconstruction-planner'}), ('belt', 2, 7, 4, None), ('underground', 3, 1, 4, None), ('belt', 3, 3, 4, None), ('splitter', 3, 4, 4, None),
        ('splitter', 3, 6, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None),
        ('underground', 4, 5, 4, None), ('underground', 4, 6, 4, None), ('belt', 4, 7, 4, None), ('belt', 5, 0, 12, None), ('underground', 5, 1, 0, 'output'),
        ('splitter', 5, 2, 4, None), ('belt', 5, 4, 4, None), ('belt', 5, 5, 0, None), ('underground', 5, 6, 0, None), ('belt', 5, 7, 0, None),
        ('underground', 6, 0, 4, 'output'), ('underground', 6, 1, 4, 'output'), ('underground', 6, 2, 4, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 12, None),
        ('belt', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('splitter', 7, 0, 4, None), ('belt', 7, 3, 4, None), ('belt', 7, 4, 8, None),
        ('belt', 7, 5, 8, None), ('splitter', 7, 6, 8, None), ('belt', 7, 7, 12, None), ('belt', 8, 0, 4, None), ('belt', 8, 1, 8, None),
        ('underground', 8, 2, 8, None), ('belt', 8, 3, 8, None), ('belt', 8, 4, 4, None), ('underground', 8, 5, 8, 'output'), ('belt', 8, 7, 4, None),
        ('belt', 9, 0, 8, None), ('belt', 9, 1, 8, None), ('belt', 9, 2, 8, None), ('belt', 9, 3, 4, None), ('belt', 9, 4, 4, None),
        ('underground', 9, 5, 4, 'output'), ('underground', 9, 6, 4, 'output'), ('belt', 9, 7, 4, None), ('underground', 10, 2, 4, 'output'), ('splitter', 10, 3, 4, None),
        ('splitter', 10, 5, 4, None), ('belt', 10, 7, 4, None), ('splitter', 11, 2, 4, None), ('splitter', 11, 4, 4, None), ('splitter', 11, 6, 4, None),
        ('belt', 12, 2, 4, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('belt', 12, 5, 4, None), ('belt', 12, 6, 4, None),
        ('belt', 12, 7, 4, None),
    ],
    (8, 7, 1, False): [  # Raynquist book 10/16: 8-7 yellow balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None),
        ('underground', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('splitter', 3, 0, 4, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('splitter', 3, 6, 4, None), ('underground', 4, 0, 4, None),
        ('underground', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 4, 'output'), ('belt', 4, 5, 4, None), ('belt', 4, 6, 0, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'), ('splitter', 5, 3, 4, None), ('belt', 5, 5, 4, None),
        ('underground', 5, 6, 0, None), ('belt', 5, 7, 0, None), ('belt', 6, 0, 4, None), ('underground', 6, 1, 4, 'output'), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 0, None), ('splitter', 6, 4, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 4, None),
        ('underground', 7, 1, 4, None), ('underground', 7, 2, 4, None), ('underground', 7, 3, 4, 'output'), ('underground', 7, 4, 4, None), ('splitter', 7, 5, 4, None),
        ('belt', 7, 7, 12, None), ('belt', 8, 0, 8, None), ('belt', 8, 1, 8, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 4, None),
        ('belt', 8, 4, 4, None), ('belt', 8, 5, 0, None), ('belt', 8, 6, 4, None), ('belt', 8, 7, 12, None), ('underground', 9, 0, 4, 'output'),
        ('belt', 9, 2, 4, None), ('splitter', 9, 3, 4, None), ('underground', 9, 6, 4, None), ('belt', 9, 7, 12, None), ('belt', 10, 0, 8, None),
        ('underground', 10, 1, 8, None), ('splitter', 10, 2, 4, None), ('belt', 10, 4, 4, None), ('underground', 10, 5, 8, 'output'), ('splitter', 10, 6, 8, None),
        ('belt', 10, 7, 12, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 0, None), ('belt', 11, 2, 0, None), ('belt', 11, 3, 4, None),
        ('belt', 11, 4, 8, None), ('belt', 11, 5, 8, None), ('belt', 11, 7, 4, None), ('belt', 12, 0, 4, None), ('underground', 12, 1, 4, 'output'),
        ('underground', 12, 2, 4, 'output'), ('belt', 12, 3, 4, None), ('underground', 12, 4, 4, 'output'), ('belt', 12, 5, 4, None), ('belt', 12, 6, 0, None),
        ('belt', 12, 7, 0, None), ('belt', 13, 0, 4, None), ('splitter', 13, 1, 4, None), ('belt', 13, 3, 4, None), ('belt', 13, 4, 4, None),
        ('belt', 13, 5, 4, None), ('splitter', 14, 0, 4, None), ('splitter', 14, 2, 4, None), ('splitter', 14, 4, 4, None), ('underground', 14, 6, 4, 'output'),
        ('belt', 15, 0, 4, None), ('belt', 15, 1, 4, None), ('belt', 15, 2, 4, None), ('belt', 15, 3, 4, None), ('belt', 15, 4, 4, None),
        ('belt', 15, 5, 4, None), ('belt', 15, 6, 4, None),
    ],
    (8, 7, 2, False): [  # Raynquist book 7/8: 8-7 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('underground', 2, 3, 4, None),
        ('underground', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('splitter', 3, 0, 4, None), ('belt', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 0, None), ('belt', 3, 5, 0, None), ('splitter', 3, 6, 4, None), ('underground', 4, 0, 4, None),
        ('underground', 4, 1, 4, None), ('splitter', 4, 2, 4, None), ('underground', 4, 4, 4, 'output'), ('belt', 4, 5, 4, None), ('belt', 4, 6, 0, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'), ('splitter', 5, 3, 4, None), ('belt', 5, 5, 4, None),
        ('underground', 5, 6, 0, None), ('belt', 5, 7, 0, None), ('belt', 6, 0, 4, None), ('underground', 6, 1, 4, 'output'), ('belt', 6, 2, 4, None),
        ('belt', 6, 3, 0, None), ('splitter', 6, 4, 4, None), ('belt', 6, 6, 4, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 4, None),
        ('splitter', 7, 1, 4, None), ('underground', 7, 4, 4, None), ('splitter', 7, 5, 4, None), ('belt', 7, 7, 12, None), ('belt', 8, 0, 4, None),
        ('underground', 8, 1, 4, None), ('underground', 8, 2, 4, None), ('underground', 8, 3, 4, 'output'), ('belt', 8, 4, 4, None), ('belt', 8, 5, 0, None),
        ('underground', 8, 6, 4, None), ('belt', 8, 7, 12, None), ('belt', 9, 0, 8, None), ('belt', 9, 1, 8, None), ('belt', 9, 2, 4, None),
        ('splitter', 9, 3, 4, None), ('belt', 9, 7, 12, None), ('underground', 10, 0, 4, 'output'), ('splitter', 10, 2, 4, None), ('belt', 10, 4, 8, None),
        ('splitter', 10, 5, 8, None), ('belt', 10, 6, 8, None), ('belt', 10, 7, 12, None), ('belt', 11, 0, 8, None), ('underground', 11, 1, 8, None),
        ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('underground', 11, 4, 8, 'output'), ('belt', 11, 6, 4, None), ('belt', 12, 0, 4, None),
        ('belt', 12, 1, 0, None), ('belt', 12, 2, 0, None), ('belt', 12, 3, 4, None), ('belt', 12, 5, 4, None), ('belt', 12, 6, 0, None),
        ('belt', 13, 0, 4, None), ('underground', 13, 1, 4, 'output'), ('underground', 13, 2, 4, 'output'), ('belt', 13, 3, 4, None), ('underground', 13, 4, 4, 'output'),
        ('belt', 13, 5, 4, None), ('splitter', 14, 0, 4, None), ('splitter', 14, 2, 4, None), ('splitter', 14, 4, 4, None), ('underground', 14, 6, 4, 'output'),
        ('belt', 15, 0, 4, None), ('belt', 15, 1, 4, None), ('belt', 15, 2, 4, None), ('belt', 15, 3, 4, None), ('belt', 15, 4, 4, None),
        ('belt', 15, 5, 4, None), ('belt', 15, 6, 4, None),
    ],
    (8, 8, 1, False): [  # Raynquist book 10/17: 8-8 yellow balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None),
        ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('underground', 4, 3, 4, None), ('underground', 4, 4, 4, None),
        ('belt', 4, 5, 4, None), ('belt', 4, 7, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'), ('belt', 5, 2, 8, None),
        ('belt', 5, 3, 4, None), ('underground', 5, 4, 0, None), ('belt', 5, 5, 0, None), ('belt', 5, 7, 4, None), ('belt', 6, 0, 4, None),
        ('underground', 6, 2, 4, 'output'), ('belt', 6, 3, 8, None), ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None),
        ('belt', 6, 7, 0, None), ('belt', 7, 0, 4, None), ('belt', 7, 2, 8, None), ('underground', 7, 3, 8, None), ('splitter', 7, 4, 4, None),
        ('underground', 7, 6, 8, 'output'), ('belt', 7, 7, 4, None), ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('belt', 8, 2, 4, None),
        ('belt', 8, 3, 0, None), ('belt', 8, 4, 0, None), ('belt', 8, 5, 4, None), ('underground', 8, 6, 4, 'output'), ('belt', 8, 7, 4, None),
        ('belt', 9, 0, 4, None), ('belt', 9, 1, 4, None), ('belt', 9, 2, 4, None), ('underground', 9, 3, 4, 'output'), ('underground', 9, 4, 4, 'output'),
        ('belt', 9, 5, 4, None), ('belt', 9, 6, 4, None), ('belt', 9, 7, 4, None), ('splitter', 10, 0, 4, None), ('splitter', 10, 2, 4, None),
        ('splitter', 10, 4, 4, None), ('splitter', 10, 6, 4, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None), ('belt', 11, 2, 4, None),
        ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None), ('belt', 11, 6, 4, None), ('belt', 11, 7, 4, None),
    ],
    (8, 8, 1, True): [  # Raynquist book 10/19: 8-8 yellow TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('belt', 2, 3, 4, None),
        ('belt', 2, 4, 4, None), ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, None),
        ('underground', 3, 2, 4, None), ('belt', 3, 3, 4, None), ('belt', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('underground', 3, 6, 4, None),
        ('belt', 3, 7, 4, None), ('belt', 4, 0, 8, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 4, None),
        ('belt', 4, 4, 4, None), ('belt', 4, 5, 4, None), ('belt', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 8, None),
        ('underground', 5, 1, 8, None), ('splitter', 5, 2, 4, None), ('splitter', 5, 4, 4, None), ('underground', 5, 6, 8, 'output'), ('belt', 5, 7, 4, None),
        ('belt', 6, 0, 12, None), ('belt', 6, 1, 0, None), ('belt', 6, 2, 0, None), ('splitter', 6, 3, 4, None), ('belt', 6, 5, 8, None),
        ('belt', 6, 6, 4, None), ('belt', 6, 7, 4, None), ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('underground', 7, 2, 4, 'output'),
        ('underground', 7, 3, 4, None), ('underground', 7, 4, 4, None), ('underground', 7, 5, 0, None), ('belt', 7, 6, 0, None), ('belt', 7, 7, 4, None),
        ('belt', 8, 0, 4, None), ('underground', 8, 1, 4, 'output'), ('belt', 8, 2, 8, None), ('belt', 8, 3, 8, None), ('belt', 8, 4, 4, None),
        ('underground', 8, 5, 4, 'output'), ('underground', 8, 6, 4, 'output'), ('belt', 8, 7, 4, None), ('splitter', 9, 0, 4, None), ('splitter', 9, 4, 4, None),
        ('splitter', 9, 6, 4, None), ('belt', 10, 0, 4, None), ('underground', 10, 1, 4, None), ('belt', 10, 2, 4, None), ('belt', 10, 3, 0, None),
        ('belt', 10, 4, 0, None), ('underground', 10, 5, 4, None), ('underground', 10, 6, 4, None), ('belt', 10, 7, 4, None), ('belt', 11, 0, 8, None),
        ('underground', 11, 1, 8, None), ('underground', 11, 2, 4, None), ('underground', 11, 5, 8, 'output'), ('belt', 11, 6, 4, None), ('belt', 11, 7, 4, None),
        ('belt', 12, 0, 8, None), ('belt', 12, 1, 8, None), ('belt', 12, 2, 4, None), ('underground', 12, 3, 4, 'output'), ('underground', 12, 4, 4, 'output'),
        ('belt', 12, 5, 4, None), ('belt', 12, 6, 0, None), ('belt', 12, 7, 4, None), ('belt', 13, 0, 12, None), ('underground', 13, 1, 0, 'output'),
        ('splitter', 13, 2, 4, None), ('splitter', 13, 4, 4, None), ('underground', 13, 6, 0, None), ('belt', 13, 7, 0, None), ('belt', 14, 0, 4, None),
        ('belt', 14, 1, 0, None), ('belt', 14, 2, 0, None), ('belt', 14, 3, 4, None), ('belt', 14, 4, 4, None), ('belt', 14, 5, 8, None),
        ('belt', 14, 6, 8, None), ('belt', 14, 7, 4, None), ('belt', 15, 0, 4, None), ('underground', 15, 1, 4, 'output'), ('underground', 15, 2, 4, 'output'),
        ('belt', 15, 3, 4, None), ('belt', 15, 4, 4, None), ('underground', 15, 5, 4, 'output'), ('underground', 15, 6, 4, 'output'), ('belt', 15, 7, 4, None),
        ('belt', 16, 0, 4, None), ('splitter', 16, 1, 4, None), ('belt', 16, 3, 4, None), ('belt', 16, 4, 4, None), ('splitter', 16, 5, 4, None),
        ('belt', 16, 7, 4, None), ('splitter', 17, 0, 4, None), ('splitter', 17, 2, 4, None), ('splitter', 17, 4, 4, None), ('splitter', 17, 6, 4, None),
        ('belt', 18, 0, 4, None), ('belt', 18, 1, 4, None), ('belt', 18, 2, 4, None), ('belt', 18, 3, 4, None), ('belt', 18, 4, 4, None),
        ('belt', 18, 5, 4, None), ('belt', 18, 6, 4, None), ('belt', 18, 7, 4, None),
    ],
    (8, 8, 2, False): [  # Raynquist book 7/9: 8-8 balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('underground', 3, 1, 4, None), ('underground', 3, 2, 4, None),
        ('underground', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('belt', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 8, None), ('belt', 4, 1, 8, None), ('belt', 4, 2, 8, None), ('belt', 4, 3, 4, None), ('belt', 4, 5, 4, None),
        ('belt', 4, 7, 4, None), ('belt', 5, 0, 4, None), ('underground', 5, 1, 0, 'output'), ('belt', 5, 3, 4, None), ('underground', 5, 4, 0, None),
        ('belt', 5, 5, 0, None), ('belt', 5, 7, 4, None), ('belt', 6, 0, 4, None), ('underground', 6, 2, 4, 'output'), ('belt', 6, 3, 8, None),
        ('belt', 6, 4, 4, None), ('belt', 6, 5, 4, None), ('belt', 6, 6, 0, None), ('belt', 6, 7, 0, None), ('belt', 7, 0, 4, None),
        ('belt', 7, 2, 8, None), ('underground', 7, 3, 8, None), ('splitter', 7, 4, 4, None), ('underground', 7, 6, 8, 'output'), ('belt', 7, 7, 4, None),
        ('belt', 8, 0, 4, None), ('belt', 8, 2, 4, None), ('belt', 8, 3, 0, None), ('belt', 8, 4, 0, None), ('belt', 8, 5, 4, None),
        ('belt', 8, 7, 4, None), ('belt', 9, 0, 4, None), ('underground', 9, 1, 4, 'output'), ('belt', 9, 2, 4, None), ('underground', 9, 3, 4, 'output'),
        ('underground', 9, 4, 4, 'output'), ('belt', 9, 5, 4, None), ('underground', 9, 6, 4, 'output'), ('belt', 9, 7, 4, None), ('splitter', 10, 0, 4, None),
        ('splitter', 10, 2, 4, None), ('splitter', 10, 4, 4, None), ('splitter', 10, 6, 4, None), ('belt', 11, 0, 4, None), ('belt', 11, 1, 4, None),
        ('belt', 11, 2, 4, None), ('belt', 11, 3, 4, None), ('belt', 11, 4, 4, None), ('belt', 11, 5, 4, None), ('belt', 11, 6, 4, None),
        ('belt', 11, 7, 4, None),
    ],
    (8, 8, 2, True): [  # Raynquist book 7/14: 8-8 TU balancer
        ('belt', 0, 0, 4, None), ('belt', 0, 1, 4, None), ('belt', 0, 2, 4, None), ('belt', 0, 3, 4, None), ('belt', 0, 4, 4, None),
        ('belt', 0, 5, 4, None), ('belt', 0, 6, 4, None), ('belt', 0, 7, 4, None), ('splitter', 1, 0, 4, None), ('splitter', 1, 2, 4, None),
        ('splitter', 1, 4, 4, None), ('splitter', 1, 6, 4, None), ('belt', 2, 0, 4, None), ('splitter', 2, 1, 4, None), ('splitter', 2, 3, 4, None),
        ('splitter', 2, 5, 4, None), ('belt', 2, 7, 4, None), ('belt', 3, 0, 4, None), ('belt', 3, 1, 4, None), ('underground', 3, 2, 4, None),
        ('underground', 3, 3, 4, None), ('underground', 3, 4, 4, None), ('underground', 3, 5, 4, None), ('underground', 3, 6, 4, None), ('belt', 3, 7, 4, None),
        ('belt', 4, 0, 4, None), ('underground', 4, 1, 4, None), ('belt', 4, 2, 4, None), ('belt', 4, 3, 0, None), ('belt', 4, 4, 0, None),
        ('belt', 4, 5, 0, None), ('splitter', 4, 6, 0, None), ('belt', 4, 7, 0, None), ('belt', 5, 0, 4, None), ('belt', 5, 2, 4, None),
        ('underground', 5, 3, 4, 'output'), ('underground', 5, 4, 4, 'output'), ('belt', 5, 5, 4, None), ('belt', 5, 7, 0, None), ('belt', 6, 0, 8, None),
        ('underground', 6, 1, 8, None), ('splitter', 6, 2, 4, None), ('splitter', 6, 4, 4, None), ('underground', 6, 6, 8, 'output'), ('belt', 6, 7, 12, None),
        ('belt', 7, 0, 4, None), ('underground', 7, 1, 0, 'output'), ('belt', 7, 2, 4, None), ('splitter', 7, 3, 4, None), ('belt', 7, 5, 4, None),
        ('underground', 7, 6, 0, None), ('belt', 7, 7, 0, None), ('underground', 8, 0, 4, None), ('belt', 8, 2, 4, None), ('underground', 8, 3, 4, None),
        ('underground', 8, 4, 4, None), ('belt', 8, 5, 8, None), ('splitter', 8, 6, 8, None), ('belt', 8, 7, 12, None), ('belt', 9, 2, 8, None),
        ('belt', 9, 3, 8, None), ('belt', 9, 4, 8, None), ('belt', 9, 5, 8, None), ('belt', 9, 7, 4, None), ('belt', 10, 0, 4, None),
        ('underground', 10, 1, 0, 'output'), ('underground', 10, 2, 4, 'output'), ('underground', 10, 3, 0, None), ('belt', 10, 4, 0, None), ('underground', 10, 5, 4, 'output'),
        ('underground', 10, 6, 4, 'output'), ('belt', 10, 7, 4, None), ('belt', 11, 0, 4, None), ('underground', 11, 1, 4, 'output'), ('belt', 11, 2, 4, None),
        ('belt', 11, 4, 12, None), ('belt', 11, 5, 0, None), ('belt', 11, 6, 4, None), ('belt', 11, 7, 4, None), ('splitter', 12, 0, 4, None),
        ('belt', 12, 2, 8, None), ('belt', 12, 3, 4, None), ('belt', 12, 4, 4, None), ('belt', 12, 5, 0, None), ('belt', 12, 6, 0, None),
        ('belt', 12, 7, 4, None), ('belt', 13, 0, 4, None), ('belt', 13, 1, 8, None), ('underground', 13, 2, 8, None), ('splitter', 13, 3, 4, None),
        ('underground', 13, 5, 8, 'output'), ('belt', 13, 6, 4, None), ('belt', 13, 7, 4, None), ('belt', 14, 0, 8, None), ('belt', 14, 1, 4, None),
        ('belt', 14, 2, 4, None), ('belt', 14, 3, 0, None), ('belt', 14, 4, 8, None), ('belt', 14, 5, 4, None), ('belt', 14, 6, 4, None),
        ('belt', 14, 7, 4, None), ('underground', 15, 0, 4, 'output'), ('splitter', 15, 1, 4, None), ('underground', 15, 3, 4, 'output'), ('underground', 15, 4, 4, 'output'),
        ('splitter', 15, 5, 4, None), ('belt', 15, 7, 4, None), ('splitter', 16, 0, 4, None), ('splitter', 16, 2, 4, None), ('splitter', 16, 4, 4, None),
        ('splitter', 16, 6, 4, None), ('belt', 17, 0, 4, None), ('belt', 17, 1, 4, None), ('belt', 17, 2, 4, None), ('belt', 17, 3, 4, None),
        ('belt', 17, 4, 4, None), ('belt', 17, 5, 4, None), ('belt', 17, 6, 4, None), ('belt', 17, 7, 4, None),
    ],
}
# END BALANCER TEMPLATES
