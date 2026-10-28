"""
Static blueprint validator for FactorioLLM (no code execution).

Checks a blueprint IR (tile grid, see blueprint_ir.py) for layouts that cannot work in game.
Severity policy:
- error: never works in game (overlapping entities; an inserter taking from / dropping into a
  building without an inventory, such as a wall, lamp, pole or pipe);
- warning: usually a mistake but sometimes intentional (belt ends, lone underground belts,
  inserters working on the ground, unpowered consumers).
Every issue has a category so warning rates can be compared with real player blueprints.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from factoriollm.blueprint_ir import (
    IRBlueprint,
    IREntity,
    _geometry,
    _prototype,
    entity_type,
    extract_facts,
    footprint,
    parse_blueprint_string,
)

_DIRECTION_VECTORS = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}

BELT_TYPES = frozenset({"transport-belt", "underground-belt", "splitter", "loader", "loader-1x1", "linked-belt"})

# Prototype fields that give an entity an inventory an inserter can reach (chests, wagons, crafting
# machines, labs, roboports, turrets, the rocket silo); besides, a burner takes fuel. A heat exchanger
# is a "boiler" without fuel: no inventory.
_INVENTORY_FIELDS = ("inventory_size", "source_inventory_size", "result_inventory_size", "crafting_categories",
                     "inputs", "robot_slots_count", "automated_ammo_count", "rocket_parts_required")

_CONSUMER_PRIORITIES = frozenset({"lamp", "primary-input", "secondary-input"})


@dataclass(frozen=True)
class Issue:
    category: str
    message: str
    position: Tuple[int, int]


@dataclass
class ValidationReport:
    errors: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def category_counts(self) -> Dict[str, int]:
        return dict(Counter(i.category for i in self.errors + self.warnings))

    def feedback(self, max_items: int = 10, include_warnings: bool = False) -> str:
        """Short issue list suitable as feedback for the model."""
        issues = self.errors + (self.warnings if include_warnings else [])
        lines = [f"- [{i.category}] {i.message}" for i in issues[:max_items]]
        if len(issues) > max_items:
            lines.append(f"- ... and {len(issues) - max_items} more")
        return "\n".join(lines)


# --- Prototype helpers --------------------------------------------------------------------

@lru_cache(maxsize=None)
def _energy(name: str) -> Tuple[Optional[str], Optional[str]]:
    source = (_prototype(name) or {}).get("energy_source") or {}
    return source.get("type"), source.get("usage_priority")


def is_electric_consumer(name: str) -> bool:
    kind, priority = _energy(name)
    return kind == "electric" and priority in _CONSUMER_PRIORITIES


def has_item_interface(name: str) -> bool:
    """Whether an inserter can take from / put into the entity: belts, an inventory, or a burner's fuel (from its prototype)."""
    if entity_type(name) in BELT_TYPES:
        return True
    proto = _prototype(name) or {}
    if proto.get("type") == "inserter":
        return False
    return any(key in proto for key in _INVENTORY_FIELDS) or _energy(name)[0] == "burner"


def _rotate(vec: Tuple[float, float], direction: int) -> Tuple[float, float]:
    x, y = vec
    for _ in range((direction // 4) % 4):
        x, y = -y, x  # clockwise quarter turn (y points down)
    return x, y


def _center(e: IREntity) -> Tuple[float, float]:
    _, _, cdx, cdy = _geometry(e.name, e.direction)
    return e.x + cdx, e.y + cdy


# --- Validation ---------------------------------------------------------------------------

def _occupied_tiles(e: IREntity) -> List[Tuple[int, int]]:
    w, h = footprint(e.name, e.direction)
    return [(e.x + dx, e.y + dy) for dx in range(w) for dy in range(h)]


def _reach_tiles(e: IREntity) -> List[Tuple[int, int]]:
    """Tiles outside the footprint that the collision box still blocks: the 2.0 offshore pump
    reaches about one tile in front of it (towards the water)."""
    if e.name == "offshore-pump":
        dx, dy = _DIRECTION_VECTORS[e.direction]
        return [(e.x + dx, e.y + dy)]
    return []


def validate(ir: IRBlueprint) -> ValidationReport:
    report = ValidationReport(facts=extract_facts(ir))
    ents = ir.entities
    if not ents:
        return report

    tiles: Dict[Tuple[int, int], int] = {}
    seen_pairs = set()
    for i, e in enumerate(ents):
        for tile in _occupied_tiles(e):
            j = tiles.get(tile)
            if j is None:
                tiles[tile] = i
            elif (j, i) not in seen_pairs:
                seen_pairs.add((j, i))
                report.errors.append(Issue("collision", f"{e.name} at {(e.x, e.y)} overlaps {ents[j].name} at {(ents[j].x, ents[j].y)}", (e.x, e.y)))
    for e in ents:
        for tile in _reach_tiles(e):
            j = tiles.get(tile)
            if j is not None:
                report.errors.append(Issue("collision", f"{e.name} at {(e.x, e.y)} reaches into {ents[j].name} at {(ents[j].x, ents[j].y)}", (e.x, e.y)))

    min_x = min(t[0] for t in tiles)
    max_x = max(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)
    max_y = max(t[1] for t in tiles)

    def inside(tile: Tuple[int, int]) -> bool:
        return min_x <= tile[0] <= max_x and min_y <= tile[1] <= max_y

    _check_belts(ents, tiles, inside, report)
    _check_undergrounds(ents, report)
    _check_inserters(ents, tiles, inside, report)
    _check_power(ents, report)
    _check_pole_network(ents, report)
    return report


def validate_string(blueprint_string: str) -> ValidationReport:
    """
    Parses (size-limited, no execution) and validates a blueprint string. Off-grid entities are
    snapped to the tile the game would use and reported as an `off_grid` error, so the rest of the
    report (counts, collisions) still reflects what would appear in game.
    """
    stats: Dict[str, int] = {}
    report = validate(parse_blueprint_string(blueprint_string, allowed=None, check_collisions=False,
                                             snap_off_grid=True, stats=stats))
    if stats.get("off_grid"):
        report.errors.insert(0, Issue(
            "off_grid",
            f"{stats['off_grid']} entities are not aligned to the tile grid; use integer "
            f"tile_position=(x, y) for the top-left tile instead of center positions",
            (0, 0),
        ))
    return report


def _check_belts(ents, tiles, inside, report: ValidationReport) -> None:
    for e in ents:
        etype = entity_type(e.name)
        if etype not in ("transport-belt", "underground-belt", "splitter"):
            continue
        if etype == "underground-belt" and e.extras_dict.get("io_type", "input") == "input":
            continue  # items go underground
        dx, dy = _DIRECTION_VECTORS[e.direction]
        w, h = footprint(e.name, e.direction)
        lane_tiles = [(e.x + ox, e.y + oy) for ox in range(w) for oy in range(h)]
        for tx, ty in lane_tiles:
            front = (tx + dx, ty + dy)
            j = tiles.get(front)
            if j is None:
                if inside(front) and etype != "splitter":  # unused splitter outputs are common
                    report.warnings.append(Issue("belt_end", f"{e.name} at {(tx, ty)} facing {front} ends on an empty tile", (tx, ty)))
                continue
            other = ents[j]
            if other is e:
                continue
            otype = entity_type(other.name)
            if otype in BELT_TYPES:
                if otype != "underground-belt" or other.extras_dict.get("io_type", "input") == "input":
                    if (other.direction - e.direction) % 16 == 8:
                        report.warnings.append(Issue("belt_head_on", f"{e.name} at {(tx, ty)} faces {other.name} at {front} head-on", (tx, ty)))
            elif etype != "splitter":
                report.warnings.append(Issue("belt_end", f"{e.name} at {(tx, ty)} runs into {other.name} at {front}", (tx, ty)))


def _check_undergrounds(ents, report: ValidationReport) -> None:
    from factoriollm.dataset_lifter import _extract_underground_pairs

    used: set = set()
    _extract_underground_pairs(ents, used)
    for i, e in enumerate(ents):
        if entity_type(e.name) == "underground-belt" and i not in used:
            io = e.extras_dict.get("io_type", "input")
            report.warnings.append(Issue("lone_underground", f"{e.name} ({io}) at {(e.x, e.y)} has no matching pair", (e.x, e.y)))


def _check_inserters(ents, tiles, inside, report: ValidationReport) -> None:
    for e in ents:
        if entity_type(e.name) != "inserter":
            continue
        proto = _prototype(e.name) or {}
        cx, cy = _center(e)
        for key, verb in (("pickup_position", "takes from"), ("insert_position", "drops into")):
            vec = proto.get(key)
            if not vec:
                continue
            vx, vy = _rotate((float(vec[0]), float(vec[1])), e.direction)
            target = (math.floor(cx + vx), math.floor(cy + vy))
            j = tiles.get(target)
            if j is None:
                if not inside(target):
                    continue  # tileable modules work with belts / wagons outside the blueprint
                report.warnings.append(Issue("inserter_ground", f"{e.name} at {(e.x, e.y)} {verb} empty tile {target}", (e.x, e.y)))
            elif not has_item_interface(ents[j].name):
                report.errors.append(Issue("inserter_target", f"{e.name} at {(e.x, e.y)} {verb} {ents[j].name} at {target}, which holds no items", (e.x, e.y)))


def _check_pole_network(ents, report: ValidationReport) -> None:
    """Poles that cannot be wired into one network (wire reach), so some groups get no power."""
    poles = [e for e in ents if entity_type(e.name) == "electric-pole"]
    if len(poles) < 2:
        return
    centers = [_center(e) for e in poles]
    reach = [float((_prototype(e.name) or {}).get("maximum_wire_distance") or 0) for e in poles]
    seen, islands = set(), []
    for start in range(len(poles)):
        if start in seen:
            continue
        seen.add(start)
        stack, size = [start], 0
        while stack:
            i = stack.pop()
            size += 1
            for j in range(len(poles)):
                if j not in seen and math.dist(centers[i], centers[j]) <= min(reach[i], reach[j]) + 1e-9:
                    seen.add(j)
                    stack.append(j)
        islands.append((size, poles[start]))
    if len(islands) > 1:
        islands.sort(key=lambda s: s[0])
        size, pole = islands[0]
        report.warnings.append(Issue("pole_islands", f"{len(islands)} separate pole networks that wires cannot join "
                                     f"(smallest: {size} poles near {(pole.x, pole.y)})", (pole.x, pole.y)))


def _check_power(ents, report: ValidationReport) -> None:
    consumers = [e for e in ents if is_electric_consumer(e.name)]
    if not consumers:
        return
    areas = []
    for e in ents:
        if entity_type(e.name) == "electric-pole":
            d = float((_prototype(e.name) or {}).get("supply_area_distance") or 0)
            cx, cy = _center(e)
            areas.append((cx - d, cy - d, cx + d, cy + d))
    if not areas:
        report.warnings.append(Issue("no_poles", f"{len(consumers)} electric consumers but no power poles", (consumers[0].x, consumers[0].y)))
        return
    unpowered = []
    for e in consumers:
        w, h = footprint(e.name, e.direction)
        if not any(e.x < x2 and e.x + w > x1 and e.y < y2 and e.y + h > y1 for x1, y1, x2, y2 in areas):
            unpowered.append(e)
    if unpowered:
        sample = ", ".join(f"{e.name} at {(e.x, e.y)}" for e in unpowered[:3])
        report.warnings.append(Issue("unpowered", f"{len(unpowered)} electric consumers outside pole coverage (e.g. {sample})", (unpowered[0].x, unpowered[0].y)))
