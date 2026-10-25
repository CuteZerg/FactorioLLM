"""
Intermediate representation (IR) for Factorio blueprints.

Blueprint strings are parsed from their raw JSON (not via draftsman's importer, which fails on
many 1.x strings) into a list of entities on an integer tile grid:

    IREntity(name, x, y, direction, extras)

where (x, y) is the top-left tile of the entity footprint. Integer tiles are what the model
writes (`tile_position=(x, y)`), so 1x1 and 2x2 entities share one consistent grid and there is
no half-tile ambiguity.

Draftsman is only used as a prototype database (entity sizes, types, rotatability).
"""

from __future__ import annotations

import base64
import json
import math
import warnings
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple

warnings.filterwarnings("ignore", module="draftsman")

# Hard caps for decoding untrusted blueprint strings (zip-bomb protection).
MAX_BLUEPRINT_STRING_LEN = 4_000_000
MAX_DECOMPRESSED_BYTES = 32_000_000

# --- Entity policy -------------------------------------------------------------------------

# Factorio 2.0 base-game (no Space Age) player-buildable entities we train on.
VANILLA_ENTITIES = frozenset({
    "transport-belt", "fast-transport-belt", "express-transport-belt",
    "underground-belt", "fast-underground-belt", "express-underground-belt",
    "splitter", "fast-splitter", "express-splitter",
    "burner-inserter", "inserter", "long-handed-inserter", "fast-inserter", "bulk-inserter",
    "wooden-chest", "iron-chest", "steel-chest",
    "active-provider-chest", "passive-provider-chest", "storage-chest", "buffer-chest", "requester-chest",
    "stone-furnace", "steel-furnace", "electric-furnace",
    "assembling-machine-1", "assembling-machine-2", "assembling-machine-3",
    "oil-refinery", "chemical-plant", "centrifuge",
    "burner-mining-drill", "electric-mining-drill", "pumpjack", "offshore-pump", "pump",
    "pipe", "pipe-to-ground", "storage-tank",
    "boiler", "heat-exchanger", "steam-engine", "steam-turbine", "nuclear-reactor", "heat-pipe",
    "solar-panel", "accumulator",
    "small-electric-pole", "medium-electric-pole", "big-electric-pole", "substation",
    "lab", "beacon", "radar", "small-lamp", "roboport", "rocket-silo",
    "stone-wall", "gate", "gun-turret", "laser-turret", "flamethrower-turret", "artillery-turret",
    "land-mine",
    "constant-combinator", "display-panel",
})

# Players mark what a belt or block carries with unwired constant combinators and display panels.
# They are kept as labels (signals / text / icon); wired to anything, they are circuit logic,
# which the IR cannot express, and the blueprint is rejected.
ANNOTATION_ENTITIES = frozenset({"constant-combinator", "display-panel"})
LABEL_SIGNAL_TYPES = frozenset({"item", "fluid", "virtual", "entity", "recipe"})
MAX_LABEL_SIGNALS = 40

# Vanilla, but excluded for now: the decompiler does not emit wires / combinator conditions /
# train schedules, so these blueprints would teach prompts that the code cannot fulfil.
RAIL_AND_CIRCUIT_ENTITIES = frozenset({
    "straight-rail", "curved-rail-a", "curved-rail-b", "half-diagonal-rail",
    "legacy-straight-rail", "legacy-curved-rail",
    "rail-signal", "rail-chain-signal", "train-stop",
    "locomotive", "cargo-wagon", "fluid-wagon", "artillery-wagon",
    "arithmetic-combinator", "decider-combinator", "selector-combinator",
    "power-switch", "programmable-speaker",
})

# Factorio 1.x -> 2.0 prototype renames (official migrations).
LEGACY_RENAMES = {
    "stack-inserter": "bulk-inserter",
    "stack-filter-inserter": "bulk-inserter",
    "filter-inserter": "fast-inserter",
    "logistic-chest-active-provider": "active-provider-chest",
    "logistic-chest-passive-provider": "passive-provider-chest",
    "logistic-chest-storage": "storage-chest",
    "logistic-chest-buffer": "buffer-chest",
    "logistic-chest-requester": "requester-chest",
    "straight-rail": "legacy-straight-rail",
    "curved-rail": "legacy-curved-rail",
}

# Entity attributes carried into the IR (and into generated code). Everything else
# (modules, wires, inserter filters, quality, tags) is dropped.
EXTRA_KEYS = ("recipe", "io_type", "input_priority", "output_priority", "filter", "signals", "text", "icon")

# Types whose direction is cosmetic: never emitted, never rotated.
_NON_DIRECTIONAL_TYPES = frozenset({"furnace"})


class BlueprintIRError(ValueError):
    """Raised when a blueprint cannot be represented in the IR."""


@dataclass(frozen=True)
class IREntity:
    name: str
    x: int
    y: int
    direction: int = 0
    extras: Tuple[Tuple[str, str], ...] = ()

    @property
    def extras_dict(self) -> Dict[str, str]:
        return dict(self.extras)

    @property
    def directional(self) -> bool:
        return is_directional(self.name, self.extras_dict.get("recipe"))


@dataclass
class IRBlueprint:
    entities: List[IREntity]
    label: str = ""
    version: Tuple[int, int] = (2, 0)
    book_path: str = ""
    # Grid snapping ("snap-to-grid" etc.): copies of a tileable blueprint line up on this grid.
    # The game writes "absolute-snapping" only as true; a missing key means relative snapping.
    snap: Optional[Tuple[int, int]] = None
    absolute_snap: bool = False
    snap_offset: Tuple[int, int] = (0, 0)

    @property
    def size(self) -> Tuple[int, int]:
        if not self.entities:
            return (0, 0)
        w = max(e.x + footprint(e.name, e.direction)[0] for e in self.entities) - min(e.x for e in self.entities)
        h = max(e.y + footprint(e.name, e.direction)[1] for e in self.entities) - min(e.y for e in self.entities)
        return (w, h)

    def snap_key(self) -> Tuple[Any, ...]:
        return (self.snap, self.absolute_snap, self.snap_offset if self.absolute_snap else (0, 0))

    def with_entities(self, entities: List[IREntity], snap: Optional[Tuple[int, int]] = None) -> "IRBlueprint":
        """Same blueprint metadata with other entities (and optionally another grid size)."""
        return IRBlueprint(entities, self.label, self.version, self.book_path, snap or self.snap,
                           self.absolute_snap, self.snap_offset)

    def canonical_set(self) -> frozenset:
        return frozenset(self.entities)


# --- Prototype database (draftsman) -------------------------------------------------------

@lru_cache(maxsize=None)
def _prototype(name: str) -> Optional[Dict[str, Any]]:
    from draftsman.data import entities
    return entities.raw.get(name)


def entity_type(name: str) -> str:
    proto = _prototype(name)
    if proto is None:
        raise BlueprintIRError(f"unknown entity '{name}'")
    return proto["type"]


@lru_cache(maxsize=None)
def _geometry(name: str, direction: int) -> Tuple[int, int, float, float]:
    """Returns (tile_width, tile_height, center_dx, center_dy) for an entity at tile (0, 0)."""
    from draftsman.entity import new_entity
    if _prototype(name) is None:
        raise BlueprintIRError(f"unknown entity '{name}'")
    kwargs = {"direction": direction} if direction else {}  # non-rotatable classes reject `direction`
    ent = new_entity(name, tile_position=(0, 0), **kwargs)
    return (int(ent.tile_width), int(ent.tile_height), float(ent.position.x), float(ent.position.y))


def footprint(name: str, direction: int = 0) -> Tuple[int, int]:
    w, h, _, _ = _geometry(name, direction)
    return (w, h)


@lru_cache(maxsize=None)
def _recipe_exists(recipe: str) -> bool:
    from draftsman.data import recipes
    return recipe in recipes.raw


@lru_cache(maxsize=None)
def _recipe_uses_fluid(recipe: str) -> bool:
    from draftsman.data import recipes
    data = recipes.raw.get(recipe) or {}
    parts = (data.get("ingredients") or []) + (data.get("results") or [])
    return any(isinstance(p, dict) and p.get("type") == "fluid" for p in parts)


@lru_cache(maxsize=None)
def is_directional(name: str, recipe: Optional[str] = None) -> bool:
    """Whether the entity's direction matters in game (and therefore appears in code)."""
    etype = entity_type(name)
    if etype in _NON_DIRECTIONAL_TYPES:
        return False
    if etype == "assembling-machine":
        # Only fluid boxes give an assembler an orientation.
        return recipe is not None and _recipe_uses_fluid(recipe)
    from draftsman.entity import new_entity
    return bool(getattr(new_entity(name), "rotatable", False))


# --- Decoding -----------------------------------------------------------------------------

def decode_blueprint_string(blueprint_string: str) -> Dict[str, Any]:
    """Safely decodes a Factorio blueprint string into JSON with size limits."""
    s = (blueprint_string or "").strip()
    if not s or len(s) > MAX_BLUEPRINT_STRING_LEN:
        raise BlueprintIRError("blueprint string is empty or too long")
    if s[0] != "0":
        raise BlueprintIRError(f"unsupported blueprint string version byte '{s[0]}'")
    try:
        compressed = base64.b64decode(s[1:], validate=False)
        decomp = zlib.decompressobj()
        raw = decomp.decompress(compressed, MAX_DECOMPRESSED_BYTES)
        if decomp.unconsumed_tail:
            raise BlueprintIRError("decompressed blueprint exceeds size limit")
        return json.loads(raw)
    except BlueprintIRError:
        raise
    except Exception as e:
        raise BlueprintIRError(f"cannot decode blueprint string: {e}") from e


def _decode_version(version: Optional[int]) -> Tuple[int, int]:
    if not version:
        return (0, 0)
    v = int(version)
    return ((v >> 48) & 0xFFFF, (v >> 32) & 0xFFFF)


def iter_blueprints(data: Dict[str, Any], _path: str = "", _version: Optional[int] = None) -> Iterator[Tuple[str, Dict[str, Any], Tuple[int, int]]]:
    """Yields (book_path, blueprint_dict, (major, minor)) for a blueprint or (nested) book."""
    if "blueprint" in data:
        bp = data["blueprint"]
        yield _path, bp, _decode_version(bp.get("version") or _version)
    elif "blueprint_book" in data:
        book = data["blueprint_book"]
        version = book.get("version") or _version
        for i, child in enumerate(book.get("blueprints") or []):
            idx = child.get("index", i) if isinstance(child, dict) else i
            child_path = f"{_path}/{idx}" if _path else str(idx)
            if isinstance(child, dict):
                yield from iter_blueprints(child, child_path, version)


# --- Parsing ------------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    """Single-line printable text (labels come from untrusted sources)."""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value or ""))
    return " ".join(text.split())[:200]


def _signal_id(signal: Dict[str, Any]) -> str:
    """'type:name' of a label signal; Space Age signal types and quality levels are rejected."""
    kind = str(signal.get("type") or "item")
    name = str(signal.get("name") or "")
    if kind not in LABEL_SIGNAL_TYPES or signal.get("quality") not in (None, "normal"):
        raise BlueprintIRError(f"unsupported label signal {kind}:{name}")
    return f"{kind}:{name}"


def _label_signals(raw: Dict[str, Any]) -> str:
    """Constant combinator signals as 'type:name=count;...' (1.x filters or 2.0 sections)."""
    behavior = raw.get("control_behavior") or {}
    found: List[Tuple[int, int, str, int]] = []
    for f in behavior.get("filters") or []:  # 1.x
        if isinstance(f, dict) and isinstance(f.get("signal"), dict) and f["signal"].get("name"):
            found.append((0, int(f.get("index", 0)), _signal_id(f["signal"]), int(f.get("count", 0))))
    for section in (behavior.get("sections") or {}).get("sections") or []:  # 2.0
        for f in section.get("filters") or []:
            if isinstance(f, dict) and f.get("name"):
                found.append((int(section.get("index", 0)), int(f.get("index", 0)), _signal_id(f), int(f.get("count", 0))))
    if len(found) > MAX_LABEL_SIGNALS:
        raise BlueprintIRError(f"constant combinator with {len(found)} signals")
    return ";".join(f"{sid}={count}" for _, _, sid, count in sorted(found))


def _convert_extras(raw: Dict[str, Any], name: str = "") -> Tuple[Tuple[str, str], ...]:
    extras: Dict[str, str] = {}
    if name == "rocket-silo":
        raw = {k: v for k, v in raw.items() if k != "recipe"}  # its only recipe; draftsman does not export it
    if name == "constant-combinator":
        signals = _label_signals(raw)
        if signals:
            extras["signals"] = signals
    elif name == "display-panel":
        text = _clean_text(raw.get("text"))[:80]
        if text:
            extras["text"] = text
        if isinstance(raw.get("icon"), dict) and raw["icon"].get("name"):
            extras["icon"] = _signal_id(raw["icon"])
    if raw.get("recipe"):
        extras["recipe"] = str(raw["recipe"])
        if not _recipe_exists(extras["recipe"]):  # e.g. recipes removed or renamed since 0.x / 1.x
            raise BlueprintIRError(f"unknown recipe '{extras['recipe']}'")
    if raw.get("type") == "output":  # "input" is the default and is omitted on export
        extras["io_type"] = "output"
    for key in ("input_priority", "output_priority"):
        if raw.get(key) in ("left", "right"):
            extras[key] = raw[key]
    flt = raw.get("filter")
    if isinstance(flt, dict):
        flt = flt.get("name")
    if flt:
        extras["filter"] = str(flt)
    for key, value in extras.items():
        allowed = "-_:;=" if key in ("signals", "icon") else "-_"
        if key != "text" and not all(ch.isalnum() or ch in allowed for ch in value):  # text: free, emitted via repr
            raise BlueprintIRError(f"suspicious value for {key}: {value!r}")
    return tuple(sorted(extras.items()))


def _circuit_wired(bp: Dict[str, Any]) -> set:
    """entity_numbers with a red/green circuit wire (2.0 'wires' or 1.x 'connections')."""
    wired = set()
    for w in bp.get("wires") or []:
        if isinstance(w, (list, tuple)) and len(w) == 4 and w[1] in (1, 2, 3, 4) and w[3] in (1, 2, 3, 4):
            wired.update((w[0], w[2]))
    for raw in bp.get("entities") or []:
        for point in (raw.get("connections") or {}).values():
            if isinstance(point, dict) and (point.get("red") or point.get("green")):
                wired.add(raw.get("entity_number"))
    return wired


def _snap_fields(bp: Dict[str, Any]) -> Dict[str, Any]:
    grid = bp.get("snap-to-grid")
    if not isinstance(grid, dict):
        return {}
    size = (int(grid.get("x", 0)), int(grid.get("y", 0)))
    if min(size) < 1 or max(size) > 10_000:
        raise BlueprintIRError(f"invalid snap-to-grid {grid}")
    absolute = bool(bp.get("absolute-snapping", False))
    offset = bp.get("position-relative-to-grid") or {}
    return {"snap": size, "absolute_snap": absolute,
            "snap_offset": (int(offset.get("x", 0)), int(offset.get("y", 0))) if absolute else (0, 0)}


def parse_blueprint_dict(
    bp: Dict[str, Any],
    version: Tuple[int, int] = (2, 0),
    book_path: str = "",
    allowed: Optional[frozenset] = VANILLA_ENTITIES,
    normalize: bool = True,
    check_collisions: bool = True,
    snap_off_grid: bool = False,
    stats: Optional[Dict[str, int]] = None,
) -> IRBlueprint:
    """
    Converts a raw blueprint JSON dict into an IRBlueprint on the integer tile grid.
    Raises BlueprintIRError for unknown/disallowed entities, for off-grid positions (unless
    snap_off_grid=True: they are snapped to the tile containing them, as the game does, and
    counted in stats["off_grid"]), and for overlapping entities unless check_collisions=False.
    """
    legacy = version[0] < 2
    entities: List[IREntity] = []
    placed: List[Tuple[str, int, Tuple[Tuple[str, str], ...], Dict[str, Any], float, float]] = []
    wired = _circuit_wired(bp)
    for raw in bp.get("entities") or []:
        name = str(raw.get("name", ""))
        if legacy:
            name = LEGACY_RENAMES.get(name, name)
        if _prototype(name) is None:
            raise BlueprintIRError(f"unknown entity '{name}'")
        if allowed is not None and name not in allowed:
            raise BlueprintIRError(f"entity '{name}' not allowed")
        if name in ANNOTATION_ENTITIES and raw.get("entity_number") in wired:
            raise BlueprintIRError(f"circuit-wired '{name}' (circuit logic) is not supported")

        direction = int(raw.get("direction", 0) or 0)
        if legacy:
            direction *= 2  # 1.x used 8 directions, 2.0 uses 16
        if direction % 4 != 0:
            raise BlueprintIRError(f"non-cardinal direction {direction} on '{name}'")
        extras = _convert_extras(raw, name)
        if not is_directional(name, dict(extras).get("recipe")):
            direction = 0

        pos = raw.get("position") or {}
        _, _, cdx, cdy = _geometry(name, direction)
        placed.append((name, direction, extras, pos, float(pos.get("x", 0.0)) - cdx, float(pos.get("y", 0.0)) - cdy))

    # Strings older than 1.0 often place the whole blueprint half a tile off the grid; a shift shared
    # by every entity is only a different origin, so it is removed (mixed offsets stay an error).
    shift = (0.0, 0.0)
    if version < (1, 0) and placed:
        fractions = {(round(tx % 1, 6), round(ty % 1, 6)) for _, _, _, _, tx, ty in placed}
        if len(fractions) == 1 and set(next(iter(fractions))) <= {0.0, 0.5}:
            shift = next(iter(fractions))

    for name, direction, extras, pos, tx, ty in placed:
        tx, ty = tx - shift[0], ty - shift[1]
        if abs(tx - round(tx)) > 1e-6 or abs(ty - round(ty)) > 1e-6:
            if not snap_off_grid:
                raise BlueprintIRError(f"'{name}' at {pos} is off the tile grid")
            if stats is not None:
                stats["off_grid"] = stats.get("off_grid", 0) + 1
            tx, ty = math.floor(tx + 0.5 + 1e-9), math.floor(ty + 0.5 + 1e-9)
        entities.append(IREntity(name, int(round(tx)), int(round(ty)), direction, extras))

    ir = IRBlueprint(entities=entities, label=_clean_text(bp.get("label")), version=version, book_path=book_path,
                     **_snap_fields(bp))
    if normalize:
        ir = normalized(ir)
    if check_collisions:
        _check_collisions(ir)
    return ir


def parse_blueprint_string(blueprint_string: str, **kwargs) -> IRBlueprint:
    """Parses a single-blueprint string (books raise; use iter_blueprints for those)."""
    data = decode_blueprint_string(blueprint_string)
    items = list(iter_blueprints(data))
    if "blueprint" not in data or len(items) != 1:
        raise BlueprintIRError("expected a single blueprint, got a book or other blueprintable")
    path, bp, version = items[0]
    return parse_blueprint_dict(bp, version=version, book_path=path, **kwargs)


def _check_collisions(ir: IRBlueprint) -> None:
    occupied: Dict[Tuple[int, int], str] = {}
    for e in ir.entities:
        w, h = footprint(e.name, e.direction)
        for dx in range(w):
            for dy in range(h):
                tile = (e.x + dx, e.y + dy)
                if tile in occupied:
                    raise BlueprintIRError(f"'{e.name}' overlaps '{occupied[tile]}' at {tile}")
                occupied[tile] = e.name


# --- Transformations ----------------------------------------------------------------------

def normalized(ir: IRBlueprint) -> IRBlueprint:
    """Shifts entities by an integer offset so the minimal tile is (0, 0); sorts deterministically."""
    if not ir.entities:
        return ir
    min_x = min(e.x for e in ir.entities)
    min_y = min(e.y for e in ir.entities)
    if ir.snap:  # keep entities where they are relative to the grid cells: shift by whole cells only
        min_x -= min_x % ir.snap[0]
        min_y -= min_y % ir.snap[1]
    shifted = sorted(
        (IREntity(e.name, e.x - min_x, e.y - min_y, e.direction, e.extras) for e in ir.entities),
        key=lambda e: (e.y, e.x, e.name),
    )
    return ir.with_entities(shifted)


def rotated(ir: IRBlueprint, quarter_turns: int) -> IRBlueprint:
    """Rotates the blueprint clockwise by 90 degrees * quarter_turns (as Factorio does)."""
    k = quarter_turns % 4
    if k == 0:
        return normalized(ir)
    out: List[IREntity] = []
    for e in ir.entities:
        _, _, cdx, cdy = _geometry(e.name, e.direction)
        cx, cy = e.x + cdx, e.y + cdy
        direction = e.direction
        for _ in range(k):
            cx, cy = -cy, cx  # clockwise in screen coordinates (y points down)
            if e.directional:
                direction = (direction + 4) % 16
        _, _, ndx, ndy = _geometry(e.name, direction)
        out.append(IREntity(e.name, int(round(cx - ndx)), int(round(cy - ndy)), direction, e.extras))
    if ir.absolute_snap:
        raise BlueprintIRError("absolute-snapped blueprints are tied to the world grid and are not rotated")
    snap = (ir.snap[1], ir.snap[0]) if ir.snap and k % 2 else ir.snap
    return normalized(ir.with_entities(out, snap))


# --- Facts --------------------------------------------------------------------------------

BELT_TIERS = {
    "transport-belt": "yellow", "underground-belt": "yellow", "splitter": "yellow",
    "fast-transport-belt": "red", "fast-underground-belt": "red", "fast-splitter": "red",
    "express-transport-belt": "blue", "express-underground-belt": "blue", "express-splitter": "blue",
}
_DIRECTION_NAMES = {0: "north", 4: "east", 8: "south", 12: "west"}


def direction_name(direction: int) -> str:
    return _DIRECTION_NAMES.get(direction, str(direction))


def extract_facts(ir: IRBlueprint) -> Dict[str, Any]:
    """Programmatic facts about a blueprint, used to ground generated prompts."""
    counts: Dict[str, int] = {}
    recipes: Dict[str, int] = {}
    belt_dirs: Dict[str, int] = {}
    tiers: Dict[str, int] = {}
    for e in ir.entities:
        counts[e.name] = counts.get(e.name, 0) + 1
        recipe = e.extras_dict.get("recipe")
        if recipe:
            recipes[recipe] = recipes.get(recipe, 0) + 1
        if e.name in BELT_TIERS:
            tiers[BELT_TIERS[e.name]] = tiers.get(BELT_TIERS[e.name], 0) + 1
        if entity_type(e.name) == "transport-belt":
            d = direction_name(e.direction)
            belt_dirs[d] = belt_dirs.get(d, 0) + 1

    by_type: Dict[str, int] = {}
    for name, n in counts.items():
        t = entity_type(name)
        by_type[t] = by_type.get(t, 0) + n

    width, height = ir.size
    label_signals = sorted({sid.split("=")[0].split(":", 1)[1] for e in ir.entities
                            for sid in (e.extras_dict.get("signals") or "").split(";") if sid}
                           | {e.extras_dict["icon"].split(":", 1)[1] for e in ir.entities if "icon" in e.extras_dict})
    return {
        "snap": list(ir.snap) if ir.snap else None,
        "absolute_snapping": ir.absolute_snap,
        "snap_offset": list(ir.snap_offset) if ir.absolute_snap else None,
        "label_signals": label_signals[:12],
        "entity_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "type_counts": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "recipes": recipes,
        "belt_tiers": tiers,
        "main_belt_direction": max(belt_dirs, key=belt_dirs.get) if belt_dirs else None,
        "width": width,
        "height": height,
        "total_entities": len(ir.entities),
    }
