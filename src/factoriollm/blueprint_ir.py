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
})

# Vanilla, but excluded for now: the decompiler does not emit wires / combinator conditions /
# train schedules, so these blueprints would teach prompts that the code cannot fulfil.
RAIL_AND_CIRCUIT_ENTITIES = frozenset({
    "straight-rail", "curved-rail-a", "curved-rail-b", "half-diagonal-rail",
    "legacy-straight-rail", "legacy-curved-rail",
    "rail-signal", "rail-chain-signal", "train-stop",
    "locomotive", "cargo-wagon", "fluid-wagon", "artillery-wagon",
    "arithmetic-combinator", "decider-combinator", "constant-combinator", "selector-combinator",
    "power-switch", "programmable-speaker", "display-panel",
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
EXTRA_KEYS = ("recipe", "io_type", "input_priority", "output_priority", "filter")

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

    @property
    def size(self) -> Tuple[int, int]:
        if not self.entities:
            return (0, 0)
        w = max(e.x + footprint(e.name, e.direction)[0] for e in self.entities)
        h = max(e.y + footprint(e.name, e.direction)[1] for e in self.entities)
        return (w, h)

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


def _convert_extras(raw: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    extras: Dict[str, str] = {}
    if raw.get("recipe"):
        extras["recipe"] = str(raw["recipe"])
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
        if not all(ch.isalnum() or ch in "-_" for ch in value):
            raise BlueprintIRError(f"suspicious value for {key}: {value!r}")
    return tuple(sorted(extras.items()))


def parse_blueprint_dict(
    bp: Dict[str, Any],
    version: Tuple[int, int] = (2, 0),
    book_path: str = "",
    allowed: Optional[frozenset] = VANILLA_ENTITIES,
    normalize: bool = True,
) -> IRBlueprint:
    """
    Converts a raw blueprint JSON dict into an IRBlueprint on the integer tile grid.
    Raises BlueprintIRError for unknown/disallowed entities or off-grid positions.
    """
    legacy = version[0] < 2
    entities: List[IREntity] = []
    for raw in bp.get("entities") or []:
        name = str(raw.get("name", ""))
        if legacy:
            name = LEGACY_RENAMES.get(name, name)
        if _prototype(name) is None:
            raise BlueprintIRError(f"unknown entity '{name}'")
        if allowed is not None and name not in allowed:
            raise BlueprintIRError(f"entity '{name}' not allowed")

        direction = int(raw.get("direction", 0) or 0)
        if legacy:
            direction *= 2  # 1.x used 8 directions, 2.0 uses 16
        if direction % 4 != 0:
            raise BlueprintIRError(f"non-cardinal direction {direction} on '{name}'")
        extras = _convert_extras(raw)
        if not is_directional(name, dict(extras).get("recipe")):
            direction = 0

        pos = raw.get("position") or {}
        _, _, cdx, cdy = _geometry(name, direction)
        tx, ty = float(pos.get("x", 0.0)) - cdx, float(pos.get("y", 0.0)) - cdy
        if abs(tx - round(tx)) > 1e-6 or abs(ty - round(ty)) > 1e-6:
            raise BlueprintIRError(f"'{name}' at {pos} is off the tile grid")
        entities.append(IREntity(name, int(round(tx)), int(round(ty)), direction, extras))

    ir = IRBlueprint(entities=entities, label=_clean_text(bp.get("label")), version=version, book_path=book_path)
    if normalize:
        ir = normalized(ir)
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
    shifted = sorted(
        (IREntity(e.name, e.x - min_x, e.y - min_y, e.direction, e.extras) for e in ir.entities),
        key=lambda e: (e.y, e.x, e.name),
    )
    return IRBlueprint(shifted, ir.label, ir.version, ir.book_path)


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
    return normalized(IRBlueprint(out, ir.label, ir.version, ir.book_path))


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
    return {
        "entity_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "type_counts": dict(sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))),
        "recipes": recipes,
        "belt_tiers": tiers,
        "main_belt_direction": max(belt_dirs, key=belt_dirs.get) if belt_dirs else None,
        "width": width,
        "height": height,
        "total_entities": len(ir.entities),
    }
