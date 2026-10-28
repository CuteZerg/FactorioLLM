"""
Base-game data layer: recipes with amounts, item / fluid kinds and crafting times, and what entities
can do - the properties the simulator and the validator use instead of lists of names.

Built from `base_game.py`, which tools/extract_game_data.py generates from the base mod of
wube/factorio-data (draftsman's own data includes Space Age: there the personal roboport MK2 needs
superconductors). A mod or Space Age would come in the same way, through its prototypes.

Entity properties come from the prototype: the energy source (a burner burns the fuel of its fuel
categories, some keep the burnt result), crafting categories, the inputs of a lab, the ammunition
categories of a turret (or of its gun). Furnaces have no recipe in a blueprint: a furnace smelts
whatever is put into it; `SMELTING` maps each input of a smelting recipe to its product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Tuple

from factoriollm.base_game import ENTITIES, ITEM_PROPERTIES, RECIPES


def _smelting() -> Dict[str, str]:
    """Input -> product of every smelting recipe (one solid ingredient, one result)."""
    out = {}
    for data in RECIPES.values():
        if data["category"] == "smelting" and len(data["ingredients"]) == 1 and len(data["results"]) == 1:
            out[next(iter(data["ingredients"]))] = next(iter(data["results"]))
    return out


# Smelting input -> product (iron ore -> plate, iron plate -> steel, ...), from the smelting recipes.
SMELTING = _smelting()


@dataclass(frozen=True)
class Recipe:
    name: str
    ingredients: Dict[str, float]
    results: Dict[str, float]  # expected amounts (probability included)
    fluids: FrozenSet[str]     # ingredients and results that are fluids
    time: float
    category: str
    # fluid -> 1-based index among the machine's input / output fluid boxes, where the recipe says so
    box_in: Dict[str, int] = field(default_factory=dict)
    box_out: Dict[str, int] = field(default_factory=dict)


@lru_cache(maxsize=None)
def recipe(name: str) -> Optional[Recipe]:
    """A base-game recipe, or None for recipes outside the base game."""
    data = RECIPES.get(name)
    if data is None:
        return None
    boxes = data.get("fluid_boxes") or {}
    return Recipe(name=name, ingredients=dict(data["ingredients"]), results=dict(data["results"]),
                  fluids=frozenset(data["fluids"]), time=float(data["time"]), category=data["category"],
                  box_in=dict(boxes.get("in") or {}), box_out=dict(boxes.get("out") or {}))


@lru_cache(maxsize=None)
def places_entity(item: str) -> bool:
    """Whether the item is a building / placeable thing (belts, robots, panels) rather than a material."""
    if item in ITEM_PROPERTIES:
        return "places" in ITEM_PROPERTIES[item]
    from draftsman.data import items
    return bool((items.raw.get(item) or {}).get("place_result"))


# --- What entities can do -------------------------------------------------------------------------

def entity(name: str) -> Dict[str, Any]:
    """The base-game properties of a buildable entity ({} for anything else)."""
    return ENTITIES.get(name) or {}


def is_burner(name: str) -> bool:
    """Whether the entity runs on fuel it burns (boilers, stone furnaces, reactors, burner drills ...)."""
    return entity(name).get("energy") == "burner"


@lru_cache(maxsize=None)
def items_of(key: str, values: Tuple[str, ...], item_type: Optional[str] = None) -> FrozenSet[str]:
    """Items whose property `key` is one of `values` (and of that prototype type)."""
    return frozenset(n for n, p in ITEM_PROPERTIES.items()
                     if p.get(key) in values and (item_type is None or p.get("type") == item_type))


def fuel_items(name: str) -> FrozenSet[str]:
    """What a burner entity burns: the items of its fuel categories (nothing for other entities)."""
    return items_of("fuel_category", tuple(entity(name).get("fuel_categories") or ())) if is_burner(name) else frozenset()


def burnt_result(item: str) -> Optional[str]:
    """What burning the item leaves (a depleted fuel cell), if anything."""
    return (ITEM_PROPERTIES.get(item) or {}).get("burnt_result")


def keeps_burnt_result(name: str) -> bool:
    """Whether the entity has a slot for burnt results an inserter can take out (a reactor)."""
    return bool(entity(name).get("burnt_inventory"))


def lab_inputs(name: str) -> Tuple[str, ...]:
    """The science packs a lab takes."""
    return tuple(entity(name).get("lab_inputs") or ())


def ammo_items(name: str) -> FrozenSet[str]:
    """The ammunition a turret takes: items of type ammo in the categories of the turret or its gun."""
    return items_of("ammo_category", tuple(entity(name).get("ammo_categories") or ()), "ammo")


def turret_fluids(name: str) -> Tuple[str, ...]:
    """The fluids a fluid turret (flamethrower) fires."""
    return tuple(entity(name).get("fluids") or ())


@lru_cache(maxsize=None)
def craftable_inputs(name: str) -> FrozenSet[str]:
    """Solid ingredients of every recipe the machine can craft (a furnace: whatever can be smelted)."""
    cats = set(entity(name).get("crafting_categories") or ())
    return frozenset(i for data in RECIPES.values() if data["category"] in cats
                     for i in data["ingredients"] if i not in data["fluids"])


@lru_cache(maxsize=None)
def all_fuels() -> FrozenSet[str]:
    """Every item some burner can burn."""
    return frozenset(n for n, p in ITEM_PROPERTIES.items() if "fuel_category" in p)


@lru_cache(maxsize=None)
def all_ammo() -> FrozenSet[str]:
    """Every item of type ammo."""
    return frozenset(n for n, p in ITEM_PROPERTIES.items() if p.get("type") == "ammo" and "ammo_category" in p)


@lru_cache(maxsize=None)
def is_storage(name: str) -> bool:
    """
    Whether inserters may put anything into the entity: an inventory that is not a turret's
    magazine (chests, wagons, cars). Entities outside the base game: their draftsman prototype.
    """
    e = entity(name)
    if e:
        return "inventory" in e and not e.get("ammo_categories")
    from factoriollm.blueprint_ir import _prototype
    proto = _prototype(name) or {}
    return bool(proto.get("inventory_size")) and "attack_parameters" not in proto


def inserter_rate(name: str) -> float:
    """Rough items per second of an inserter between a belt and a machine (for ranking only): swings, x2 for bulk."""
    e = entity(name)
    return float(e.get("rotation_speed") or 0.014) * 60 * (2 if e.get("bulk") else 1)


@lru_cache(maxsize=None)
def made_by_recipe(item: str) -> bool:
    """Whether some recipe makes the item (the space science pack comes from rocket launches)."""
    return any(item in data["results"] for data in RECIPES.values())


def robot_items() -> FrozenSet[str]:
    """What a roboport holds: repair packs and the items that place robots."""
    return frozenset(n for n, p in ITEM_PROPERTIES.items()
                     if p.get("type") == "repair-tool" or str(p.get("places", "")).endswith("-robot"))


def final_products(recipes: Iterable[str]) -> Set[str]:
    """
    What a set of machines delivers outside: everything their recipes produce minus what the same
    machines consume (copper cable inside a green circuit block is an intermediate, not a product).
    """
    produced: Set[str] = set()
    consumed: Set[str] = set()
    for name in set(recipes):
        r = recipe(name)
        if r is None:
            continue
        produced |= set(r.results)
        consumed |= set(r.ingredients)
    return produced - consumed
