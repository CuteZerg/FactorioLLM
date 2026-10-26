"""
Base-game data layer: recipes with amounts, item / fluid kinds and crafting times.

Built from `base_game.py`, which tools/extract_game_data.py generates from the base mod of
wube/factorio-data (draftsman's own data includes Space Age: there the personal roboport MK2 needs
superconductors).

Furnaces have no recipe in a blueprint: a furnace smelts whatever is put into it. `SMELTING` maps
each smelting input to its product.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Optional, Set

from factoriollm.base_game import FLUIDS, ITEMS, RECIPES

# Smelting input -> product. Steel takes iron plates (5 per steel, 16 s); everything else takes ore or stone.
SMELTING = {"iron-ore": "iron-plate", "copper-ore": "copper-plate", "stone": "stone-brick", "iron-plate": "steel-plate"}


@dataclass(frozen=True)
class Recipe:
    name: str
    ingredients: Dict[str, float]
    results: Dict[str, float]  # expected amounts (probability included)
    fluids: FrozenSet[str]     # ingredients and results that are fluids
    time: float
    category: str


@lru_cache(maxsize=None)
def recipe(name: str) -> Optional[Recipe]:
    """A base-game recipe, or None for recipes outside the base game."""
    data = RECIPES.get(name)
    if data is None:
        return None
    return Recipe(name=name, ingredients=dict(data["ingredients"]), results=dict(data["results"]),
                  fluids=frozenset(data["fluids"]), time=float(data["time"]), category=data["category"])


def is_item(name: str) -> bool:
    return name in ITEMS


def is_fluid(name: str) -> bool:
    return name in FLUIDS


@lru_cache(maxsize=None)
def places_entity(item: str) -> bool:
    """Whether the item is a building / placeable thing (belts, robots, panels) rather than a material."""
    from draftsman.data import items
    return bool((items.raw.get(item) or {}).get("place_result"))


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
