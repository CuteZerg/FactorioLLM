"""
Base-game data layer: recipes with amounts, item / fluid kinds and crafting times.

Built from draftsman's prototype data (which includes Space Age) filtered to the Factorio 2.0 base
game by `VANILLA_RECIPES`. draftsman's own `get_recipe_ingredients` is not used: it returns names
only, without amounts and without telling items from fluids.

Furnaces have no recipe in a blueprint: a furnace smelts whatever is put into it. `SMELTING` maps
each smelting input to its product.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Optional, Set

from factoriollm.vanilla_recipes import VANILLA_RECIPES

# Smelting input -> product. Steel takes iron plates (5 per steel, 16 s); everything else takes ore or stone.
SMELTING = {"iron-ore": "iron-plate", "copper-ore": "copper-plate", "stone": "stone-brick", "iron-plate": "steel-plate"}
DEFAULT_CRAFTING_TIME = 0.5  # draftsman leaves energy_required empty when the game default applies


@dataclass(frozen=True)
class Recipe:
    name: str
    ingredients: Dict[str, float]
    results: Dict[str, float]  # expected amounts (probability included)
    fluids: FrozenSet[str]     # ingredients and results that are fluids
    time: float
    category: str


def _amount(part: dict) -> float:
    amount = part.get("amount")
    if amount is None:  # ranged results
        amount = (part.get("amount_min", 0) + part.get("amount_max", 0)) / 2
    return float(amount) * float(part.get("probability", 1))


@lru_cache(maxsize=None)
def recipe(name: str) -> Optional[Recipe]:
    """A base-game recipe, or None for recipes outside the base game."""
    if name not in VANILLA_RECIPES:
        return None
    from draftsman.data import recipes
    data = recipes.raw.get(name)
    if data is None:
        return None
    ingredients = [p for p in data.get("ingredients") or [] if isinstance(p, dict)]
    results = [p for p in data.get("results") or [] if isinstance(p, dict)]
    return Recipe(
        name=name,
        ingredients={p["name"]: _amount(p) for p in ingredients},
        results={p["name"]: _amount(p) for p in results},
        fluids=frozenset(p["name"] for p in ingredients + results if p.get("type") == "fluid"),
        time=float(data.get("energy_required") or DEFAULT_CRAFTING_TIME),
        category=str(data.get("category") or "crafting"),
    )


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
