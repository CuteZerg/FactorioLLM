"""
Template miner: finds the best tileable player designs per category in the scraped corpus.

For every blueprint (and every child of a blueprint book) with enough favorites:
    raw JSON -> IR -> category by content -> the design must work on its own (flow simulator) ->
    cut of the author's surroundings (by flow; the cut must keep it working) -> template (repeating
    segment with head and tail, or a whole cell tiled by copies) -> tiling check with several
    lengths -> builds must work too -> metrics
The best templates per category go to data/templates/catalog.json, and a readable report with
ASCII maps to outputs/templates/catalog.md. Rejected designs are counted by reason, and a few per
reason go to a second book (outputs/templates/rejects_book.txt) to check the reasons in game.

    uv run python -m factoriollm.template_miner [--min-favorites 10] [--per-category 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from factoriollm.blueprint_ir import (
    BlueprintIRError,
    IRBlueprint,
    IREntity,
    decode_blueprint_string,
    entity_type,
    footprint,
    iter_blueprints,
    parse_blueprint_dict,
    rotated,
)
from factoriollm.flow_sim import FlowResult, solve
from factoriollm.game_data import final_products, places_entity, recipe
from factoriollm.templates import (
    CROP_STEPS,
    DEFENSE,
    Template,
    add_port_labels,
    along_x,
    extract,
    extract_cell,
    has_weaving,
    main_belt,
    metrics,
    normalize_modules,
    prepare,
    strip_unused,
    repair_power,
    signature,
    tiling_check,
)

CATALOG = Path("data/templates/catalog.json")
STATS = Path("data/templates/miner_stats.json")
REPORT = Path("outputs/templates/catalog.md")
REJECTS_BOOK = Path("outputs/templates/rejects_book.txt")
REJECTS_PER_REASON = 5
MAX_ENTITIES = 3000

ASSEMBLERS = ("assembling-machine-1", "assembling-machine-2", "assembling-machine-3")
FURNACES = {"stone-furnace": "smelting_stone", "steel-furnace": "smelting_steel", "electric-furnace": "smelting_electric"}
SCIENCE = {"automation-science-pack": "science_red", "logistic-science-pack": "science_green",
           "military-science-pack": "science_military", "chemical-science-pack": "science_blue",
           "production-science-pack": "science_purple", "utility-science-pack": "science_yellow"}
# Machines with a recipe in the blueprint (furnaces pick theirs from the input).
RECIPE_MACHINES = {*ASSEMBLERS, "chemical-plant", "oil-refinery", "centrifuge"}
# Single final product -> category. Products that need a layout of their own: circuits (direct
# insertion, lane tricks), plastic, science. Every other product goes to a family by the shape of
# its recipe (see recipe_family): gears, pipes, rods are instances of "1 solid input".
PRODUCT_CATEGORIES = {"electronic-circuit": "green_circuits", "advanced-circuit": "red_circuits",
                      "processing-unit": "blue_circuits", "plastic-bar": "plastic", **SCIENCE}
MULTI_MAX_PRODUCTS = 3  # more final products than this is a mall: no tileable segment, not mined
FLUID_ENTITIES = {"offshore-pump", "pump", "pipe", "pipe-to-ground", "storage-tank"}


# Machines that make up each category; a module must consist mostly of them.
FAMILIES = {
    "defense": set(DEFENSE),
    "nuclear": {"nuclear-reactor", "heat-exchanger", "steam-turbine"},
    "steam": {"boiler", "steam-engine"},
    "solar": {"solar-panel", "accumulator"},
    "labs": {"lab"},
    "mining": {"electric-mining-drill"},
    **{cat: {name} for name, cat in FURNACES.items()},
}
PRODUCTION = {"nuclear-reactor", "heat-exchanger", "steam-turbine", "boiler", "steam-engine", "solar-panel",
              "accumulator", "lab", "electric-mining-drill", "burner-mining-drill", "stone-furnace", "steel-furnace",
              "electric-furnace", *RECIPE_MACHINES}
MIN_PURITY = 0.6
# Power and mining: share of the blueprint's footprint taken by the category's own entities
# (a pump station with a few panels is not a solar field).
AREA_PURITY = {"nuclear": 0.4, "steam": 0.5, "solar": 0.5, "mining": 0.5, "defense": 0.5}
AREA_FAMILIES = {**FAMILIES, "mining": {"electric-mining-drill", "beacon"}}  # beaconed mining is still mining
FAMILY_CATEGORIES = ("assembly_1in", "assembly_2in", "assembly_3in", "assembly_fluid", "chemistry", "oil", "uranium")
TURRET_KINDS = {"gun-turret": "gun", "laser-turret": "laser", "flamethrower-turret": "flame", "artillery-turret": "artillery"}


def products(ir: IRBlueprint) -> List[str]:
    """Final products of the blueprint's recipe machines (what leaves the block), sorted."""
    return sorted(final_products(e.extras_dict["recipe"] for e in ir.entities
                                 if e.name in RECIPE_MACHINES and e.extras_dict.get("recipe")))


def is_mall(final: List[str]) -> bool:
    """Many products, or several buildings (a belt or robot mall), make a mall rather than a production block."""
    return len(final) > MULTI_MAX_PRODUCTS or sum(map(places_entity, final)) >= 2


def area_share(ir: IRBlueprint, family: set) -> float:
    """Share of the summed entity footprint taken by entities of the family."""
    areas = Counter()
    for e in ir.entities:
        w, h = footprint(e.name, e.direction)
        areas[e.name in family] += w * h
    return areas[True] / max(areas[True] + areas[False], 1)


def recipe_family(product: str) -> Optional[str]:
    """
    Family of a single-product block by the shape of the product's recipe: chemistry (chemical
    plants) and oil (refineries) by their machines, else assembly_<n>in by the number of solid
    ingredients (1-3) or assembly_fluid with a fluid ingredient.
    """
    for name in (product, *(n for n in _producers(product))):
        r = recipe(name)
        if r is None:
            continue
        if r.category == "chemistry":
            return "chemistry"
        if r.category == "oil-processing":
            return "oil"
        if r.category == "centrifuging":
            return "uranium"
        if r.fluids & set(r.ingredients):
            return "assembly_fluid"
        solid = len(set(r.ingredients) - r.fluids)
        return f"assembly_{min(max(solid, 1), 3)}in"
    return None


def _producers(product: str) -> List[str]:
    from factoriollm.base_game import RECIPES
    return [name for name, r in RECIPES.items() if product in r["results"]]


def _category(names: Counter, ir: IRBlueprint) -> Optional[str]:
    furnaces = {k: names[k] for k in FURNACES}
    turrets = sum(n for name, n in names.items() if name in DEFENSE and name.endswith("turret"))
    if (turrets or names["stone-wall"] >= 8) and area_share(ir, DEFENSE) >= AREA_PURITY["defense"] \
            and area_share(ir, PRODUCTION) <= 0.1:  # not a production block behind its walls
        return "defense"
    if names["nuclear-reactor"]:
        return "nuclear"
    if names["boiler"] and names["steam-engine"] >= 2:
        return "steam"
    if names["solar-panel"] >= 4 and not any(names[n] for n in FLUID_ENTITIES):  # panels powering a pump station
        return "solar"
    if names["lab"] >= 4:
        return "labs"
    if sum(furnaces.values()) >= 4:
        return FURNACES[max(furnaces, key=furnaces.get)]
    if names["electric-mining-drill"] >= 4:
        return "mining"
    if sum(names[m] for m in RECIPE_MACHINES) >= 2:
        final = products(ir)
        if len(final) == 1:
            return PRODUCT_CATEGORIES.get(final[0]) or recipe_family(final[0])
        if len(final) >= 2 and not is_mall(final):
            return "multi"
    return None


def classify(ir: IRBlueprint) -> Optional[str]:
    """
    Main purpose of a blueprint from what it delivers (not from author tags):
    - power and mining by their machines, with most of the footprint taken by them;
    - production blocks by their final products (recipe graph): one product gives its category,
      2-3 products a 'multi' block, more is a mall and is not mined;
    - blocks that mix purposes (a science block with its own smelting) are skipped: the category's
      own machines must make up at least MIN_PURITY of all production machines.
    """
    names = Counter(e.name for e in ir.entities)
    category = _category(names, ir)
    if category is None or category == "defense":
        return category
    family = FAMILIES.get(category, RECIPE_MACHINES)
    if category in AREA_PURITY and area_share(ir, AREA_FAMILIES[category]) < AREA_PURITY[category]:
        return None
    production = sum(n for name, n in names.items() if name in PRODUCTION)
    own = sum(names[name] for name in family)
    return category if production and own / production >= MIN_PURITY else None


def main_machine(t: Template) -> str:
    machines = t.metrics.get("machines_per_segment") or {}
    family = FAMILIES.get(t.category, set(ASSEMBLERS))
    own = {k: v for k, v in machines.items() if k in family} or machines
    return max(own, key=own.get) if own else "-"


def _belts_along_x(t: Template) -> int:
    return sum(1 for e in t.segment if e.name.endswith("transport-belt") and e.direction in (4, 12))


def _docking_cell(ir: IRBlueprint, category: str, step: Tuple[int, int]) -> Template:
    """A design that docks with its copies `step` apart, as a cell repeated along x at that pitch."""
    relative = IRBlueprint(ir.entities, ir.label, ir.version, ir.book_path, ir.snap)  # a cell keeps no world grid
    oriented = relative if step[0] else rotated(relative, 1)
    template = extract_cell(oriented, category)
    template.period = step[0] or step[1]
    template.snap = (template.period, template.snap[1])
    return template


def mine_one(ir: IRBlueprint, category: str) -> Optional[Template]:
    """
    Best template of one blueprint: an inner repeat if there is one; otherwise the whole blueprint
    tiled by copies along the axis where copies join cleanly (belts continue, nothing overlaps),
    preferring the axis the belts run along.
    """
    for oriented in along_x(ir):
        template = extract(oriented, category)
        if template is not None:
            return template
    # world-grid placement does not matter for a template (modules set their own grid): allow rotation
    relative = IRBlueprint(ir.entities, ir.label, ir.version, ir.book_path, ir.snap)
    options = [relative, rotated(relative, 1)]
    cells = [cell for cell in (extract_cell(o, category) for o in options) if tiling_check(cell)["ok"]]
    # an author's grid smaller than the blueprint marks the intended (overlapping) tiling axis
    return max(cells, key=lambda c: (c.period < c.metrics_width, _belts_along_x(c))) if cells else None


def _modules(ents: List[Any]) -> Optional[str]:
    """Most common module set of these entities, e.g. 'speed-module-3=2' (None without modules)."""
    sets = Counter(e.extras_dict.get("modules") for e in ents)
    top = sets.most_common(1)[0][0] if sets else None
    return top.replace("-module", "") if top else None


def traits(t: Template) -> Dict[str, Any]:
    """
    What tells layouts of one category apart beyond tiers, so a user can ask for them and the
    catalog keeps them as separate variants:
    - pitch: main machines side by side ('tight') or with gaps along the repeat;
    - poles: in the inserter rows ('inline'), in a row of their own ('own row'), for drills
      'between drills' when they stand among the drill pairs;
    - machine / beacon modules, a chain of intermediate products, belt weaving, lamps.
    """
    main = main_machine(t)
    machines = [e for e in t.segment if e.name == main]
    widths: Dict[int, int] = {}
    for e in machines:
        widths[e.y] = widths.get(e.y, 0) + footprint(e.name, e.direction)[0]
    pole_rows = {e.y for e in t.segment if entity_type(e.name) == "electric-pole"}
    inserter_rows = {e.y for e in t.segment if entity_type(e.name) == "inserter"}
    if not pole_rows:
        poles = "none"
    elif t.category == "mining" and machines and all(
            min(e.y for e in machines) <= y < max(e.y + footprint(e.name, e.direction)[1] for e in machines) for y in pole_rows):
        poles = "between drills"
    else:
        poles = "inline" if pole_rows <= inserter_rows else "own row"
    recipes = {e.extras_dict.get("recipe") for e in t.segment + t.head + t.tail} - {None}
    if t.category == "defense":
        return defense_traits(t)
    main_recipe = Counter(e.extras_dict.get("recipe") for e in machines if e.extras_dict.get("recipe")).most_common(1)
    return {
        "recipe": main_recipe[0][0] if main_recipe else None,
        "pitch": "tight" if machines and max(widths.values()) >= t.period else "gaps",
        "poles": poles,
        "machine_modules": _modules(machines),
        "beacon_modules": _modules([e for e in t.segment if e.name == "beacon"]),
        "chain": len(recipes) > 1,
        "weaving": has_weaving(t.build(3).entities),
        "lamps": bool(t.metrics.get("lamps")),
    }


def defense_traits(t: Template) -> Dict[str, Any]:
    """Defence variants: turret kinds, wall thickness (rows of wall), gates, ammunition by belt or by hand."""
    from factoriollm.flow_sim import Layout

    kinds = sorted({TURRET_KINDS[e.name] for e in t.segment if e.name in TURRET_KINDS})
    L = Layout(t.build(2))
    fed = any(dst[0] == "entity" and L.ents[dst[1]].name == "gun-turret" for _, dst in L.links)
    return {
        "turrets": "+".join(kinds) if kinds else "walls only",
        # thickness: walls in a column across the line, the most common count
        "wall_rows": Counter(Counter(e.x for e in t.segment if e.name == "stone-wall").values()).most_common(1)[0][0]
        if any(e.name == "stone-wall" for e in t.segment) else 0,
        "gates": any(e.name == "gate" for e in t.segment),
        "ammo": ("belt" if fed else "by hand / robots") if "gun" in kinds else None,
        "lamps": bool(t.metrics.get("lamps")),
    }


def variant(t: Template) -> str:
    """
    Variant label a user can ask for: beacons (and their modules) or not, the main machine tier and
    its modules, what a furnace column smelts (ore or steel) or which products a multi-product block
    makes, machine pitch, pole placement, chains, belt weaving and lamps.
    """
    tr = t.metrics.get("traits") or {}
    if t.category == "defense":
        parts = [f"turrets {tr.get('turrets')}", f"wall {tr.get('wall_rows')} rows"]
        parts += ["gates"] if tr.get("gates") else []
        parts += [f"ammo {tr['ammo']}"] if tr.get("ammo") else []
        return ", ".join(parts)
    beaconed = (t.metrics.get("beacons_per_machine") or 0) > 0
    parts = [f"beacons {tr['beacon_modules']}" if beaconed and tr.get("beacon_modules")
             else "beacons" if beaconed else "no beacons", main_machine(t)]
    if t.category in FAMILY_CATEGORIES and tr.get("recipe"):
        parts.append(tr["recipe"])
    if tr.get("machine_modules"):
        parts.append(tr["machine_modules"])
    if t.metrics.get("input"):
        parts.append(t.metrics["input"])
    if t.metrics.get("products"):
        parts.append(" + ".join(t.metrics["products"]))
    if tr:
        parts += [f"pitch {tr['pitch']}", f"poles {tr['poles']}"]
        parts += [name for name in ("chain", "weaving", "lamps") if tr.get(name)]
    return ", ".join(parts)


def sanity(t: Template) -> Optional[str]:
    """
    Why a template is not a working module of its category (None if it is). The validator checks
    geometry; this checks that the parts and their proportions make sense, e.g. a reactor block
    without heat exchangers is only a piece of a power plant.
    """
    ir = t.build(2 if t.repeats > 1 or t.head or t.tail else 1)
    names = Counter(e.name for e in ir.entities)
    recipes = Counter(e.extras_dict.get("recipe") for e in ir.entities if e.name in ASSEMBLERS)
    cat = t.category
    if cat == "nuclear":
        if names["heat-exchanger"] < 3 * names["nuclear-reactor"] or names["steam-turbine"] < 1.5 * names["heat-exchanger"]:
            return "reactors without enough heat exchangers / turbines"
    elif cat == "steam":
        if not 1.5 <= names["steam-engine"] / max(names["boiler"], 1) <= 2.0:
            return "not ~2 steam engines per boiler"
    elif cat == "solar":
        if names["accumulator"] and not 0.6 <= names["accumulator"] / names["solar-panel"] <= 1.2:
            return "accumulator / panel ratio far from 0.84"
    elif cat == "green_circuits":
        if recipes["copper-cable"] and not 1.2 <= recipes["copper-cable"] / max(recipes["electronic-circuit"], 1) <= 1.8:
            return "cable / circuit machines far from 3:2"
    # production categories come from the recipe graph: the product is right by construction
    return None


def draftsman_overlaps(t: Template) -> bool:
    """
    Second collision check with draftsman's exact collision boxes (the validator works on tiles,
    and a few entities reach outside their tiles, e.g. the offshore pump).
    """
    import warnings

    from draftsman.warning import OverlappingObjectsWarning

    from factoriollm.dataset_lifter import lift_ir_to_code, run_trusted_script

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_trusted_script(lift_ir_to_code(t.build(2 if t.repeats > 1 or t.head or t.tail else 1)))
    return any(issubclass(w.category, OverlappingObjectsWarning) for w in caught)


def score(t: Template) -> float:
    """Ranking: popularity (log), a bonus for 2.0 designs, density as a tie breaker."""
    fav = t.source.get("favorites", 0) or 0
    newer = 1.0 if t.source.get("version", [0])[0] >= 2 else 0.5 if t.source.get("version", [0])[0] >= 1 else 0.0
    density = t.metrics.get("tiles_per_machine") or 50
    return math.log10(fav + 1) + newer - density / 100


def ascii_map(ir: IRBlueprint, max_w: int = 80, max_h: int = 40) -> str:
    arrows = {0: "^", 4: ">", 8: "v", 12: "<"}
    special = {"underground-belt": "u", "splitter": "=", "inserter": "i", "long-handed-inserter": "l",
               "fast-inserter": "f", "bulk-inserter": "k", "burner-inserter": "n", "small-electric-pole": "p",
               "medium-electric-pole": "P", "substation": "$", "big-electric-pole": "B", "constant-combinator": "c",
               "small-lamp": "*", "pipe": "|", "pipe-to-ground": "!", "solar-panel": "S", "accumulator": "A",
               "lab": "L", "boiler": "O", "steam-engine": "E", "offshore-pump": "W", "electric-mining-drill": "D",
               "beacon": "#", "stone-furnace": "F", "steel-furnace": "F", "electric-furnace": "F"}
    recipe_chars = {"copper-cable": "C", "electronic-circuit": "G", "advanced-circuit": "R", "iron-gear-wheel": "V"}
    grid: Dict[tuple, str] = {}
    for e in ir.entities:
        base = e.name.replace("fast-", "").replace("express-", "")
        if base == "transport-belt":
            ch = arrows[e.direction]
        elif e.extras_dict.get("recipe") in recipe_chars:
            ch = recipe_chars[e.extras_dict["recipe"]]
        else:
            ch = special.get(base, special.get(e.name, e.name[0].upper()))
        w, h = footprint(e.name, e.direction)
        for dx in range(w):
            for dy in range(h):
                grid[(e.x + dx, e.y + dy)] = ch
    xs = [t[0] for t in grid]
    ys = [t[1] for t in grid]
    rows = []
    for y in range(min(ys), min(max(ys), min(ys) + max_h - 1) + 1):
        rows.append("".join(grid.get((x, y), ".") for x in range(min(xs), min(max(xs), min(xs) + max_w - 1) + 1)))
    return "\n".join(rows)


async def _load(min_favorites: int) -> List[Any]:
    from factoriollm.database import get_connection
    conn = await get_connection()
    rows = await conn.fetch("SELECT key, title, favorites, raw_data->'tags' AS tags, blueprint_string FROM blueprints "
                            "WHERE favorites >= $1 ORDER BY favorites DESC", min_favorites)
    await conn.close()
    return rows


def _vanilla(tags: Optional[str]) -> bool:
    for tag in json.loads(tags or "[]"):
        if tag.startswith("/mods/") and tag != "/mods/vanilla/":
            return False
        if tag.startswith("/planets/") and tag != "/planets/nauvis/":
            return False
    return True


SMELT_CHOICES = ("iron-ore", "iron-plate", "copper-ore", "stone")
SIMPLE_QUOTA = 3  # simple designs kept per category besides the best-ranked ones


def author_hints(ir: IRBlueprint) -> Dict[Tuple[int, int], set]:
    """What the author's labels name (combinator signals, display panel icons), by label tile."""
    out: Dict[Tuple[int, int], set] = {}
    for e in ir.entities:
        ex = e.extras_dict
        names = {part.split("=")[0].split(":", 1)[1] for part in (ex.get("signals") or "").split(";")
                 if part.startswith(("item:", "fluid:"))}
        if ex.get("icon", "").startswith(("item:", "fluid:")):
            names.add(ex["icon"].split(":", 1)[1])
        if names:
            out[(e.x, e.y)] = names
    return out


def run_flow(ir: IRBlueprint, hints: Optional[Dict] = None, first: str = "iron-ore") -> Tuple[FlowResult, str]:
    """The flow simulator's verdict; designs with furnaces are tried with each smelting input in turn."""
    furnaces = any(entity_type(e.name) == "furnace" for e in ir.entities)
    choices = [first] + [s for s in SMELT_CHOICES if s != first] if furnaces else [first]
    found: Optional[Tuple[FlowResult, str]] = None
    for smelt in choices:
        result = solve(ir, smelt=smelt, hints=hints)
        if result.ok:
            return result, smelt
        found = found or (result, smelt)
        if not any(p.kind == "starved" and "furnace" in p.message for p in result.problems):
            break  # another smelting input only helps a furnace that got nothing
    return found


REASON_ORDER = ("starved", "clog", "output_mixed", "fluid_mix", "no_output", "dangling_inserter")


def copies(ir: IRBlueprint, step: Tuple[int, int], n: int = 3) -> Optional[IRBlueprint]:
    """n copies of a design `step` tiles apart (copies may share identical entities); None if they collide."""
    placed: Dict[IREntity, None] = {}
    occupied: Dict[Tuple[int, int], IREntity] = {}
    for k in range(n):
        for e in ir.entities:
            c = IREntity(e.name, e.x + k * step[0], e.y + k * step[1], e.direction, e.extras)
            if c in placed:
                continue
            w, h = footprint(c.name, c.direction)
            for t in ((c.x + dx, c.y + dy) for dx in range(w) for dy in range(h)):
                if occupied.setdefault(t, c) != c:
                    return None
            placed[c] = None
    return ir.with_entities(list(placed))


def middle_flow(ir: IRBlueprint, step: Tuple[int, int], hints: Optional[Dict] = None,
                first: str = "iron-ore") -> Optional[Tuple[FlowResult, str]]:
    """
    The verdict for a design made to dock with copies of itself (its grid says how far apart): three
    copies, and only the problems of the middle one count - the outer copies miss their own outer
    neighbours. None if copies placed that way collide.
    """
    tiled = copies(ir, step)
    if tiled is None:
        return None
    result, smelt = run_flow(tiled, hints, first)
    axis = 0 if step[0] else 1
    lo = min((e.x, e.y)[axis] for e in ir.entities) + step[axis]
    middle = [p for p in result.problems if lo <= p.position[axis] < lo + step[axis]]
    kept = FlowResult(middle, result.lanes, result.machines, result.working, result.ports, result.inputs,
                      "ok" if not middle else result.status, result.score, result.fluids)
    return kept, smelt


def reason(result: FlowResult) -> str:
    """Why a design does not work: the search ran out, or its root problem (REASON_ORDER)."""
    if result.status == "inputs_ambiguous":
        return "inputs_ambiguous"
    kinds = {p.kind for p in result.problems}
    # the cause before its consequences: a starved machine leaves its inserters with nothing to move
    return next((k for k in REASON_ORDER if k in kinds), next(iter(kinds), result.status))


def furnace_input(result: FlowResult) -> str:
    """'steel' (iron plates come in), 'ore_to_steel' (ore in, steel out: furnace -> furnace) or 'ore'."""
    inputs = {v for v in result.inputs.values() if v}
    outputs = {x for p in result.ports if p.role == "output" for x in p.items}
    if "iron-plate" in inputs:
        return "steel"
    return "ore_to_steel" if "steel-plate" in outputs else "ore"


def _parse_reason(message: str) -> str:
    for key in ("legacy recipe", "quality", "label signal", "not allowed", "unknown entity", "base game", "circuit-wired"):
        if key in message:
            return key.replace(" ", "_")
    return "other"


class Rejects:
    """
    A few rejected designs per reason (reservoir sample), for the in-game rejects book. Workers send
    only a reference to the design (row key, book path, category); resolve() rebuilds the blueprint
    of the sampled ones.
    """

    def __init__(self, per_reason: int = REJECTS_PER_REASON, seed: int = 7):
        import random
        self.per_reason, self.random = per_reason, random.Random(seed)
        self.seen: Counter = Counter()
        self.kept: Dict[str, List[Tuple[str, Any, FlowResult]]] = {}  # (name, IR or reference, result)
        self.ref: Any = None  # what add() files the next rejection under (set per design)

    def add(self, why: str, name: str, ir: Any, result: FlowResult) -> None:
        self.seen[why] += 1
        bucket = self.kept.setdefault(why, [])
        if len(bucket) < self.per_reason:
            bucket.append((name, ir, result))
        else:
            k = self.random.randrange(self.seen[why])
            if k < self.per_reason:
                bucket[k] = (name, ir, result)

    def resolve(self, rows: Dict[str, Dict[str, Any]]) -> None:
        """Rebuilds the blueprint of each sampled rejection from its reference (see _Collected)."""
        for why, bucket in self.kept.items():
            for n, (name, ref, result) in enumerate(bucket):
                if isinstance(ref, tuple):
                    bucket[n] = (name, _rejected_design(rows[ref[0]], ref[1], ref[2], why), result)


def _rejected_design(row: Dict[str, Any], path: str, category: str, why: str) -> IRBlueprint:
    """The blueprint a rejection was about: the author's design, or the template build that failed."""
    bp, version = next((bp, v) for p, bp, v in iter_blueprints(decode_blueprint_string(row["blueprint_string"]))
                       if p == path)
    ir = parse_blueprint_dict(bp, version=version, book_path=path)
    if why.startswith("author:"):
        return prepare(ir, margin=None, keep_defense=category == "defense")
    capture = Rejects()
    _mine_design(ir, category, "", Counter(), capture)  # deterministic: fails the same way again
    return next(ir for bucket in capture.kept.values() for _, ir, _ in bucket)


MAX_WALL_THICKNESS = 20  # rows of wall (staggered rows and gaps included)
MIN_WALL_COVER = 0.6     # share of the segment columns with a wall piece (gaps for gates, turrets)
MAX_WALL_GAP = 3         # empty rows between staggered wall rows


def straight_wall(t: Template) -> bool:
    """
    A piece of defence to repeat along the line: a segment form with walls in most columns of the
    segment (MIN_WALL_COVER), all within one band at most MAX_WALL_THICKNESS thick, and the head and tail walls inside
    that band too (dragon's teeth are fine; two walls far apart, or caps across the line, are a ring
    or a perimeter; a thicker band repeats across the wall, not along it).
    """
    if not (t.head or t.repeats > 1 or t.tail):
        return False
    walls = ("stone-wall", "gate")
    rows: Dict[int, set] = {}
    for e in t.segment:
        if e.name in walls:
            w, h = footprint(e.name, e.direction)
            for dx in range(w):
                rows.setdefault((e.x + dx) % t.period, set()).update(range(e.y, e.y + h))
    if not rows:
        return True  # a row of turrets meant to stand behind a wall
    every = set().union(*rows.values())
    lo, hi = min(every), max(every)
    ends = [y for e in t.head + t.tail if e.name in walls for y in range(e.y, e.y + footprint(e.name, e.direction)[1])]
    gap = max((b - a - 1 for a, b in zip(sorted(every), sorted(every)[1:])), default=0)
    return (len(rows) >= MIN_WALL_COVER * t.period and hi - lo + 1 <= MAX_WALL_THICKNESS
            and gap <= MAX_WALL_GAP  # two walls with room between them are the sides of a ring
            and all(lo <= y <= hi for y in ends))  # the caps of a ring stick out of the band


def is_simple(t: Template) -> bool:
    """No beacons, no machine modules, no belt weaving, no assembling machine 3."""
    tr = t.metrics.get("traits") or {}
    return (not (t.metrics.get("beacons_per_machine") or 0) and not tr.get("machine_modules")
            and not tr.get("weaving") and main_machine(t) != "assembling-machine-3")


class _Collected:
    """Rejections of one worker, light to send back: a reference to the design and its first problems."""

    def __init__(self):
        self.items: List[Tuple[str, str, Tuple[str, str, str], FlowResult]] = []
        self.ref: Tuple[str, str, str] = ("", "", "")

    def add(self, why: str, name: str, ir: IRBlueprint, result: FlowResult) -> None:
        self.items.append((why, name, self.ref,
                           FlowResult(result.problems[:3], {}, {}, frozenset(), [], status=result.status)))


def _mine_row(job: Tuple[Dict[str, Any], Optional[set]]) -> Tuple[Counter, list, List[Tuple[str, Template]]]:
    """Every design of one corpus row (a blueprint or a book): counters, rejections, templates found."""
    row, only = job
    stats: Counter = Counter()
    rejects = _Collected()
    found: List[Tuple[str, Template]] = []
    if not _vanilla(row["tags"]):
        stats["modded"] += 1
        return stats, [], []
    try:
        items = list(iter_blueprints(decode_blueprint_string(row["blueprint_string"])))
    except BlueprintIRError:
        stats["decode_error"] += 1
        return stats, [], []
    for path, bp, version in items:
        if len(bp.get("entities") or []) > MAX_ENTITIES:
            stats["too_large"] += 1
            continue
        try:
            ir = parse_blueprint_dict(bp, version=version, book_path=path)
        except BlueprintIRError as e:
            stats[f"not_parsed: {_parse_reason(str(e))}"] += 1
            continue
        category = classify(ir)
        if category is None:
            if is_mall(products(ir)):
                stats["mall"] += 1
            continue
        if only and category not in only:
            continue
        stats[f"candidates {category}"] += 1
        name = (ir.label or row["title"] or "").strip()
        rejects.ref = (row["key"], path, category)
        template = _mine_design(ir, category, name, stats, rejects)
        if template is None:
            continue
        template.source = {"key": row["key"], "book_path": path, "label": ir.label, "title": row["title"],
                           "favorites": row["favorites"], "version": list(version)}
        template.metrics.update(metrics(template))
        template.metrics["kind"] = "segment" if template.head or template.repeats > 1 or template.tail else "cell"
        if category == "multi":
            template.metrics["products"] = products(ir)
        template.metrics["traits"] = traits(template)
        found.append((category, template))
    return stats, rejects.items, found


def mine(min_favorites: int = 10, per_category: int = 8, only: Optional[set] = None, workers: int = 0
         ) -> Tuple[Dict[str, List[Template]], Rejects]:
    """
    Mines the corpus rows in `workers` processes (0: all cores but one; 1: in this process). Results
    come back in row order, so the catalog and the rejects sample do not depend on the worker count.
    """
    import multiprocessing
    import time

    from factoriollm import template_miner as this  # workers find _mine_row by its package path

    rows = [{k: r[k] for k in ("key", "title", "favorites", "tags", "blueprint_string")}
            for r in asyncio.run(_load(min_favorites))]
    workers = workers or max(1, (multiprocessing.cpu_count() or 2) - 1)
    stats: Counter = Counter()
    rejects = Rejects()
    found: Dict[str, List[Template]] = {}
    jobs = ((row, only) for row in rows)
    start = time.time()

    def collect(results) -> None:
        for n, (row_stats, row_rejects, row_found) in enumerate(results, 1):
            stats.update(row_stats)
            for why, name, ref, result in row_rejects:
                rejects.add(why, name, ref, result)
            for category, template in row_found:
                found.setdefault(category, []).append(template)
            if n % 1000 == 0:
                print(f"[*] {n}/{len(rows)} rows, {time.time() - start:.0f} s", flush=True)

    if workers == 1:
        collect(map(this._mine_row, jobs))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            collect(pool.imap(this._mine_row, jobs, chunksize=1))  # rows differ a lot (whole books)
    print(f"[*] {len(rows)} rows mined in {time.time() - start:.0f} s with {workers} worker(s)")
    rejects.resolve({row["key"]: row for row in rows})
    best: Dict[str, List[Template]] = {}
    for cat, ts in sorted(found.items()):
        unique: Dict[Any, Template] = {}
        for t in sorted(ts, key=score, reverse=True):  # identical segments from sibling blueprints: keep the best
            unique.setdefault(signature(t), t)
        by_variant: Dict[str, List[Template]] = {}
        for t in unique.values():
            by_variant.setdefault(variant(t), []).append(t)
        chosen = [t for vs in by_variant.values() for t in vs[:3]][:per_category * 2]
        # simple early designs lose on popularity; keep a few anyway
        chosen += [t for t in unique.values() if is_simple(t) and all(t is not c for c in chosen)][:SIMPLE_QUOTA]
        best[cat] = sorted(chosen, key=lambda t: (variant(t), -score(t)))
    stats.update({f"rejected: {why}": n for why, n in rejects.seen.items()})
    print(f"[*] {dict(stats)}")
    if not only:  # a partial run keeps the full run's counters
        STATS.parent.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    return best, rejects


def _mine_design(ir: IRBlueprint, category: str, name: str, stats: Counter, rejects: Rejects) -> Optional[Template]:
    """
    One candidate design: it must work as the author built it (without radars, labels and, unless
    it is defence, walls and turrets); the cut of the surroundings must keep it working (gentler cuts
    otherwise); the template's builds must work as well. Rejections are counted by reason.
    """
    defense = category == "defense"
    hints = author_hints(ir)
    full = prepare(ir, margin=None, keep_defense=defense)
    source_flow, smelt = run_flow(full, hints)
    dock: Optional[Tuple[int, int]] = None  # step to the neighbouring copy for a design made to dock
    if not source_flow.ok:
        # the author's grid gives the step to the next copy; without one, copies stand side by side
        width, height = ir.snap or full.size
        for step in ((width, 0), (0, height)):
            tiled = middle_flow(full, step, hints, smelt)
            if tiled is not None and tiled[0].ok:
                (source_flow, smelt), dock = tiled, step
                stats["works docked to its copies"] += 1
                break
    if not source_flow.ok:
        rejects.add(f"author: {reason(source_flow)}", name, full, source_flow)
        return None

    def verdict(design: IRBlueprint) -> FlowResult:
        if not dock:
            return run_flow(design, hints, smelt)[0]
        docked = middle_flow(design, dock, hints, smelt)
        return docked[0] if docked else FlowResult([], {}, {}, frozenset(), [], status="copies_collide")

    cut, cut_flow = full, source_flow
    for step_no, (margin, taps) in enumerate(CROP_STEPS):
        candidate = prepare(ir, margin, defense, taps) if margin is not None else full
        flow = verdict(candidate) if margin is not None else source_flow
        if flow.ok:
            cut, cut_flow = candidate, flow
            stats[f"crop step {step_no}"] += 1
            break
    # never used, whatever comes in: the author's bus passing through, taps to nowhere, extra poles
    if not dock:
        stripped = strip_unused(cut)
        if len(stripped.entities) < len(cut.entities) and verdict(stripped).ok:
            cut = stripped
            stats["unused parts removed"] += 1
    try:
        template = _docking_cell(cut, category, dock) if dock else mine_one(cut, category)
    except BlueprintIRError:
        template = None
    if template is None:
        stats["no_template"] += 1
        return None
    template.belt = main_belt(template.head + template.segment + template.tail)
    try:
        check = tiling_check(template, docks=bool(dock))
        if not check["ok"] and check["power_only"] and not dock:
            repaired = repair_power(template)
            repaired_check = tiling_check(repaired) if repaired is not None else None
            if repaired_check and repaired_check["ok"]:
                template, check = repaired, repaired_check
                stats["power_repaired"] += 1
    except ValueError:  # BlueprintIRError is handled above; this is retier
        stats["belt_tier_failed"] += 1
        return None
    if not check["ok"]:
        stats["power_failed" if check["power_only"] else "tiling_failed"] += 1
        return None
    if sanity(template):
        stats["not_a_working_module"] += 1
        return None
    if draftsman_overlaps(template):
        stats["collision_boxes_overlap"] += 1
        return None
    template = normalize_modules(template)
    segment = template.head or template.repeats > 1 or template.tail
    if category == "defense" and not straight_wall(template):
        stats["defense_not_a_straight_wall"] += 1  # an outpost, a ring or a perimeter, not a piece of wall to repeat
        return None
    build_flow = None
    if dock:  # a docking cell works in a row of copies: judge the middle one of three
        build = template.build(1)
        docked = middle_flow(build, (template.period, 0), None, smelt)
        build_flow, smelt = docked if docked else (FlowResult([], {}, {}, frozenset(), [], status="collide"), smelt)
        if not build_flow.ok:
            rejects.add(f"template: {reason(build_flow)}", name, template.build(3), build_flow)
            return None
    for n in () if dock else (1, 2) if segment else (1,):
        build = template.build(n)
        build_flow, smelt = run_flow(build, first=smelt)
        if not build_flow.ok:
            rejects.add(f"template: {reason(build_flow)}", name, build, build_flow)
            return None
    template.metrics = {"source_warnings": check["source_warnings"], "smelt": smelt, "docks": bool(dock)}
    if category.startswith("smelting_"):
        template.metrics["input"] = furnace_input(build_flow)
    return template


def load_catalog(path: Path = CATALOG) -> Dict[str, List[Template]]:
    if not path.exists():
        return {}
    return {cat: [Template.from_json(d) for d in ts] for cat, ts in json.loads(path.read_text(encoding="utf-8")).items()}


CATEGORY_NAMES = {
    "smelting_stone": "Плавка в каменных печах", "smelting_steel": "Плавка в стальных печах",
    "smelting_electric": "Плавка в электропечах", "mining": "Добыча (электробуры)", "steam": "Паровая энергетика",
    "solar": "Солнечные поля", "nuclear": "Атомная энергетика", "labs": "Лаборатории",
    "green_circuits": "Зелёные схемы", "red_circuits": "Красные схемы", "blue_circuits": "Синие схемы",
    "plastic": "Пластик", "multi": "Мульти-блоки (несколько продуктов)",
    "assembly_1in": "Сборщики: 1 твёрдый вход", "assembly_2in": "Сборщики: 2 входа", "assembly_3in": "Сборщики: 3 входа",
    "assembly_fluid": "Сборщики: вход с жидкостью", "chemistry": "Химия (химзаводы)", "oil": "Нефтепереработка",
    "defense": "Оборона (отрезки стены)", "uranium": "Обогащение урана (центрифуги)",
    "science_red": "Красная наука", "science_green": "Зелёная наука", "science_military": "Военная наука",
    "science_blue": "Синяя наука", "science_purple": "Фиолетовая наука", "science_yellow": "Жёлтая наука",
}


DENSITY_SLACK = 1.15  # a default may use up to 15% more tiles per machine than the densest design


def default_template(ts: List[Template]) -> Template:
    """
    Default for requests without details: designs without beacons, then without machine modules
    (both late game), for furnaces ore smelting rather than steel, and with belts of their own (a
    city-block cell takes its items from outside); each preference applies only if some design
    meets it. Among those, the ones within DENSITY_SLACK of the densest by tiles per
    machine, and of them the best-ranked (popularity, 2.0 bonus). Popularity alone picked layouts
    that waste rows on poles.
    """
    plain = ts
    for prefer in (lambda t: not (t.metrics.get("beacons_per_machine") or 0),
                   lambda t: not (t.metrics.get("traits") or {}).get("machine_modules"),
                   lambda t: t.metrics.get("input") in (None, "ore"),
                   is_simple,
                   # defence: a wall with turrets (not a turret row meant to stand behind another wall)
                   lambda t: t.category != "defense" or ((t.metrics.get("traits") or {}).get("wall_rows")
                                                         and (t.metrics.get("traits") or {}).get("turrets") != "walls only"),
                   lambda t: t.metrics.get("belt_tier")):  # brings its own belts, not a city-block cell
        plain = [t for t in plain if prefer(t)] or plain

    def tiles(t: Template) -> float:
        return t.metrics.get("tiles_per_machine") or math.inf

    densest = min(map(tiles, plain))
    dense = [t for t in plain if tiles(t) <= DENSITY_SLACK * densest] if densest < math.inf else plain
    return max(dense, key=score)


def _describe(t: Template) -> str:
    m, s = t.metrics, t.source
    kind = (f"повторяемый сегмент {m['segment_size'][0]}×{m['segment_size'][1]} (в оригинале {t.repeats} шт.), "
            f"голова {m['head_entities']} и хвост {m['tail_entities']} сущностей" if m["kind"] == "segment"
            else f"целая ячейка {m['segment_size'][0]}×{m['segment_size'][1]}, тайлится копиями с шагом {t.period}")
    label = (s.get("label") or s.get("title") or "").replace("|", "/")
    return (f"«{label}» из «{s.get('title', '')}» — {s['favorites']}★, версия {'.'.join(map(str, s['version']))}; {kind}; "
            f"машин на сегмент: {m['machines_per_segment']}; тайлов на машину: {m['tiles_per_machine']}; "
            f"ленты: {m['belt_tier'] or '—'}; источник: `{s['key']}` / `{s['book_path'] or '-'}`")


def _name(t: Template) -> str:
    return (t.source.get("label") or t.source.get("title") or "").replace("|", "/").strip()[:60]


def _changes(best: Dict[str, List[Template]], previous: Dict[str, List[Template]]) -> List[str]:
    """'Before -> after' table per category: template counts and the default design."""
    lines = ["## Что изменилось по сравнению с прошлым каталогом", "",
             "| Категория | Шаблонов было | Стало | Умолчание было | Умолчание стало |", "| --- | --- | --- | --- | --- |"]
    def old_default(ts: List[Template]) -> Template:
        if all("traits" in t.metrics for t in ts):
            return default_template(ts)
        # a catalog from before variant traits: the rule of that time (best-ranked without beacons)
        return max([t for t in ts if not (t.metrics.get("beacons_per_machine") or 0)] or ts, key=score)

    for cat in sorted(set(best) | set(previous), key=lambda c: CATEGORY_NAMES.get(c, c)):
        old, new = previous.get(cat, []), best.get(cat, [])
        before = f"«{_name(old_default(old))}»" if old else "—"
        after = f"«{_name(default_template(new))}»" if new else "—"
        lines.append(f"| {CATEGORY_NAMES.get(cat, cat)} | {len(old)} | {len(new)} | {before} | {after} |")
    return lines + [""]


def write_outputs(best: Dict[str, List[Template]], previous: Optional[Dict[str, List[Template]]] = None,
                  stats: Optional[Dict[str, int]] = None) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps({cat: [t.to_json() for t in ts] for cat, ts in best.items()}, ensure_ascii=False),
                       encoding="utf-8")
    lines = ["# Каталог шаблонов из чертежей игроков", "",
             "Сгенерировано `template_miner.py`. Для каждой категории — шаблон по умолчанию и альтернативы по вариантам. "
             "Умолчание: среди вариантов без маяков — не больше чем на 15% хуже самого плотного по тайлам на машину, "
             "из них лучший по рейтингу (популярность, бонус за версию 2.0).", "",
             "Как готовятся шаблоны: категория — по конечному продукту (моллы исключены; промежуточные продукты — "
             "семейства по форме рецепта); чертёж должен работать сам по себе (симулятор потока: каждая машина получает "
             "ингредиенты, продукт уходит, нет засоров и висячих манипуляторов); окружение автора обрезано по потоку "
             "(врезки в шину, радары, авторские метки, оборона), обрезка не должна ломать работу; все ленты одного уровня, "
             "пары подземок и переплетение сохраняются; модули одинаковые у машин одного вида (исключения — по рецепту); "
             "лампы — необязательный слой; сетка привязки своя (период × высота сегмента). Проверка: сборки из 1, 2 и 5 "
             "сегментов без ошибок валидатора и без новых предупреждений, питание строгое, сборки из 1 и 2 сегментов "
             "работают по симулятору. Метки входов и выходов — из симулятора, по полосам.", ""]
    if previous:
        lines += _changes(best, previous)
    for cat, ts in best.items():
        top = default_template(ts)
        lines += [f"## {CATEGORY_NAMES.get(cat, cat)}", "", f"**По умолчанию ({variant(top)}).** {_describe(top)}", "",
                  "```", ascii_map(labelled(top, top.build(max(2, top.repeats) if top.metrics['kind'] == 'segment' else 1,
                                                           lamps=True))), "```", ""]
        others: Dict[str, Template] = {}
        for t in sorted(ts, key=score, reverse=True):
            if t is not top:
                others.setdefault(variant(t), t)
        if others:
            lines += ["Альтернативы:", ""]
            lines += [f"- *{v}*: {_describe(t)}" for v, t in sorted(others.items())]
            lines.append("")
    lines += ["## Обозначения на картах", "",
              "`^>v<` ленты, `u` подземка, `=` сплиттер, `i` манипулятор, `l` длинный, `f` быстрый, `k` bulk, "
              "`n` угольный, `p`/`P`/`$`/`B` малый/средний столб, подстанция, большой столб, `F` печь, `C`/`G`/`R`/`V` "
              "сборщик кабеля/зелёных/красных схем/шестерён, прочие сборщики — первая буква имени, `L` лаборатория, "
              "`D` бур, `S`/`A` панель/аккумулятор, `O` котёл, `E` паровой двигатель, `W` насос, `#` маяк, "
              "`c` метка-комбинатор, `*` лампа, `|`/`!` труба/подземная труба.", ""]
    if stats:
        lines += ["## Статистика отбора", "", "| Счётчик | Значение |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(stats.items())] + [""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Catalog: {CATALOG}; report: {REPORT}")


def blueprint_string(ir: IRBlueprint, label: str) -> str:
    """Blueprint string of an IR built by our own lifter (trusted code), with copper wires."""
    from draftsman.blueprintable import Blueprint

    from factoriollm.dataset_lifter import lift_ir_to_code, run_trusted_script
    from factoriollm.helpers import connect_power, game_string

    bp = Blueprint.from_string(run_trusted_script(lift_ir_to_code(ir)))
    bp.label = label[:100]
    connect_power(bp)
    return game_string(bp)


def labelled(t: Template, ir: IRBlueprint) -> IRBlueprint:
    """The build with input / output labels from the flow simulator (furnaces: the smelting input found when mining)."""
    return add_port_labels(ir, smelt=t.metrics.get("smelt") or "iron-ore")


def write_book(best: Dict[str, List[Template]], path: Path = Path("outputs/templates/catalog_book.txt")) -> None:
    """
    One blueprint book to import in game: a sub-book per category, the default first. Every design
    is built with 3 segments, lamps (if the author had them) and new port labels; the default also
    comes as a single segment on its own grid, to check tiling in game.
    """
    from draftsman.blueprintable import BlueprintBook, get_blueprintable_from_string

    book = BlueprintBook()
    book.label = "FactorioLLM templates"
    for cat, ts in best.items():
        top = default_template(ts)
        sub = BlueprintBook()
        sub.label = CATEGORY_NAMES.get(cat, cat)
        for t in [top] + [t for t in sorted(ts, key=score, reverse=True) if t is not top]:
            segments = 3 if t.metrics["kind"] == "segment" or t.metrics.get("docks") else 1
            name = (t.source.get("label") or t.source.get("title") or "").strip()
            label = f"{name[:40]} | {'DEFAULT ' if t is top else ''}{variant(t)}"  # the name first: labels are cut at 100
            builds = [(label, lambda t=t, n=segments: labelled(t, t.build(n, lamps=True)))]
            if t is top:
                builds.append((f"{name[:40]} | SEGMENT {t.snap[0]}x{t.snap[1]}", t.build_segment))
            for text, build in builds:
                try:
                    sub.blueprints.append(get_blueprintable_from_string(blueprint_string(build(), text)))
                except Exception as e:  # a template that cannot be rebuilt is reported, not hidden
                    print(f"    [!] {cat}: {name[:40]}: {type(e).__name__}: {e}")
        book.blueprints.append(sub)
    from factoriollm.helpers import game_string
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(game_string(book), encoding="utf-8")
    print(f"[*] Blueprint book: {path}")


def write_rejects_book(rejects: Rejects, path: Path = REJECTS_BOOK) -> None:
    """
    A few rejected designs per reason, to check in game that the reasons are fair. Each blueprint is
    labelled with the reason and the first problem; up to three problem spots carry a red X signal.
    """
    from draftsman.blueprintable import BlueprintBook, get_blueprintable_from_string

    from factoriollm.blueprint_ir import IREntity
    from factoriollm.helpers import game_string

    book = BlueprintBook()
    book.label = "FactorioLLM rejected designs"
    for why, kept in sorted(rejects.kept.items()):
        sub = BlueprintBook()
        sub.label = f"{why} ({rejects.seen[why]})"
        for name, ir, result in kept:
            used = {(e.x + dx, e.y + dy) for e in ir.entities for dx in range(footprint(e.name, e.direction)[0])
                    for dy in range(footprint(e.name, e.direction)[1])}
            marks = []
            for p in result.problems[:3]:
                x, y = p.position
                spot = next(((x + dx, y + dy) for dx, dy in ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (2, 0), (0, 2))
                             if (x + dx, y + dy) not in used), None)
                if spot:
                    used.add(spot)
                    marks.append(IREntity("constant-combinator", spot[0], spot[1], 0, (("signals", "virtual:signal-X=1"),)))
            first = result.problems[0].message if result.problems else result.status
            try:
                text = f"{name[:30]} | {why}: {first}"
                sub.blueprints.append(get_blueprintable_from_string(blueprint_string(ir.with_entities(ir.entities + marks), text)))
            except Exception as e:
                print(f"    [!] reject {why}: {name[:40]}: {type(e).__name__}: {e}")
        book.blueprints.append(sub)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(game_string(book), encoding="utf-8")
    print(f"[*] Rejects book: {path}")


def main():
    parser = argparse.ArgumentParser(description="Mine tileable layout templates from popular blueprints")
    parser.add_argument("--min-favorites", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--report-only", action="store_true",
                        help="Rewrite the report and the book from the existing catalog (no mining)")
    parser.add_argument("--previous", type=Path, default=None,
                        help="Catalog to compare with (default: the catalog being replaced)")
    parser.add_argument("--workers", type=int, default=0, help="Processes for mining (0: all cores but one)")
    parser.add_argument("--only", default="",
                        help="Comma-separated categories to mine again; the rest of the catalog is kept")
    args = parser.parse_args()
    only = {c for c in args.only.split(",") if c}
    previous = load_catalog(args.previous) if args.previous else ({} if args.report_only else load_catalog())
    rejects = None
    if args.report_only:
        best = load_catalog()
    elif only:
        kept = {cat: ts for cat, ts in load_catalog().items() if cat not in only}  # a category may now be empty
        best = {**kept, **mine(args.min_favorites, args.per_category, only, args.workers)[0]}
        previous = load_catalog(args.previous) if args.previous else {}
    else:
        best, rejects = mine(args.min_favorites, args.per_category, workers=args.workers)
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else None
    write_outputs(best, previous, stats)
    write_book(best)
    if rejects is not None:
        write_rejects_book(rejects)


if __name__ == "__main__":
    main()
