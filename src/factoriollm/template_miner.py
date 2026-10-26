"""
Template miner: finds the best tileable player designs per category in the scraped corpus.

For every blueprint (and every child of a blueprint book) with enough favorites:
    raw JSON -> IR -> category by content -> template (repeating segment with head and tail, or a
    whole cell tiled by copies) -> tiling check with several lengths -> metrics
The best templates per category go to data/templates/catalog.json, and a readable report with
ASCII maps to outputs/templates/catalog.md.

    uv run python -m factoriollm.template_miner [--min-favorites 10] [--per-category 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from factoriollm.blueprint_ir import (
    BlueprintIRError,
    IRBlueprint,
    decode_blueprint_string,
    entity_type,
    footprint,
    iter_blueprints,
    parse_blueprint_dict,
    rotated,
)
from factoriollm.game_data import final_products, places_entity
from factoriollm.templates import (
    Template,
    add_port_labels,
    along_x,
    extract,
    extract_cell,
    main_belt,
    metrics,
    has_weaving,
    prepare,
    repair_power,
    signature,
    tiling_check,
)

CATALOG = Path("data/templates/catalog.json")
STATS = Path("data/templates/miner_stats.json")
REPORT = Path("outputs/templates/catalog.md")
MAX_ENTITIES = 3000

ASSEMBLERS = ("assembling-machine-1", "assembling-machine-2", "assembling-machine-3")
FURNACES = {"stone-furnace": "smelting_stone", "steel-furnace": "smelting_steel", "electric-furnace": "smelting_electric"}
SCIENCE = {"automation-science-pack": "science_red", "logistic-science-pack": "science_green",
           "military-science-pack": "science_military", "chemical-science-pack": "science_blue",
           "production-science-pack": "science_purple", "utility-science-pack": "science_yellow"}
# Machines with a recipe in the blueprint (furnaces pick theirs from the input).
RECIPE_MACHINES = {*ASSEMBLERS, "chemical-plant", "oil-refinery", "centrifuge"}
# Single final product -> category.
PRODUCT_CATEGORIES = {"electronic-circuit": "green_circuits", "advanced-circuit": "red_circuits",
                      "processing-unit": "blue_circuits", "iron-gear-wheel": "gears", "plastic-bar": "plastic",
                      **SCIENCE}
MULTI_MAX_PRODUCTS = 3  # more final products than this is a mall: no tileable segment, not mined
FLUID_ENTITIES = {"offshore-pump", "pump", "pipe", "pipe-to-ground", "storage-tank"}


# Machines that make up each category; a module must consist mostly of them.
FAMILIES = {
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
AREA_PURITY = {"nuclear": 0.4, "steam": 0.5, "solar": 0.5, "mining": 0.5}
AREA_FAMILIES = {**FAMILIES, "mining": {"electric-mining-drill", "beacon"}}  # beaconed mining is still mining
# "steel furnace" / "стальные печи" name the furnace, not steel smelting
FURNACE_WORDS = re.compile(r"steel[\s_-]*furnaces?|стальн\w*", re.IGNORECASE)
STEEL_WORDS = re.compile(r"\bsteel\b|\bстал[ьи]\b", re.IGNORECASE)
ORES = {"iron-ore", "copper-ore", "stone"}


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


def smelting_input(ir: IRBlueprint, text: str = "") -> str:
    """
    'steel' for a column that smelts iron plates into steel, 'ore' for ore / stone smelting. Furnaces
    carry no recipe, so this comes from the author's labels and the blueprint name.
    """
    signals = {part.split("=")[0].split(":", 1)[1] for e in ir.entities
               for part in (e.extras_dict.get("signals") or "").split(";") if part}
    signals |= {e.extras_dict["icon"].split(":", 1)[1] for e in ir.entities if "icon" in e.extras_dict}
    text = FURNACE_WORDS.sub(" ", f"{text} {' '.join(e.extras_dict.get('text', '') for e in ir.entities)}")
    # an 'iron-plate' label alone is no hint (ore columns label their output with it); a steel label
    # next to ore labels marks a mixed bus smelter that is mostly ore smelting
    if STEEL_WORDS.search(text) or ("steel-plate" in signals and not signals & ORES):
        return "steel"
    return "ore"


def _category(names: Counter, ir: IRBlueprint) -> Optional[str]:
    furnaces = {k: names[k] for k in FURNACES}
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
            return PRODUCT_CATEGORIES.get(final[0])
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
    if category is None:
        return None
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
    return {
        "pitch": "tight" if machines and max(widths.values()) >= t.period else "gaps",
        "poles": poles,
        "machine_modules": _modules(machines),
        "beacon_modules": _modules([e for e in t.segment if e.name == "beacon"]),
        "chain": len(recipes) > 1,
        "weaving": has_weaving(t.build(3).entities),
        "lamps": bool(t.metrics.get("lamps")),
    }


def variant(t: Template) -> str:
    """
    Variant label a user can ask for: beacons (and their modules) or not, the main machine tier and
    its modules, what a furnace column smelts (ore or steel) or which products a multi-product block
    makes, machine pitch, pole placement, chains, belt weaving and lamps.
    """
    tr = t.metrics.get("traits") or {}
    beaconed = (t.metrics.get("beacons_per_machine") or 0) > 0
    parts = [f"beacons {tr['beacon_modules']}" if beaconed and tr.get("beacon_modules")
             else "beacons" if beaconed else "no beacons", main_machine(t)]
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


def mine(min_favorites: int = 10, per_category: int = 8) -> Dict[str, List[Template]]:
    rows = asyncio.run(_load(min_favorites))
    stats: Counter = Counter()
    found: Dict[str, List[Template]] = {}
    for row in rows:
        if not _vanilla(row["tags"]):
            stats["modded"] += 1
            continue
        try:
            items = list(iter_blueprints(decode_blueprint_string(row["blueprint_string"])))
        except BlueprintIRError:
            stats["decode_error"] += 1
            continue
        for path, bp, version in items:
            if len(bp.get("entities") or []) > MAX_ENTITIES:
                stats["too_large"] += 1
                continue
            try:
                ir = parse_blueprint_dict(bp, version=version, book_path=path)
            except BlueprintIRError:
                stats["not_parsed"] += 1
                continue
            category = classify(ir)
            if category is None:
                if is_mall(products(ir)):
                    stats["mall"] += 1
                continue
            stats[f"candidates {category}"] += 1
            try:
                template = mine_one(prepare(ir), category)
            except BlueprintIRError:
                template = None
            if template is None:
                stats["no_template"] += 1
                continue
            template.belt = main_belt(template.head + template.segment + template.tail)
            try:
                check = tiling_check(template)
                if not check["ok"] and check["power_only"]:
                    repaired = repair_power(template)
                    if repaired is not None and tiling_check(repaired)["ok"]:
                        template, check = repaired, tiling_check(repaired)
                        stats["power_repaired"] += 1
            except ValueError:  # BlueprintIRError is handled above; this is retier
                stats["belt_tier_failed"] += 1
                continue
            if not check["ok"] and check["power_only"]:
                stats["power_failed"] += 1
                continue
            if not check["ok"]:
                stats["tiling_failed"] += 1
                continue
            if sanity(template):
                stats["not_a_working_module"] += 1
                continue
            if draftsman_overlaps(template):
                stats["collision_boxes_overlap"] += 1
                continue
            template.source = {"key": row["key"], "book_path": path, "label": ir.label, "title": row["title"],
                               "favorites": row["favorites"], "version": list(version)}
            template.metrics = {**metrics(template), "kind": "segment" if template.head or template.repeats > 1 or template.tail else "cell",
                                "source_warnings": check["source_warnings"]}
            if category == "multi":
                template.metrics["products"] = products(ir)
            if category.startswith("smelting_"):
                template.metrics["input"] = smelting_input(ir, ir.label if path else f"{ir.label} {row['title']}")
            template.metrics["traits"] = traits(template)
            found.setdefault(category, []).append(template)
    best: Dict[str, List[Template]] = {}
    for cat, ts in sorted(found.items()):
        unique: Dict[Any, Template] = {}
        for t in sorted(ts, key=score, reverse=True):  # identical segments from sibling blueprints: keep the best
            unique.setdefault(signature(t), t)
        by_variant: Dict[str, List[Template]] = {}
        for t in unique.values():
            by_variant.setdefault(variant(t), []).append(t)
        best[cat] = [t for vs in by_variant.values() for t in vs[:3]][:per_category * 2]
        best[cat].sort(key=lambda t: (variant(t), -score(t)))
    print(f"[*] {dict(stats)}")
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(dict(stats), ensure_ascii=False, indent=1), encoding="utf-8")
    return best


def load_catalog(path: Path = CATALOG) -> Dict[str, List[Template]]:
    if not path.exists():
        return {}
    return {cat: [Template.from_json(d) for d in ts] for cat, ts in json.loads(path.read_text(encoding="utf-8")).items()}


CATEGORY_NAMES = {
    "smelting_stone": "Плавка в каменных печах", "smelting_steel": "Плавка в стальных печах",
    "smelting_electric": "Плавка в электропечах", "mining": "Добыча (электробуры)", "steam": "Паровая энергетика",
    "solar": "Солнечные поля", "nuclear": "Атомная энергетика", "labs": "Лаборатории",
    "green_circuits": "Зелёные схемы", "red_circuits": "Красные схемы", "blue_circuits": "Синие схемы",
    "gears": "Шестерни", "plastic": "Пластик", "multi": "Мульти-блоки (несколько продуктов)",
    "science_red": "Красная наука", "science_green": "Зелёная наука", "science_military": "Военная наука",
    "science_blue": "Синяя наука", "science_purple": "Фиолетовая наука", "science_yellow": "Жёлтая наука",
}


DENSITY_SLACK = 1.15  # a default may use up to 15% more tiles per machine than the densest design


def default_template(ts: List[Template]) -> Template:
    """
    Default for requests without details: among designs without beacons (beacons are late game),
    those within DENSITY_SLACK of the densest by tiles per machine; among them the best-ranked
    (popularity, 2.0 bonus). Popularity alone picked layouts that waste rows on poles.
    """
    plain = [t for t in ts if not (t.metrics.get("beacons_per_machine") or 0)] or ts

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
    for cat in sorted(set(best) | set(previous), key=lambda c: CATEGORY_NAMES.get(c, c)):
        old, new = previous.get(cat, []), best.get(cat, [])
        before = f"«{_name(default_template(old))}»" if old else "—"
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
             "Как готовятся шаблоны: категория — по конечному продукту (моллы исключены); окружение автора обрезано "
             "(врезки в шину, радары, авторские метки); все ленты одного уровня, пары подземок и переплетение сохраняются; "
             "лампы — необязательный слой; сетка привязки своя (период × высота сегмента). Проверка: сборки из 1, 2 и 5 "
             "сегментов без ошибок валидатора и без новых предупреждений, питание строгое (без островов столбов и "
             "незапитанных машин, даже если так было у автора; при необходимости столбы сегмента расставлены заново).", ""]
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
    from factoriollm.helpers import connect_power

    bp = Blueprint.from_string(run_trusted_script(lift_ir_to_code(ir)))
    bp.label = label[:100]
    connect_power(bp)
    return bp.to_string()


def labelled(t: Template, ir: IRBlueprint) -> IRBlueprint:
    """The build with input / output labels from recipe data (furnaces: iron ore or, for steel, iron plates)."""
    return add_port_labels(ir, smelt="iron-plate" if t.metrics.get("input") == "steel" else "iron-ore")


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
            segments = 3 if t.metrics["kind"] == "segment" else 1
            name = (t.source.get("label") or t.source.get("title") or "").strip()
            label = f"{'DEFAULT ' if t is top else ''}{variant(t)} | {name}"
            builds = [(label, lambda t=t, n=segments: labelled(t, t.build(n, lamps=True)))]
            if t is top:
                builds.append((f"SEGMENT {t.snap[0]}x{t.snap[1]} | {name}", t.build_segment))
            for text, build in builds:
                try:
                    sub.blueprints.append(get_blueprintable_from_string(blueprint_string(build(), text)))
                except Exception as e:  # a template that cannot be rebuilt is reported, not hidden
                    print(f"    [!] {cat}: {name[:40]}: {type(e).__name__}: {e}")
        book.blueprints.append(sub)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(book.to_string(), encoding="utf-8")
    print(f"[*] Blueprint book: {path}")


def main():
    parser = argparse.ArgumentParser(description="Mine tileable layout templates from popular blueprints")
    parser.add_argument("--min-favorites", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--report-only", action="store_true",
                        help="Rewrite the report and the book from the existing catalog (no mining)")
    parser.add_argument("--previous", type=Path, default=None,
                        help="Catalog to compare with (default: the catalog being replaced)")
    args = parser.parse_args()
    previous = load_catalog(args.previous) if args.previous else ({} if args.report_only else load_catalog())
    best = load_catalog() if args.report_only else mine(args.min_favorites, args.per_category)
    stats = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else None
    write_outputs(best, previous, stats)
    write_book(best)


if __name__ == "__main__":
    main()
