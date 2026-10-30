"""
Generic templates (step 2b): a row of machines built with any recipe of the right shape.

A player's blueprint stores the machines, inserters and belts, but not which item goes on which lane.
So a generic template keeps the author's geometry, puts the new recipe into the row's machines and
lets the flow simulator find the lanes (one item per lane, inputs where a belt can reach), as for any
template build. What the row can take comes from its roles (templates.segment_slots): how many lanes
feed a machine, how many fluid connections it has, whether its products leave on a belt.

- with_recipe: the template with another recipe, or Incompatible with the reason;
- instantiate: a working build of it, with labels, our own bare ends first and the author's head and
  tail as the fallback;
- throughput: how much a machine eats and how many a lane and an inserter can feed;
- shaped_like: recipes of the same shape to try a template with (the template x recipe check);
- check_catalog, write_report, write_test_book: that check over the catalog, its report and a book for the game.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from factoriollm.base_game import ENTITIES, RECIPES
from factoriollm.blueprint_ir import IRBlueprint, IREntity, entity_type, footprint
from factoriollm.flow_sim import FlowResult
from factoriollm.game_data import entity, inserter_rate, recipe
from factoriollm.templates import (POWER_CATEGORIES, Template, add_port_labels, machine_family, repair_power,
                                   segment_slots)

MACHINE_TYPES = ("assembling-machine", "furnace")
INSERTER_MARGIN = 0.8  # an inserter is busy enough at this share of its rate (research note, step 2b)
LANE_ITEMS_PER_TILE = 4  # items on one lane of one belt tile


class Incompatible(ValueError):
    """The recipe does not fit the template; the message says why, as a user would read it."""


@dataclass
class Generic:
    """A template built with a recipe: the build, its flow verdict and which ends it uses."""
    template: Template
    recipe: str
    machine: str
    build: IRBlueprint
    flow: FlowResult
    ends: str  # "own" (bare segments) or "author" (the author's head and tail)
    notes: List[str] = field(default_factory=list)


def row_machines(t: Template) -> Tuple[str, Optional[str]]:
    """The row's machine and recipe: the most common machine of the main family in the segment."""
    machines = Counter((e.name, e.extras_dict.get("recipe")) for e in t.segment if entity_type(e.name) in MACHINE_TYPES)
    if not machines:
        raise Incompatible("the template has no crafting machines")
    return machines.most_common(1)[0][0]


def _machine_for(r, name: str) -> Optional[str]:
    """The template's machine if it crafts the recipe, else one of the same kind and size that does
    of the same family (the slowest one at least as fast: assembling machine 1 has no fluid input, 2
    takes its place; a chemical plant is 3x3 too, but it is no assembler)."""
    if r.category in (entity(name).get("crafting_categories") or ()):
        return name
    size, family, speed = footprint(name, 0), machine_family(name), float(entity(name).get("crafting_speed") or 1)
    fits = [n for n, e in ENTITIES.items() if machine_family(n) == family and footprint(n, 0) == size
            and r.category in (e.get("crafting_categories") or ()) and float(e.get("crafting_speed") or 1) >= speed]
    return min(fits, key=lambda n: float(ENTITIES[n].get("crafting_speed") or 1)) if fits else None


def with_recipe(t: Template, name: str, machine: Optional[str] = None) -> Template:
    """
    The template with recipe `name` in every machine of its row (furnaces: the smelting input it is
    fed). Incompatible if the recipe does not fit: another machine kind, more solid ingredients than
    lanes feed a machine, more fluids than the machine has connections, products with no belt to
    leave on, or a block of several recipes (a chain, not a row).
    """
    r = recipe(name)
    if r is None:
        raise Incompatible(f"{name}: no such recipe in the base game")
    row_name, row_recipe = row_machines(t)
    family = machine_family(row_name)
    slots = segment_slots(t)
    if family is None:
        raise Incompatible(f"{row_name} is not a crafting machine")
    others = {e.extras_dict.get("recipe") for e in t.segment if entity_type(e.name) in MACHINE_TYPES} - {row_recipe}
    if others:
        raise Incompatible(f"a block of several recipes ({row_recipe} with {', '.join(sorted(map(str, others)))}), not a row")
    target = machine or _machine_for(r, row_name)
    if target is None or r.category not in (entity(target).get("crafting_categories") or ()):
        raise Incompatible(f"{name} ({r.category}) is not crafted by {row_name} or a machine of its size")
    solids = [i for i in r.ingredients if i not in r.fluids]
    fluids_in = [i for i in r.ingredients if i in r.fluids]
    fluids_out = [i for i in r.results if i in r.fluids]
    if len(solids) > slots["max_item_inputs"]:
        raise Incompatible(f"{name} has {len(solids)} solid ingredients, a machine of this row is fed from "
                           f"{slots['max_item_inputs']} lanes")
    if len(fluids_in) > slots["max_fluid_in"] or len(fluids_out) > slots["max_fluid_out"]:
        raise Incompatible(f"{name} needs {len(fluids_in)} fluid inputs and {len(fluids_out)} fluid outputs, "
                           f"the row has {slots['max_fluid_in']} and {slots['max_fluid_out']}")
    topology = slots["topology"] or ""
    out = topology.split("out[")[-1].rstrip("]")
    if any(i not in r.fluids for i in r.results) and not [o for o in out.split(",") if o and o != "direct"]:
        raise Incompatible(f"the products of {name} have no belt or chest to leave to ({topology})")
    if family == "furnace_column":  # a furnace smelts what it is fed: the recipe is the smelting input
        return replace(t, metrics={**t.metrics, "smelt": solids[0], "recipe": name})

    def swap(ents: List[IREntity]) -> List[IREntity]:
        out_ents = []
        for e in ents:
            if e.name == row_name and e.extras_dict.get("recipe") == row_recipe:
                extras = dict(e.extras)
                extras["recipe"] = name
                if not r.productivity and "productivity" in extras.get("modules", ""):
                    extras.pop("modules")  # productivity modules only go into intermediates
                e = IREntity(target, e.x, e.y, e.direction, tuple(sorted(extras.items())))
            out_ents.append(e)
        return out_ents

    return replace(t, head=swap(t.head), segment=swap(t.segment), tail=swap(t.tail),
                   metrics={**t.metrics, "recipe": name, "machine": target})


def drop_spare_pipes(t: Template, smelt: str) -> Template:
    """
    The template without the pipe networks its new recipe does not need: a row built for two input
    fluids (sulfur: water and petroleum gas) fed one (solid fuel from heavy oil) would carry the same
    fluid in two networks to every machine, where one straight pipe does (in-game review of 30.10).
    Each fed input network is taken out in turn (its pipes and pipes-to-ground) and stays out if the
    row still works with one item per lane; the validator check of the caller follows.
    """
    from factoriollm.flow_sim import Layout
    from factoriollm.template_miner import run_flow
    from factoriollm.templates import PIPE_TYPES

    form = segment_form(t)
    n = 2 if form else 1
    while True:
        build = t.build(n)
        flow, _ = run_flow(build, first=smelt, share_lanes=False)
        fed = sorted({k[1] for k, v in flow.inputs.items() if k[0] == "fluid" and v})
        if not flow.ok or len(fed) < 2:
            return t
        L = Layout(build)
        parts = t._tiered_parts(t.belt)
        placed = t._place(parts, n)
        dx = min(e.x for e in build.entities) - min(e.x for e in placed)
        dy = min(e.y for e in build.entities) - min(e.y for e in placed)
        where: Dict[Tuple[int, int], Tuple[str, int]] = {}  # build tile -> (part, index) of the template
        for part, shift in (("head", [0]), ("segment", [k * t.period for k in range(n)]),
                            ("tail", [(n - t.repeats) * t.period])):
            for i, e in enumerate(parts[part]):
                for s in shift:
                    where[(e.x + s + dx, e.y + dy)] = (part, i)
        for net in fed:
            pipes = {where[(L.ents[j].x, L.ents[j].y)] for j in {b[0] for b in L.boxes}
                     if L.types[j] in PIPE_TYPES and (L.ents[j].x, L.ents[j].y) in where
                     and all(L.network[b] == net for b, box in enumerate(L.boxes) if box[0] == j)}
            if not pipes:
                continue
            thinner = replace(t, **{part: [e for i, e in enumerate(getattr(t, part)) if (part, i) not in pipes]
                                    for part in ("head", "segment", "tail")})
            flows = [run_flow(thinner.build(k), first=smelt, share_lanes=False)[0] for k in range(1, n + 1)]
            # fewer networks to feed, not a box of every machine left open to be piped one by one
            fewer = len({k[1] for k, v in flows[-1].inputs.items() if k[0] == "fluid" and v}) < len(fed)
            if all(f.ok for f in flows) and fewer:
                t = thinner
                break
        else:
            return t


def bare(t: Template, smelt: str = "iron-ore") -> Template:
    """
    The template with our own ends instead of the author's head and tail: the segments' belts simply
    start and end, and what the first and last segment take from or hand to the copy before / after
    them comes from that copy, as for a row of docking cells (template_miner.close_row: the belt the
    last copy's products go onto, its neighbour's pole that powers its last machine). If the row
    cannot be closed so, only the neighbours' poles that are still needed are added (prune_poles).
    """
    from factoriollm.template_miner import close_row
    from factoriollm.templates import prune_poles

    closed_row = close_row(replace(t, head=[], tail=[], repeats=1), smelt)
    if closed_row is not None:
        return closed_row

    def shift(ents: List[IREntity], dx: int) -> List[IREntity]:
        return [IREntity(e.name, e.x + dx, e.y, e.direction, e.extras) for e in ents]

    poles = [e for e in t.segment if entity_type(e.name) == "electric-pole"]
    head, tail = shift(poles, -t.period), shift(poles, t.period)  # tail: after one segment, see Template._place
    closed = replace(t, head=head, tail=tail, repeats=1)
    # which of them a build of two segments still needs (in _place coordinates: the tail moves one period on)
    kept = set(prune_poles(closed._place(closed._tiered_parts(closed.belt), 2)))
    return replace(closed, head=[e for e in head if e in kept],
                   tail=[e for e, moved in zip(tail, shift(tail, t.period)) if moved in kept])


def instantiate(t: Template, name: str, segments: int = 2, machine: Optional[str] = None,
                lamps: bool = False) -> Generic:
    """
    A working build of the template with recipe `name`: first with our own (bare) ends, which must
    work and tile cleanly, else with the author's head and tail (the fallback must work too). The
    build carries input / output labels. Incompatible if the recipe does not fit or neither works.
    """
    from factoriollm.template_miner import run_flow

    generic = with_recipe(t, name, machine)
    smelt = generic.metrics.get("smelt") or t.metrics.get("smelt") or "iron-ore"
    allowed = set(t.metrics.get("source_warnings") or ()) - POWER_CATEGORIES
    tries = [("own", bare(generic, smelt))] if generic.head or generic.tail or generic.repeats > 1 else []
    tries.append(("author", generic))
    reasons = []
    for ends, candidate in tries:
        candidate = drop_spare_pipes(candidate, smelt)
        # our own ends start and end the belts where the author's head and tail took them on
        candidate, why, used = _verify(candidate, smelt, allowed | OWN_ENDS_WARNINGS if ends == "own" else allowed,
                                       repair=ends == "own")
        if candidate is None:
            reasons.append(f"{ends} ends: {why}")
            continue
        build = candidate.build(segments, lamps=lamps)
        flow, used = run_flow(build, first=used, share_lanes=False)
        if not flow.ok:
            reasons.append(f"{ends} ends, {segments} segments: {flow.status} {sorted({p.kind for p in flow.problems})}")
            continue
        labelled = add_port_labels(build, smelt=used, flow=flow)
        return Generic(candidate, name, generic.metrics.get("machine") or row_machines(t)[0], labelled, flow, ends, reasons)
    raise Incompatible(f"{name} in this row does not work: {'; '.join(reasons)}")


VERIFY_SEGMENTS = (1, 2, 5)
OWN_ENDS_WARNINGS = {"belt_end"}  # input and output belts begin and end at our own ends  # builds the validator checks; the simulator checks the first two, as the miner


def _verify(candidate: Template, smelt: str, allowed: set, repair: bool) -> Tuple[Optional[Template], str, str]:
    """
    Whether the template with its new recipe works as the miner demands of a template: builds of 1
    and 2 segments work in the strict simulation (one item per lane, inputs a belt can reach), and
    builds of 1, 2 and 5 segments bring no validator error and no warning category the author's
    design did not have (power ones never). With `repair` the poles are re-placed once when power
    is the only problem (our own ends lack the author's poles at head and tail). Returns the
    template (repaired or not), else None and why; and the smelting input found.
    """
    from factoriollm.template_miner import run_flow
    from factoriollm.validator import validate

    def problems(c: Template) -> List[str]:
        out = []
        for n in VERIFY_SEGMENTS:
            report = validate(c.build(n))
            new = {w.category for w in report.warnings} - allowed
            if report.errors or new:
                out.append(f"{n} segments: {report.errors[0].message if report.errors else sorted(new)}")
        return out

    found = problems(candidate)
    if found and repair and all(any(cat in p for cat in POWER_CATEGORIES) for p in found):
        repaired = repair_power(candidate)
        if repaired is not None and not problems(repaired):
            candidate, found = repaired, []
    if found:
        return None, found[0], smelt
    form = segment_form(candidate)
    for n in VERIFY_SEGMENTS[:2] if form else VERIFY_SEGMENTS[:1]:  # a cell: one copy, as the miner
        flow, smelt = run_flow(candidate.build(n), first=smelt, share_lanes=False)
        if not flow.ok:
            return None, f"{n} segments: {flow.status} {sorted({p.kind for p in flow.problems})}", smelt
    return candidate, "", smelt


def lane_rate(belt: str) -> float:
    """Items per second one lane of the belt carries."""
    return float(entity(belt).get("speed") or 0) * 60 * LANE_ITEMS_PER_TILE


def throughput(g: Generic) -> Dict[str, Any]:
    """
    How much one machine of the row eats and makes per second (crafting speed of the machine, no
    modules or beacons), how many machines the input lanes of the build feed for each ingredient at
    its belt tier, and whether the row's inserters keep up (INSERTER_MARGIN of their rate).
    """
    r = recipe(g.recipe)
    speed = float(entity(g.machine).get("crafting_speed") or 1)
    per_machine = {i: a * speed / r.time for i, a in r.ingredients.items() if i not in r.fluids}
    made = {i: a * speed / r.time for i, a in r.results.items() if i not in r.fluids}
    lanes = Counter(x for k, x in g.flow.inputs.items() if x and k[0] == "lane")
    belts = Counter(e.name for e in g.build.entities if entity_type(e.name) == "transport-belt")
    belt = belts.most_common(1)[0][0] if belts else "transport-belt"
    feeds = {i: (lanes[i] * lane_rate(belt) / d if d else None) for i, d in per_machine.items()}
    slots = segment_slots(g.template)
    hands = [h for h in slots["inserters"] if h["machine"] in slots["machines"]]
    rates: Dict[Tuple[int, str], float] = {}
    for h in hands:
        if h["inserter"] is None:
            continue
        name = g.template.segment[h["inserter"]].name
        rates[(h["machine"], h["role"])] = rates.get((h["machine"], h["role"]), 0.0) + inserter_rate(name)
    machines = slots["machines"]
    fed = min((rates.get((m, "in"), 0.0) for m in machines), default=0.0)
    emptied = min((rates.get((m, "out"), 0.0) for m in machines), default=0.0)
    return {
        "per_machine_in": {i: round(v, 3) for i, v in per_machine.items()},
        "per_machine_out": {i: round(v, 3) for i, v in made.items()},
        "belt": belt,
        "machines_per_lane_load": {i: (int(v) if v is not None else None) for i, v in feeds.items()},
        "inserters_keep_up": sum(per_machine.values()) <= INSERTER_MARGIN * fed if fed else None,
        "output_keeps_up": sum(made.values()) <= INSERTER_MARGIN * emptied if emptied else None,
    }


def shaped_like(t: Template, limit: int = 4) -> List[str]:
    """
    Recipes to try the template with: crafted by its machine or one of its family and size, with no
    more solid ingredients and fluids than it takes, not the author's own, items as products. Those
    of the author's recipe's shape (as many solid ingredients and fluids) come first, intermediates
    before buildings, then the nearest shapes; by name within that, so the choice repeats.
    """
    row_name, row_recipe = row_machines(t)
    slots = segment_slots(t)
    own = recipe(row_recipe) if row_recipe else None

    def shape(r) -> Tuple[int, int, int]:
        return (sum(i not in r.fluids for i in r.ingredients), sum(i in r.fluids for i in r.ingredients),
                sum(i in r.fluids for i in r.results))

    target = shape(own) if own else (1, 0, 0)  # a furnace smelts one solid
    picks = []
    for name in sorted(RECIPES):
        if name == row_recipe or name.endswith("-barrel"):
            continue
        r = recipe(name)
        solids = [i for i in r.ingredients if i not in r.fluids]
        if not solids and machine_family(row_name) != "chem_row":
            continue
        if _machine_for(r, row_name) is None or len(solids) > slots["max_item_inputs"]:
            continue
        if sum(i in r.fluids for i in r.ingredients) > slots["max_fluid_in"] or \
                sum(i in r.fluids for i in r.results) > slots["max_fluid_out"]:
            continue
        if not any(i not in r.fluids for i in r.results) and machine_family(row_name) != "chem_row":
            continue
        s = shape(r)
        picks.append((s != target, sum(abs(a - b) for a, b in zip(s, target)), not r.productivity, name))
    return [name for *_, name in sorted(picks)[:limit]]


# --- The template x recipe check (step 2b.5) --------------------------------------------------------

REPORT = "outputs/templates/generic.md"


def rows(catalog: Dict[str, List[Template]]) -> List[Tuple[str, int, Template]]:
    """Catalog templates that are rows of one recipe on crafting machines (chains and cells of several recipes are not)."""
    out = []
    for cat, ts in catalog.items():
        for i, t in enumerate(ts):
            if not t.metrics.get("family"):
                continue
            try:
                name, own = row_machines(t)
            except Incompatible:
                continue
            if len({e.extras_dict.get("recipe") for e in t.segment if entity_type(e.name) in MACHINE_TYPES}) != 1:
                continue
            try:  # a chain (ore -> plates -> steel in one column) does not take even its own recipe as a row
                with_recipe(t, own or recipe_of_smelt(t) or "")
            except Incompatible:
                continue
            out.append((cat, i, t))
    return out


def segment_form(t: Template) -> bool:
    """Whether the template repeats a segment (head + segments + tail) rather than whole copies of a cell."""
    return t.metrics.get("kind") == "segment" or bool(t.head or t.tail or t.repeats > 1)


def _check(job: Tuple[str, int, str]) -> Dict[str, Any]:
    """One template with one recipe (a worker of check_catalog)."""
    from factoriollm.template_miner import load_catalog
    cat, i, name = job
    t = load_catalog()[cat][i]
    try:
        g = instantiate(t, name, segments=2 if segment_form(t) else 1)
        return {"cat": cat, "i": i, "recipe": name, "ok": True, "ends": g.ends, "machine": g.machine,
                "throughput": throughput(g), "why": "; ".join(g.notes)}
    except Incompatible as e:
        return {"cat": cat, "i": i, "recipe": name, "ok": False, "why": str(e)}


def check_catalog(workers: int = 0, limit: int = 4) -> List[Dict[str, Any]]:
    """Every row template with its author's recipe (the control) and up to `limit` recipes of its shape."""
    import multiprocessing

    from factoriollm import generic as this
    from factoriollm.template_miner import load_catalog

    jobs = []
    for cat, i, t in rows(load_catalog()):
        own = row_machines(t)[1] or recipe_of_smelt(t)
        jobs += [(cat, i, name) for name in [own] + shaped_like(t, limit) if name]
    workers = workers or max(1, (multiprocessing.cpu_count() or 2) - 1)
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap(this._check, jobs, chunksize=1))


def recipe_of_smelt(t: Template) -> Optional[str]:
    """The smelting recipe a furnace column was mined with (its smelting input), the control of the check."""
    from factoriollm.game_data import SMELTING
    product = SMELTING.get(t.metrics.get("smelt") or "")
    return next((n for n, r in RECIPES.items() if r["category"] == "smelting" and product in r["results"]), None)


def write_report(results: List[Dict[str, Any]], path: str = REPORT) -> None:
    """The template x recipe table: what works with which ends, and why the rest does not."""
    from pathlib import Path

    from factoriollm.template_miner import load_catalog
    catalog = load_catalog()
    by_template: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in results:
        by_template.setdefault((r["cat"], r["i"]), []).append(r)
    lines = ["# Общие шаблоны: шаблон × рецепт", "",
             "Каждый рядовой шаблон каталога собран с рецептом автора (контроль, первый) и с рецептами той же формы.",
             "Проверка: симулятор на 1 и 2 сегментах (один предмет на полосу), валидатор на 1, 2 и 5 сегментах.",
             "`own` — свои концы (голые сегменты), `author` — голова и хвост автора.", ""]
    total = Counter()
    for r in results:
        total["ok" if r["ok"] else "no"] += 1
        if r["ok"]:
            total[f"ends {r['ends']}"] += 1
    lines += ["| Итог | Число |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in sorted(total.items())] + [""]
    family = Counter()
    for (cat, i), rs in by_template.items():
        t = catalog[cat][i]
        control = rs[0]
        works = sum(r["ok"] for r in rs[1:])
        family[(t.metrics.get("family"), "control ok" if control["ok"] else "control fails")] += 1
        family[(t.metrics.get("family"), f"substitutes {works}/{len(rs) - 1}")] += 1
    lines += ["| Семейство | Итог | Шаблонов |", "| --- | --- | --- |"]
    lines += [f"| {f} | {k} | {n} |" for (f, k), n in sorted(family.items())] + [""]
    lines += ["## По шаблонам", "", "| Категория | Шаблон | Сечение | Рецепт | Итог |", "| --- | --- | --- | --- | --- |"]
    for (cat, i), rs in sorted(by_template.items()):
        t = catalog[cat][i]
        name = (t.source.get("label") or t.source.get("title") or "").replace("|", "/")[:40]
        for r in rs:
            verdict = f"ok ({r['ends']})" if r["ok"] else "нет: " + r["why"].replace("|", "/")[:160]
            lines.append(f"| {cat} | {name} | {t.metrics.get('topology')} | {r['recipe']} | {verdict} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[*] Report: {path}")


BOOK = "outputs/templates/generic_book.txt"
FAMILY_NAMES = {"machine_row": "Ряды сборщиков", "furnace_column": "Колонны печей", "chem_row": "Ряды химзаводов",
                "centrifuge_row": "Центрифуги", "refinery_block": "НПЗ"}


def write_test_book(results: List[Dict[str, Any]], per_family: int = 4, path: str = BOOK) -> List[str]:
    """
    A book to check substitutions in game: per family the most popular row templates that took
    other recipes, each with up to two of them (our own ends first), built with labels. Returns the
    lines of its table of contents (sub-book, position, what was built).
    """
    from pathlib import Path

    from draftsman.blueprintable import BlueprintBook, get_blueprintable_from_string

    from factoriollm.helpers import game_string
    from factoriollm.template_miner import blueprint_string, load_catalog

    catalog = load_catalog()
    works: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in results:
        if r["ok"] and r["recipe"] != (row_machines(catalog[r["cat"]][r["i"]])[1] or recipe_of_smelt(catalog[r["cat"]][r["i"]])):
            works.setdefault((r["cat"], r["i"]), []).append(r)
    book = BlueprintBook()
    book.label = "FactorioLLM generic templates"
    toc = []
    for family, title in FAMILY_NAMES.items():
        chosen = sorted((k for k in works if catalog[k[0]][k[1]].metrics.get("family") == family),
                        key=lambda k: -(catalog[k[0]][k[1]].source.get("favorites") or 0))[:per_family]
        if not chosen:
            continue
        sub = BlueprintBook()
        sub.label = title
        for k in chosen:
            t = catalog[k[0]][k[1]]
            name = (t.source.get("label") or t.source.get("title") or "").strip()[:30]
            for r in sorted(works[k], key=lambda r: r["ends"] != "own")[:2]:
                try:
                    g = instantiate(t, r["recipe"], segments=3 if segment_form(t) else 1)
                    text = f"{name} | {r['recipe']} | {g.ends} ends"
                    sub.blueprints.append(get_blueprintable_from_string(blueprint_string(g.build, text)))
                    toc.append(f"{len(book.blueprints) + 1} «{title}» #{len(sub.blueprints)}: {text}")
                except Exception as e:  # reported, not hidden
                    print(f"    [!] {name} with {r['recipe']}: {type(e).__name__}: {e}")
        book.blueprints.append(sub)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(game_string(book), encoding="utf-8")
    print(f"[*] Test book: {path}")
    return toc


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Check every row template of the catalog with other recipes")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4, help="recipes of the template's shape to try")
    args = parser.parse_args()
    results = check_catalog(args.workers, args.limit)
    Path("data/templates/generic_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    write_report(results)
    print("\n".join(write_test_book(results)))


if __name__ == "__main__":
    main()
