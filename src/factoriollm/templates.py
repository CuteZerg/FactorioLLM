"""
Layout templates mined from real player blueprints.

A template is a tileable design cut into three parts along its repeat axis (always x after
normalisation; items usually flow along it):

    head | segment | segment | ... | segment | tail

The segment is one period of the design; head and tail are the one-off parts before the first and
after the last full period (a lane-loading junction, end poles, a belt turn-around). Building a
template with N segments reproduces the player's design at any length. All coordinates are integer
tiles in the IR convention (top-left corner of the entity footprint).

This module is pure data processing on the IR (no database, no execution of blueprint content).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from factoriollm.blueprint_ir import IRBlueprint, IREntity, _prototype, entity_type, footprint, normalized, rotated
from factoriollm.game_data import SMELTING, recipe
from factoriollm.validator import _center, _rotate, is_electric_consumer, validate

# Entities whose count defines "how much" a design produces; used for density metrics.
MACHINE_TYPES = frozenset({
    "furnace", "assembling-machine", "lab", "mining-drill", "solar-panel", "accumulator", "boiler",
    "generator", "reactor", "rocket-silo", "beacon",
})
# Kinds of entities by prototype type (so entities of mods fall into place):
# cosmetic entities are not compared when looking for the repeating part (lamps become an optional
# layer of the template; author labels and radars are removed by prepare()); the author's defence
# around a production block is removed too (defence is a layer of its own, `defense` category).
DECOR_TYPES = frozenset({"lamp", "constant-combinator", "display-panel"})
AUTHOR_ONLY_TYPES = frozenset({"radar", "constant-combinator", "display-panel"})
DEFENSE_TYPES = frozenset({"wall", "gate", "ammo-turret", "electric-turret", "fluid-turret", "artillery-turret",
                           "land-mine"})
PIPE_TYPES = frozenset({"pipe", "pipe-to-ground", "storage-tank", "pump"})
LAMP = "small-lamp"
MIN_RUN_COVERAGE = 0.4  # the repeating part must hold at least this share of the entities
# Belt and pipe tiles kept in front of an input / behind an output once the author's surroundings
# (bus taps, mains, long feeds) are cut.
CROP_MARGIN = 2


def is_decor(name: str) -> bool:
    return entity_type(name) in DECOR_TYPES


def is_defense(name: str) -> bool:
    return entity_type(name) in DEFENSE_TYPES


def is_pipe(name: str) -> bool:
    return entity_type(name) in PIPE_TYPES


def _belt_tiers() -> Tuple[Tuple[str, str, str], ...]:
    """(belt, underground, splitter) per tier, slowest first, by the belts' speed in the game data."""
    from factoriollm.base_game import ENTITIES
    by_speed: Dict[float, Dict[str, str]] = {}
    for name, e in ENTITIES.items():
        if e["type"] in ("transport-belt", "underground-belt", "splitter") and e.get("speed"):
            by_speed.setdefault(e["speed"], {})[e["type"]] = name
    return tuple((t["transport-belt"], t["underground-belt"], t["splitter"]) for _, t in sorted(by_speed.items())
                 if len(t) == 3)


BELT_TIERS = _belt_tiers()
_TIER = {name: (tier, kind) for tier, names in enumerate(BELT_TIERS) for kind, name in enumerate(names)}
_DIRS = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}
# Validator categories that are never allowed in a template, whatever the source had.
POWER_CATEGORIES = frozenset({"pole_islands", "unpowered", "no_poles"})


def _key(e: IREntity) -> Tuple[Any, ...]:
    return (e.name, e.direction, e.extras)


def _span(e: IREntity) -> Tuple[int, int]:
    """[start, end) of the entity footprint along x."""
    return e.x, e.x + footprint(e.name, e.direction)[0]


def _structural(ir: IRBlueprint) -> List[IREntity]:
    return [e for e in ir.entities if not is_decor(e.name)]


def _is_belt(e: IREntity) -> bool:
    return entity_type(e.name) == "transport-belt"


def _inner_belts(belts: Dict[int, set], run: List[int], period: int) -> frozenset:
    """
    Belts of a repeating cell, taken where no belt line starts or ends: an inner cell of the run
    if there is one; with two cells, the left half from the second cell and the right half from
    the first (each half borders the other cell, not the outside).
    """
    if len(run) >= 3:
        return frozenset(belts.get(run[1], ()))
    first, last = belts.get(run[0], set()), belts.get(run[1], set())
    return frozenset({b for b in last if b[1] < period / 2} | {b for b in first if b[1] >= period / 2})


def _best_run(ents: List[IREntity], period: int) -> Tuple[int, frozenset, List[int]]:
    """
    Tries every placement of the cell grid and returns (offset, cell content, run) for the longest
    run of consecutive cells that are identical apart from plain belts (belt lines start and end
    inside the first and last cell). The returned content holds the shared entities plus the
    belts away from the run's outer edges (see _inner_belts). Ties go to the placement with fewer
    entities crossing cell borders.
    """
    x0 = min(e.x for e in ents)

    def crossings(offset: int) -> int:  # cell borders lying inside entity footprints
        return sum((_span(e)[1] - 1 - offset) // period - (_span(e)[0] - offset) // period for e in ents)

    best: Tuple[int, int, int, frozenset, List[int]] = (0, 0, x0, frozenset(), [])
    for offset in range(x0, x0 + period):
        core: Dict[int, set] = {}
        belts: Dict[int, set] = {}
        for e in ents:
            m = (e.x - offset) // period
            (belts if _is_belt(e) else core).setdefault(m, set()).add((_key(e), e.x - offset - m * period, e.y))
            core.setdefault(m, set())
        frozen = {m: frozenset(c) for m, c in core.items()}
        run: List[int] = []
        for m in sorted(frozen):
            run = run + [m] if run and m == run[-1] + 1 and frozen[m] == frozen[run[-1]] else [m]
            if len(run) < 2 or not frozen[m]:
                continue
            content = frozen[m] | _inner_belts(belts, run, period)
            score = (len(run) * len(content), -crossings(offset))
            if score > best[:2]:
                best = (score[0], score[1], offset, content, list(run))
    return best[2], best[3], best[4]


def find_period(ir: IRBlueprint) -> Optional[int]:
    """
    Smallest period along x whose longest run of identical consecutive cells has at least two
    cells and covers at least MIN_RUN_COVERAGE of the (non-decorative) entities.
    """
    ents = _structural(ir)
    if len(ents) < 6:
        return None
    width = max(_span(e)[1] for e in ents) - min(e.x for e in ents)
    for p in range(1, width // 2 + 1):
        _, common, run = _best_run(ents, p)
        if len(run) >= 2 and len(run) * len(common) >= MIN_RUN_COVERAGE * len(ents):
            return p
    return None


@dataclass
class Template:
    """A tileable design: head + segment * n + tail along x, with provenance and metrics."""
    category: str
    period: int
    head: List[IREntity]
    segment: List[IREntity]  # x in [0, period) (an entity may extend into the next period)
    tail: List[IREntity]     # positioned after `repeats` segments starting at x = 0
    repeats: int             # segments in the source blueprint
    snap: Optional[Tuple[int, int]] = None  # grid of the segment form: (period, segment height)
    source: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_width: int = 0  # width of a whole-cell design (its copies are placed `period` apart)
    belt: Optional[str] = None  # belt tier every build is brought to (see retier)
    # optional lamp layer in the coordinates of head / segment / tail
    lamps: Dict[str, List[IREntity]] = field(default_factory=lambda: {"head": [], "segment": [], "tail": []})
    # modules per machine (or beacon) type, e.g. {"beacon": "speed-module-3=2"}, and per recipe where the
    # author used other modules for some products ("" = no modules); see normalize_modules
    modules: Dict[str, str] = field(default_factory=dict)
    modules_by_recipe: Dict[str, str] = field(default_factory=dict)

    def _place(self, parts: Dict[str, List[IREntity]], segments: int) -> List[IREntity]:
        ents = list(parts["head"])
        for k in range(segments):
            ents += [IREntity(e.name, e.x + k * self.period, e.y, e.direction, e.extras) for e in parts["segment"]]
        shift = (segments - self.repeats) * self.period
        ents += [IREntity(e.name, e.x + shift, e.y, e.direction, e.extras) for e in parts["tail"]]
        return list(dict.fromkeys(ents))  # overlapping copies may share identical entities

    def build(self, segments: int, lamps: bool = False, belt: Optional[str] = None,
              modules: Optional[Dict[str, str]] = None, modules_by_recipe: Optional[Dict[str, str]] = None) -> IRBlueprint:
        """
        The design with `segments` periods (head and tail kept), at the template's belt tier or at
        `belt`, optionally with the author's lamps (only on free tiles that a pole reaches). Modules:
        `modules` per machine type ({"assembling-machine-3": "productivity-module-3=4"}, "" for none)
        and `modules_by_recipe` override the template's own, every machine of a kind alike.
        """
        if segments < 1:
            raise ValueError("a template needs at least one segment")
        ents = self._place(self._tiered_parts(belt or self.belt), segments)
        if modules or modules_by_recipe:
            ents = _set_modules(ents, {**self.modules, **(modules or {})},
                                {**self.modules_by_recipe, **(modules_by_recipe or {})})
        if lamps:
            taken = {t for e in ents for t in _tiles(e)}
            free = [e for e in self._place(self.lamps, segments) if (e.x, e.y) not in taken]
            dark = set(_uncovered(ents + free)) - set(_uncovered(ents))  # lamps no pole reaches
            ents += [e for k, e in enumerate(free) if len(ents) + k not in dark]
        return normalized(IRBlueprint(ents))

    def _tiered_parts(self, belt: Optional[str]) -> Dict[str, List[IREntity]]:
        """
        Head, segment and tail at belt tier `belt` (see retier), decided once on three copies: an
        underground pair may join two neighbouring copies (a weave between copies of a cell), which a
        single copy does not show. The segment takes the tiers of the middle copy, so every build,
        whatever its length, has the same tiers.
        """
        parts = {"head": self.head, "segment": self.segment, "tail": self.tail}
        if not belt:
            return parts
        cache = self.__dict__.setdefault("_tier_cache", {})
        if belt not in cache:
            copies = 3
            raw = self._place(parts, copies)
            name_at = {(e.x, e.y): t.name for e, t in zip(raw, retier(raw, belt)) if e.name in _TIER}
            shift = {"head": 0, "segment": self.period, "tail": (copies - self.repeats) * self.period}
            tiered = {part: [IREntity(name_at.get((e.x + shift[part], e.y), e.name) if e.name in _TIER else e.name,
                                      e.x, e.y, e.direction, e.extras) for e in ents]
                      for part, ents in parts.items()}
            # head and tail took their tiers next to copies 0 and 2, every segment those of copy 1:
            # the assembled parts must still pair their undergrounds as the design does
            if underground_pairs(self._place(tiered, copies)) != underground_pairs(raw):
                raise ValueError(f"no belt tiers at {belt} keep the undergrounds of every copy paired as in the design")
            cache[belt] = tiered
        return cache[belt]

    def build_segment(self, belt: Optional[str] = None) -> IRBlueprint:
        """One segment on its own grid (relative snapping, cell = period x segment height), to tile in game."""
        ents = list(self._tiered_parts(belt or self.belt)["segment"])
        y0 = min(e.y for e in ents)
        return IRBlueprint([IREntity(e.name, e.x, e.y - y0, e.direction, e.extras) for e in ents], snap=self.snap)

    def to_json(self) -> Dict[str, Any]:
        def pack(ents):
            return [[e.name, e.x, e.y, e.direction, dict(e.extras)] for e in ents]
        return {"category": self.category, "period": self.period, "repeats": self.repeats,
                "snap": list(self.snap) if self.snap else None, "source": self.source, "metrics": self.metrics,
                "belt": self.belt, "head": pack(self.head), "segment": pack(self.segment), "tail": pack(self.tail),
                "lamps": {part: pack(ents) for part, ents in self.lamps.items()},
                "modules": self.modules, "modules_by_recipe": self.modules_by_recipe}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Template":
        def unpack(rows):
            return [IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in rows]
        lamps = {part: unpack(rows) for part, rows in (data.get("lamps") or {}).items()}
        return Template(data["category"], data["period"], unpack(data["head"]), unpack(data["segment"]),
                        unpack(data["tail"]), data["repeats"], tuple(data["snap"]) if data.get("snap") else None,
                        data.get("source") or {}, data.get("metrics") or {}, belt=data.get("belt"),
                        lamps={"head": [], "segment": [], "tail": [], **lamps},
                        modules=data.get("modules") or {}, modules_by_recipe=data.get("modules_by_recipe") or {})


def _set_modules(ents: List[IREntity], by_type: Dict[str, str], by_recipe: Dict[str, str]) -> List[IREntity]:
    """Entities with module slots get the modules of their recipe, else of their type (unlisted ones keep theirs)."""
    out = []
    for e in ents:
        ex = e.extras_dict
        r = ex.get("recipe")
        if not (_prototype(e.name) or {}).get("module_slots") or (r not in by_recipe and e.name not in by_type):
            out.append(e)
            continue
        want = by_recipe[r] if r in by_recipe and _fits(e.name, by_recipe[r]) else by_type.get(e.name)
        if want is None or not _fits(e.name, want):
            out.append(e)
            continue
        ex = {k: v for k, v in ex.items() if k != "modules"}
        if want:
            ex["modules"] = want
        out.append(IREntity(e.name, e.x, e.y, e.direction, tuple(sorted(ex.items()))))
    return out


def _fits(name: str, modules: str) -> bool:
    """Whether a module set ('speed-module-3=2;...') fits the entity's slots (beacons take no productivity)."""
    counts = {m.split("=")[0]: int(m.split("=")[1]) for m in modules.split(";") if m}
    slots = int((_prototype(name) or {}).get("module_slots") or 0)
    if entity_type(name) == "beacon" and any("productivity" in m for m in counts):
        return False
    return sum(counts.values()) <= slots


def normalize_modules(t: Template) -> Template:
    """
    The same modules for every machine of a kind: the most common set per machine type (and for
    beacons), with per-recipe exceptions where the author chose other modules for some products
    (productivity on gears, speed on circuits). Odd machines (3 labs of 152 without modules) follow
    their group.
    """
    groups: Dict[Tuple[str, Optional[str]], Counter] = {}
    for e in t.head + t.segment + t.tail:
        if (_prototype(e.name) or {}).get("module_slots"):
            groups.setdefault((e.name, e.extras_dict.get("recipe")), Counter())[e.extras_dict.get("modules", "")] += 1
    if not groups:
        return t
    by_type: Dict[str, Counter] = {}
    for (name, _), counts in groups.items():
        by_type.setdefault(name, Counter()).update(counts)
    modules = {name: counts.most_common(1)[0][0] for name, counts in by_type.items()}
    by_recipe = {}
    for (name, r), counts in groups.items():
        top = counts.most_common(1)[0][0]
        if r and top != modules[name]:
            by_recipe[r] = top
    fixed = replace(t, modules=modules, modules_by_recipe=by_recipe)
    for part in ("head", "segment", "tail"):
        setattr(fixed, part, _set_modules(getattr(t, part), modules, by_recipe))
    return fixed


def _tiles(e: IREntity) -> List[Tuple[int, int]]:
    w, h = footprint(e.name, e.direction)
    return [(e.x + dx, e.y + dy) for dx in range(w) for dy in range(h)]


def _height(ents: Iterable[IREntity]) -> int:
    ents = list(ents)
    return max(e.y + footprint(e.name, e.direction)[1] for e in ents) - min(e.y for e in ents)


def _shifted(ents: Iterable[IREntity], dx: int) -> List[IREntity]:
    return [IREntity(e.name, e.x + dx, e.y, e.direction, e.extras) for e in ents]


def extract(ir: IRBlueprint, category: str, period: Optional[int] = None) -> Optional[Template]:
    """
    Cuts a design that repeats along x into head, segment and tail. Labels inside the repeating
    part are dropped and lamps become the optional lamp layer; the cut must rebuild the rest of
    the source exactly. The author's grid is not carried over: the segment gets its own
    (period x segment height).
    """
    ir = normalized(IRBlueprint(ir.entities))
    period = period or find_period(ir)
    if not period:
        return None
    offset, common, run = _best_run(_structural(ir), period)
    if len(run) < 2:
        return None
    start, end = offset + run[0] * period, offset + (run[-1] + 1) * period
    segment = sorted((IREntity(k[0], u, y, k[1], k[2]) for k, u, y in common), key=lambda e: (e.y, e.x, e.name))
    solid = [e for e in ir.entities if e.name != LAMP]
    lamps = [e for e in ir.entities if e.name == LAMP]
    head = _shifted((e for e in solid if e.x < start), -start)
    tail = _shifted((e for e in solid if e.x >= end), -start)
    template = Template(category, period, head, segment, tail, len(run), (period, _height(segment)))
    template.lamps = {
        "head": _shifted((e for e in lamps if e.x < start), -start),
        "segment": sorted({IREntity(e.name, (e.x - start) % period, e.y) for e in lamps if start <= e.x < end},
                          key=lambda e: (e.y, e.x)),
        "tail": _shifted((e for e in lamps if e.x >= end), -start),
    }
    kept = normalized(ir.with_entities([e for e in solid if not is_decor(e.name) or e.x < start or e.x >= end]))
    rebuilt = template.build(len(run)).canonical_set()
    if not kept.canonical_set() <= rebuilt or any(not _is_belt(e) for e in rebuilt - kept.canonical_set()):
        return None  # the cut must rebuild the source; only belt tiles closing a line may be added
    if _joins_lines(list(kept.entities), list(rebuilt)):
        return None  # an added belt tile would join two lines of the source (an input into an output)
    return template


def _belt_lines(ents: List[IREntity]) -> Dict[IREntity, int]:
    """Belt-like entity -> id of its line (connected belts, undergrounds and splitters, any direction)."""
    tiles = {t: i for i, e in enumerate(ents) for t in _tiles(e)}
    belts, edges = _belt_graph(ents, tiles)
    parent = {i: i for i in belts}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, out in edges.items():
        for j in out:
            parent[root(i)] = root(j)
    return {ents[i]: root(i) for i in belts}


def _joins_lines(source: List[IREntity], rebuilt: List[IREntity]) -> bool:
    """Whether one belt line of the rebuilt design holds belts of two different lines of the source."""
    before = _belt_lines(source)
    seen: Dict[int, int] = {}
    for e, line in _belt_lines(rebuilt).items():
        if e in before:
            if seen.setdefault(line, before[e]) != before[e]:
                return True
    return False


def extract_cell(ir: IRBlueprint, category: str) -> Template:
    """
    A design without an inner repeat that is tiled by placing whole copies side by side: the
    copy pitch is the grid snapping size when the author set one, otherwise the width. Lamps go
    to the optional lamp layer.
    """
    width = normalized(ir).size[0]
    pitch = ir.snap[0] if ir.snap and ir.snap[0] <= width else width
    ir = normalized(IRBlueprint(ir.entities))
    solid = sorted((e for e in ir.entities if e.name != LAMP), key=lambda e: (e.y, e.x, e.name))
    template = Template(category, pitch, [], solid, [], 1, (pitch, _height(solid)))
    template.lamps["segment"] = [e for e in ir.entities if e.name == LAMP]
    template.metrics_width = width
    return template


def along_x(ir: IRBlueprint) -> List[IRBlueprint]:
    """The blueprint itself and, if it repeats along y, a copy rotated so that it repeats along x."""
    out = [normalized(ir)]
    if not ir.absolute_snap and not find_period(ir) and find_period(rotated(ir, 1)):
        out = [rotated(ir, 1)]
    return out


def tiling_check(template: Template, segments: Tuple[int, ...] = (1, 2, 5), docks: bool = False) -> Dict[str, Any]:
    """
    Builds the design with several segment counts and validates each build: no errors, and no
    warning categories beyond those the source blueprint already has (a belt end the author left
    on purpose stays allowed). Power is always strict: a module must not depend on poles outside it
    (POWER_CATEGORIES are never allowed, even if the author had them). `power_only` tells whether
    power was the only problem, i.e. whether repair_power may help. A cell made to dock with its
    copies (docks=True) is built 3 and 5 copies long and judged on its inner copies: the outer ones
    miss the neighbours that would power and feed their outer edge.
    """
    if docks:
        segments = (3, 5)

    def counted(report, n):
        if not docks:
            return report.errors, report.warnings
        inner = lambda issue: template.period <= issue.position[0] < (n - 1) * template.period  # noqa: E731
        return report.errors, [w for w in report.warnings if inner(w)]

    source = validate(template.build(template.repeats))
    allowed = set(source.category_counts()) - POWER_CATEGORIES
    problems, categories = [], set()
    for n in segments:
        errors, warnings = counted(validate(template.build(n)), n)
        if errors:
            problems.append(f"{n} segments: {errors[0].message}")
            categories |= {issue.category for issue in errors}
        new = {issue.category for issue in warnings} - allowed
        if new:
            problems.append(f"{n} segments: new warnings {sorted(new)}")
            categories |= new
    return {"ok": not problems, "problems": problems[:5], "source_warnings": sorted(set(source.category_counts())),
            "power_only": bool(categories) and categories <= POWER_CATEGORIES}


def repair_power(template: Template) -> Optional[Template]:
    """
    Re-places the segment's poles for a design that relied on poles outside the blueprint or had
    pole islands: poles go on free tiles of one period, with the neighbouring copies counted as
    occupied (helpers._add_poles: cover every consumer, then bridge into one network). The pole
    type is the design's most common one. Returns the repaired template, or None if no free tiles
    do; the caller checks tiling again (poles of neighbouring copies must reach each other).
    """
    from factoriollm.helpers import _Layout, _add_poles

    every = template.head + template.segment + template.tail
    poles = Counter(e.name for e in every if entity_type(e.name) == "electric-pole")
    pole = poles.most_common(1)[0][0] if poles else "medium-electric-pole"
    p = template.period
    fixed = template
    # the segment's own poles are kept unless a middle copy (between two neighbours) lacks power
    triple = list(dict.fromkeys(IREntity(e.name, e.x + k * p, e.y, e.direction, e.extras)
                                for k in range(3) for e in template.segment))
    has_poles = any(entity_type(e.name) == "electric-pole" for e in template.segment)
    if not has_poles or any(p <= triple[i].x < 2 * p for i in _uncovered(triple)):
        seg = [e for e in template.segment if entity_type(e.name) != "electric-pole"]
        copies = list(dict.fromkeys(IREntity(e.name, e.x + k * p, e.y, e.direction, e.extras) for k in range(3) for e in seg))
        if not copies:
            return None
        x0, y0 = min(e.x for e in copies), min(e.y for e in copies)
        layout = _Layout()
        middle = []
        try:
            for e in copies:
                idx = layout.add(e.name, e.x - x0, e.y - y0, e.direction or None)
                if p <= e.x < 2 * p and is_electric_consumer(e.name):
                    middle.append(idx)
            layout.pole_tiles = {(u, v) for u in range(p - x0, 2 * p - x0) for v in range(layout.height)}
            _add_poles(layout, pole, middle)
        except (ValueError, RuntimeError):
            return None
        added = [IREntity(name, u + x0 - p, v + y0) for name, u, v, _, _ in layout.items[len(copies):]]
        fixed = replace(template, segment=sorted(seg + added, key=lambda e: (e.y, e.x, e.name)))
    # head and tail: consumers the author powered from neighbouring segments get poles of their own
    shift = (1 - fixed.repeats) * p
    for part, dx in (("head", 0), ("tail", shift)):
        own = getattr(fixed, part)
        if not own:
            continue
        built = fixed._place({"head": fixed.head, "segment": fixed.segment, "tail": fixed.tail}, 1)
        mine = {IREntity(e.name, e.x + dx, e.y, e.direction, e.extras) for e in own}
        need = [i for i in _uncovered(built) if built[i] in mine]
        if not need:
            continue
        xs = [e.x for e in mine]
        bx0, by0 = min(e.x for e in built), min(e.y for e in built)
        layout = _Layout()
        try:
            for e in built:
                layout.add(e.name, e.x - bx0, e.y - by0, e.direction or None)
            region = range(min(xs) - bx0, max(e.x + footprint(e.name, e.direction)[0] for e in mine) - bx0)
            layout.pole_tiles = {(u, v) for u in region for v in range(layout.height)}
            _add_poles(layout, pole, need)
        except (ValueError, RuntimeError):
            return None
        extra = [IREntity(name, u + bx0 - dx, v + by0) for name, u, v, _, _ in layout.items[len(built):]]
        fixed = replace(fixed, **{part: own + extra})
    return fixed


def _supplies(pole: IREntity, e: IREntity) -> bool:
    """Whether the entity's footprint touches the pole's supply area (as the validator counts it)."""
    d = float((_prototype(pole.name) or {}).get("supply_area_distance") or 0)
    cx, cy = _center(pole)
    w, h = footprint(e.name, e.direction)
    return e.x < cx + d and e.x + w > cx - d and e.y < cy + d and e.y + h > cy - d


def _uncovered(ents: List[IREntity]) -> List[int]:
    """Indices of electric consumers outside every pole's supply area."""
    poles = [e for e in ents if entity_type(e.name) == "electric-pole"]
    return [i for i, e in enumerate(ents) if is_electric_consumer(e.name) and not any(_supplies(p, e) for p in poles)]


def metrics(template: Template) -> Dict[str, Any]:
    """
    Per-segment machine counts, size, density (tiles per production machine; beacons are counted
    separately) and the variant traits a user may ask for: beacons, belt / machine tiers, poles.
    """
    machines, recipes, beacons, belts, poles = Counter(), Counter(), 0, Counter(), Counter()
    for e in template.segment:
        etype = entity_type(e.name)
        if e.name == "beacon":
            beacons += 1
        elif etype in MACHINE_TYPES:
            machines[e.name] += 1
        if e.extras_dict.get("recipe"):
            recipes[e.extras_dict["recipe"]] += 1
        if etype in ("transport-belt", "underground-belt", "splitter"):
            belts[e.name.split("-")[0] if e.name.split("-")[0] in ("fast", "express") else "yellow"] += 1
        if etype == "electric-pole":
            poles[e.name] += 1
    every = template.head + template.segment + template.tail
    height = max(e.y + footprint(e.name, e.direction)[1] for e in every) - min(e.y for e in every)
    total = sum(machines.values())
    return {
        "segment_size": [template.period, height],
        "machines_per_segment": dict(machines),
        "recipes_per_segment": dict(recipes),
        "tiles_per_machine": round(template.period * height / total, 2) if total else None,
        "beacons_per_machine": round(beacons / total, 2) if total else None,
        "belt_tier": ({"transport-belt": "yellow", "fast-transport-belt": "fast", "express-transport-belt": "express"}
                      [template.belt] if template.belt else belts.most_common(1)[0][0] if belts else None),
        "lamps": sum(len(ents) for ents in template.lamps.values()),
        "poles": dict(poles),
        "head_entities": len(template.head),
        "tail_entities": len(template.tail),
    }


def signature(template: Template) -> Tuple[Any, ...]:
    """Identity of a design for de-duplication: period and segment content."""
    return (template.period, frozenset((_key(e), e.x, e.y) for e in template.segment))


# --- Self-contained templates -------------------------------------------------------------

def underground_pairs(ents: List[IREntity], names: Optional[List[str]] = None) -> Dict[int, int]:
    """
    Underground belt pairs as {entrance index: exit index}. An entrance connects to the nearest
    underground of the same name ahead within reach if that one is an exit in the same direction;
    any other underground of that name in between (another entrance, or one going the other way)
    breaks the connection: same-tier undergrounds cannot be woven, so players use another tier even
    for a line going the other way. `names` overrides entity names (to test another belt tier
    without building new entities).
    """
    names = names or [e.name for e in ents]
    at = {(e.x, e.y): i for i, e in enumerate(ents) if entity_type(names[i]) == "underground-belt"}
    pairs: Dict[int, int] = {}
    for i in at.values():
        e = ents[i]
        if e.extras_dict.get("io_type", "input") != "input":
            continue
        dx, dy = _DIRS[e.direction]
        for k in range(1, int((_prototype(names[i]) or {}).get("max_distance") or 5) + 1):
            j = at.get((e.x + k * dx, e.y + k * dy))
            if j is not None and names[j] == names[i] and ents[j].direction % 8 == e.direction % 8:
                if ents[j].direction == e.direction and ents[j].extras_dict.get("io_type") == "output":
                    pairs[i] = j
                break  # an underground across the line (another axis) does not interact
    return pairs


def has_weaving(ents: List[IREntity]) -> bool:
    """
    Belt weaving: an underground pair passes under another underground on the same line (either way;
    it needs another belt tier).
    """
    ugs = {(e.x, e.y): e for e in ents if entity_type(e.name) == "underground-belt"}
    for i, o in underground_pairs(ents).items():
        a, b = ents[i], ents[o]
        dx, dy = _DIRS[a.direction]
        span = abs(b.x - a.x) + abs(b.y - a.y)
        if any((a.x + k * dx, a.y + k * dy) in ugs and ugs[(a.x + k * dx, a.y + k * dy)].direction % 8 == a.direction % 8
               for k in range(1, span)):
            return True
    return False


def main_belt(ents: Iterable[IREntity]) -> Optional[str]:
    """Transport belt name of the most common belt tier (by belt, underground and splitter count)."""
    tiers = Counter(_TIER[e.name][0] for e in ents if e.name in _TIER)
    return BELT_TIERS[tiers.most_common(1)[0][0]][0] if tiers else None


def retier(ents: List[IREntity], belt: str) -> List[IREntity]:
    """
    Every belt, underground and splitter at the tier of `belt` (a transport belt name), keeping
    every underground pair exactly as it was. A pair that would connect differently at that tier
    (a weave under or over another pair, or a span beyond the tier's reach) gets the closest other
    tier that keeps it, so belt weaving survives. Raises ValueError if no tier keeps a pair.
    """
    if belt not in _TIER or _TIER[belt][1] != 0:
        raise ValueError(f"'{belt}' is not a transport belt; use one of {[t[0] for t in BELT_TIERS]}")
    target = _TIER[belt][0]
    tiers = [target if e.name in _TIER else None for e in ents]
    for cluster in _underground_clusters(ents):
        members = [i for unit in cluster for i in unit]
        local = [ents[i] for i in members]
        index = {i: k for k, i in enumerate(members)}
        want = underground_pairs(local)  # the design's pairs inside the cluster
        best: Optional[Tuple[Tuple[int, int, int], Tuple[int, ...]]] = None
        if len(cluster) <= MAX_CLUSTER_UNITS:
            combos: Iterable[Tuple[int, ...]] = itertools.product(range(len(BELT_TIERS)), repeat=len(cluster))
        else:  # too many for every combination: the target, the dominant tier moved to the target, the design's own
            own = tuple(_TIER[ents[unit[0]].name][0] for unit in cluster)
            major = Counter(own).most_common(1)[0][0]
            combos = [(target,) * len(cluster), tuple(target if t == major else t for t in own), own]
        for combo in combos:
            names = [None] * len(local)
            for unit, tier in zip(cluster, combo):
                for i in unit:
                    names[index[i]] = BELT_TIERS[tier][1]
            if underground_pairs(local, names) == want:
                key = (sum(t != target for t in combo), sum(abs(t - target) for t in combo), -sum(combo))
                if best is None or key < best[0]:
                    best = (key, combo)
        if best is None:
            e = local[0]
            raise ValueError(f"no belt tiers keep the undergrounds near {(e.x, e.y)} connected as in the design")
        for unit, tier in zip(cluster, best[1]):
            for i in unit:
                tiers[i] = tier
    return [IREntity(BELT_TIERS[t][_TIER[e.name][1]] if t is not None else e.name, e.x, e.y, e.direction, e.extras)
            for e, t in zip(ents, tiers)]


MAX_CLUSTER_UNITS = 7  # every tier combination (3^7) up to this size, three candidates above


def _underground_clusters(ents: List[IREntity]) -> List[List[Tuple[int, ...]]]:
    """
    Groups of undergrounds that can affect each other's connections: pairs (and lone undergrounds,
    which might connect after a tier change) on one line in one direction whose spans overlap.
    A lone entrance looks ahead by the longest reach; a lone exit can be found from that far behind.
    """
    pairs = underground_pairs(ents)
    partner = {**pairs, **{o: i for i, o in pairs.items()}}
    reach = max(int((_prototype(t[1]) or {}).get("max_distance") or 0) for t in BELT_TIERS)
    lines: Dict[Tuple[int, int], List[Tuple[int, int, Tuple[int, ...]]]] = {}
    seen: Set[int] = set()
    for i, e in enumerate(ents):
        if i in seen or entity_type(e.name) != "underground-belt":
            continue
        unit = (i, partner[i]) if i in partner else (i,)
        seen.update(unit)
        vertical = e.direction in (0, 8)
        along = [ents[k].y if vertical else ents[k].x for k in unit]
        if len(unit) == 2:
            lo, hi = min(along), max(along)
        else:
            ahead = (1 if e.direction in (4, 8) else -1) * (1 if e.extras_dict.get("io_type", "input") == "input" else -1)
            lo, hi = sorted((along[0], along[0] + ahead * reach))
        lines.setdefault((e.direction % 8, e.x if vertical else e.y), []).append((lo, hi, unit))  # both ways interact
    clusters: List[List[Tuple[int, ...]]] = []
    for spans in lines.values():
        spans.sort()
        end = None
        for lo, hi, unit in spans:
            if end is None or lo > end:
                clusters.append([])
                end = hi
            clusters[-1].append(unit)
            end = max(end, hi)
    return clusters


def author_signs(ir: IRBlueprint) -> List[Tuple[IREntity, Tuple[int, int], FrozenSet[str]]]:
    """
    Filter inserters an author set up as signs: the inserter picks from an empty tile nothing ever
    drops onto, so it can never move anything, and drops onto a belt (often right next to it), the
    ground or something without an inventory; its filter names the item of the belt by it. (One
    dropping into a chest may load from a wagon at a station: not a sign.) Returns (inserter, drop
    tile, items).
    """
    from factoriollm.flow_sim import Layout
    from factoriollm.validator import has_item_interface

    if not any(e.extras_dict.get("filters") for e in ir.entities):
        return []
    L = Layout(ir)
    dropped = {dst for _, dst in L.links if dst[0] == "ground"}
    out = []
    for idx, (k, (src, dst)) in enumerate(zip(L.inserters, L.links)):
        flt = L.filters[idx]
        if flt is None or not flt[0] or src[0] != "ground" or ("ground", src[1]) in dropped:
            continue
        if dst[0] == "entity" and has_item_interface(L.ents[dst[1]].name):
            continue
        at = L.ents[dst[1][0]] if dst[0] == "lane" else L.ents[k]  # the belt it names, else the sign itself
        out.append((L.ents[k], (at.x, at.y), flt[1]))
    return out


def prepare(ir: IRBlueprint, margin: Optional[int] = CROP_MARGIN, keep_defense: bool = False,
            cut_taps: bool = True) -> IRBlueprint:
    """
    A self-contained copy of a player's design. Radars and author labels go (signs made of filter
    inserters too, see author_signs; new labels come from the flow simulator, see add_port_labels),
    and so does the author's defence (unless the design is defence). Belts and pipes are cut by flow,
    not by a rectangle (see _trim_tails); margin=None keeps them all. Lamps stay; templates keep them
    as an optional layer. See CROP_STEPS for gentler cuts.
    """
    signs = {e for e, _, _ in author_signs(ir)}
    ents = [e for e in ir.entities if entity_type(e.name) not in AUTHOR_ONLY_TYPES and e not in signs
            and (keep_defense or not is_defense(e.name))]
    if margin is not None and ents:
        ents = _trim_tails(ents, margin, cut_taps)
    return ir.with_entities(ents)


def strip_unused(ir: IRBlueprint) -> IRBlueprint:
    """
    Removes what the design could never use, whatever comes in (not just what one input assignment
    leaves empty: the simulator does not count throughput, so a second feed of the same item is not
    spare): belt lines from which nothing is taken anywhere and on which no product is put (the
    author's bus passing through, a tap to nowhere), pipes of networks no machine takes from or gives
    to, then poles that power nothing (prune_poles).
    """
    from factoriollm.flow_sim import Layout, _fluid_roles, _reach_all

    ents = list(ir.entities)
    L = Layout(ir)
    need = L.lane_need()
    dropped = {dst[1] for src, dst in L.links if dst[0] == "lane"} | {ref for _, (kind, ref) in L.drills if kind == "lane"}
    carries_products = _reach_all(L, dropped)
    useful = {k[0] for k in L.edges if need[k] is None or need[k] or k in carries_products}
    lines = _belt_lines(ents)  # a whole line goes or stays: the tail after its last inserter is still the author's line
    useful_lines = {lines[ents[i]] for i in useful}
    useful = {i for i in L.belts if lines.get(ents[i]) in useful_lines}
    needs_f, makes_f = _fluid_roles(L)
    used_nets = {L.network[b] for ns in needs_f.values() for bs, _ in ns for b in bs} \
        | {L.network[b] for ms in makes_f.values() for b, _ in ms}
    piped = {L.boxes[b][0] for b in range(len(L.boxes)) if L.network[b] in used_nets}
    keep = [e for i, e in enumerate(ents)
            if not (i in L.belts and i not in useful) and not (is_pipe(e.name) and i not in piped)]
    return ir.with_entities(prune_poles(keep))


def prune_poles(ents: List[IREntity]) -> List[IREntity]:
    """
    Drops poles that cover no electric consumer as long as the poles that do stay connected as before:
    long feeder lines and poles left over from the author's grid go, bridges between covered poles stay.
    """
    poles = [i for i, e in enumerate(ents) if entity_type(e.name) == "electric-pole"]
    if not poles:
        return ents
    consumers = [e for e in ents if is_electric_consumer(e.name)]
    covering = {i for i in poles if any(_supplies(ents[i], c) for c in consumers)}
    adj: Dict[int, Set[int]] = {i: set() for i in poles}
    for a, b in itertools.combinations(poles, 2):
        reach = min(float((_prototype(ents[k].name) or {}).get("maximum_wire_distance") or 0) for k in (a, b))
        (ax, ay), (bx, by) = _center(ents[a]), _center(ents[b])
        if math.hypot(ax - bx, ay - by) <= reach:
            adj[a].add(b)
            adj[b].add(a)
    def groups(nodes: Set[int]) -> int:
        """How many separate networks the covering poles fall into, using only `nodes`."""
        seen: Set[int] = set()
        count = 0
        for start in covering & nodes:
            if start in seen:
                continue
            count += 1
            stack = [start]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack += [m for m in adj[n] if m in nodes and m not in seen]
        return count

    kept = set(poles)
    target = groups(kept)
    centre = [_center(c) for c in consumers] or [(0.0, 0.0)]
    # poles far from the machines first, so a feeder line is taken back from its far end
    order = sorted((i for i in poles if i not in covering),
                   key=lambda i: -min(math.hypot(_center(ents[i])[0] - x, _center(ents[i])[1] - y) for x, y in centre))
    for i in order:
        if groups(kept - {i}) <= target:
            kept.discard(i)
    return [e for i, e in enumerate(ents) if i not in poles or i in kept]


def _prune(nodes: Set[int], adj: Dict[int, Set[int]], core: Set[int], margin: int, no_tail: Set[int]) -> Set[int]:
    """
    Removes the leaves of the graph that are not core, again and again: what remains is the core and
    every path between core nodes. Then up to `margin` steps of each cut-off tail come back, but not
    through `no_tail` nodes (splitters of bus taps).
    """
    kept = set(nodes)
    leaves = [n for n in kept if n not in core and len(adj.get(n, set()) & kept) <= 1]
    while leaves:
        n = leaves.pop()
        if n not in kept:
            continue
        kept.discard(n)
        for m in adj.get(n, set()) & kept:
            if m not in core and len(adj.get(m, set()) & kept) <= 1:
                leaves.append(m)
    frontier = set(kept)
    for _ in range(margin):
        grow = {m for n in frontier for m in adj.get(n, set()) if m in nodes and m not in kept and m not in no_tail}
        kept |= grow
        frontier = grow
    return kept


# From the strictest cut to none: (margin, cut splitters on feeds). The miner takes the first cut
# under which a working design still works (an input manifold of splitters looks like a bus tap).
CROP_STEPS = ((CROP_MARGIN, True), (CROP_MARGIN, False), (4, False), (None, False))


def _trim_tails(ents: List[IREntity], margin: int, cut_taps: bool = True) -> List[IREntity]:
    """
    Cuts the author's surroundings by flow: belt lines and pipes keep what lies between the places
    inserters, drills and machines use, plus `margin` tiles of each feed or exit. Turns, undergrounds,
    splitters and side-loads inside the design stay; a splitter on a cut-off feed (a bus tap) goes.
    Undergrounds whose partner was cut go with it, then inserters and chests left with nothing to work on.
    """
    from factoriollm.flow_sim import Layout

    L = Layout(IRBlueprint(ents))
    belt_adj: Dict[int, Set[int]] = {}
    for (i, _), outs in L.edges.items():
        for (j, _), _ in outs:
            belt_adj.setdefault(i, set()).add(j)
            belt_adj.setdefault(j, set()).add(i)
    for i, j in underground_pairs(ents).items():
        belt_adj.setdefault(i, set()).add(j)
        belt_adj.setdefault(j, set()).add(i)
    core = {src[1] for src, _ in L.links if src[0] == "belt"} | {dst[1][0] for _, dst in L.links if dst[0] == "lane"} \
        | {ref[0] for _, (kind, ref) in L.drills if kind == "lane"}
    splitters = {i for i in L.belts if L.types[i] == "splitter"}
    keep_belts = _prune(set(L.belts), belt_adj, core, margin, splitters if cut_taps else set())
    # an underground without a partner at the end of a kept line is where the author's copies join (a
    # weave between neighbouring cells) or the line leaves the design: part of the design, not of its
    # surroundings
    paired = set(underground_pairs(ents)) | set(underground_pairs(ents).values())
    keep_belts |= {i for i in L.belts if L.types[i] == "underground-belt" and i not in paired
                   and belt_adj.get(i, set()) & keep_belts}

    pipes = {i for i, e in enumerate(ents) if is_pipe(e.name)}
    pipe_adj: Dict[int, Set[int]] = {}
    pipe_core: Set[int] = set()
    for a, b in L.pipe_links:
        if a in pipes and b in pipes:
            pipe_adj.setdefault(a, set()).add(b)
            pipe_adj.setdefault(b, set()).add(a)
        elif a in pipes or b in pipes:
            pipe_core.add(a if a in pipes else b)
    keep_pipes = _prune(pipes, pipe_adj, pipe_core, margin, set())

    removed = (set(L.belts) - keep_belts) | (pipes - keep_pipes)
    # an underground (belt or pipe) without its partner
    pairs = dict(underground_pairs(ents))
    pairs.update({j: i for i, j in list(pairs.items())})
    for a, b in L.pipe_links:
        if ents[a].name == ents[b].name == "pipe-to-ground" and abs(ents[a].x - ents[b].x) + abs(ents[a].y - ents[b].y) > 1:
            pairs[a], pairs[b] = b, a
    changed = True
    while changed:
        changed = False
        for i, j in pairs.items():
            if i not in removed and j in removed:
                removed.add(i)
                changed = True
        # inserters that worked on something removed; chests nobody serves any more
        for k, (src, dst) in zip(L.inserters, L.links):
            if k in removed:
                continue
            touched = [ref if kind in ("belt", "entity") else ref[0] if kind == "lane" else None
                       for kind, ref in (src, dst)]
            if any(t is not None and t in removed for t in touched):
                removed.add(k)
                changed = True
        for j, t in enumerate(L.types):
            if t in ("container", "logistic-container") and j not in removed:
                users = [k for k, (src, dst) in zip(L.inserters, L.links)
                         if src == ("entity", j) or dst == ("entity", j)]
                if users and all(k in removed for k in users):
                    removed.add(j)
                    changed = True
    return [e for i, e in enumerate(ents) if i not in removed]


def _belt_graph(ents: List[IREntity], tiles: Dict[Tuple[int, int], int]) -> Tuple[List[int], Dict[int, Set[int]]]:
    """Belt-like entities and their flow edges (front tile, underground entrance -> exit)."""
    belts = [i for i, e in enumerate(ents) if e.name in _TIER]
    edges: Dict[int, Set[int]] = {i: set() for i in belts}
    pairs = underground_pairs(ents)
    for i in belts:
        e = ents[i]
        if i in pairs:
            edges[i].add(pairs[i])
            continue
        if entity_type(e.name) == "underground-belt" and e.extras_dict.get("io_type", "input") == "input":
            continue
        dx, dy = _DIRS[e.direction]
        for x, y in _tiles(e):
            j = tiles.get((x + dx, y + dy))
            if j is not None and j != i and ents[j].name in _TIER and (ents[j].direction - e.direction) % 16 != 8:
                edges[i].add(j)
    return belts, edges


def _target_tile(e: IREntity, key: str) -> Optional[Tuple[int, int]]:
    vec = (_prototype(e.name) or {}).get(key)
    if not vec:
        return None
    cx, cy = _center(e)
    vx, vy = _rotate((float(vec[0]), float(vec[1])), e.direction)
    return math.floor(cx + vx), math.floor(cy + vy)


def add_port_labels(ir: IRBlueprint, smelt: str = "iron-ore", fuel: str = "coal", resource: str = "iron-ore",
                    inside_only: bool = False, flow: Any = None) -> IRBlueprint:
    """
    Labels every port of a design with a constant combinator showing what goes in or comes out,
    from the flow simulator (see flow_sim.solve; `smelt`, `fuel`, `resource` are its preferences),
    not from the author:
    - an input belt gets one label at its start with its item; a label per lane, marked with the
      virtual signal L or R, where the lanes carry different items, or one lane on a line that also
      carries products (a mixed belt: one lane in, one out; a belt passing its inputs on to the next
      block is not mixed);
    - an output belt gets a label in front of its end with what leaves on it (per lane likewise);
      what machines take off the line on its way to the output is left over there (spare gears
      handed on to a neighbouring module): not labelled;
    - each half of a splitter is a belt of its own;
    - a chest filled from outside and an open pipe end get a label next to them (fluid signals).
    Labels go behind a start / in front of an end first, then to the sides, then to the nearest free
    tile (designs walled in by beacons); with inside_only the bounding box is not left.
    """
    from factoriollm.flow_sim import Layout, solve

    LABEL_REACH = 6
    ents = list(ir.entities)
    flow = flow or solve(ir, smelt=smelt, fuel=fuel, resource=resource, share_lanes=False)
    layout = Layout(ir, resource)
    used = {t for e in ents for t in _tiles(e)}
    xs, ys = [t[0] for t in used], [t[1] for t in used]
    labels: List[IREntity] = []
    lines = _belt_lines(ents)

    def free(t: Tuple[int, int]) -> bool:
        return t not in used and (not inside_only or (min(xs) <= t[0] <= max(xs) and min(ys) <= t[1] <= max(ys)))

    def dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def put(ring: List[Tuple[int, int]], signals: List[str], home: Optional[Tuple[int, int]] = None) -> bool:
        """
        The first free tile of the ring, else the nearest free tile around the port (`home`; beacons
        all round), but never one nearer to another port: there it reads as that port's label.
        """
        spot = next((t for t in ring if free(t)), None)
        if spot is None and ring:
            x0, y0 = home or ring[0]
            around = sorted(((x0 + dx, y0 + dy) for dx in range(-LABEL_REACH, LABEL_REACH + 1)
                             for dy in range(-LABEL_REACH, LABEL_REACH + 1)), key=lambda t: dist(t, (x0, y0)))
            spot = next((t for t in around if free(t) and all(
                dist(t, (x0, y0)) <= dist(t, other) for other in port_tiles if other != (x0, y0))), None)
        if spot is None:
            return False
        used.add(spot)
        labels.append(IREntity("constant-combinator", spot[0], spot[1], 0, (("signals", ";".join(
            f"{s}=1" for s in sorted(signals))),)))
        return True

    def tile_of(i: int, half: int) -> Tuple[int, int]:
        tiles = _tiles(ents[i])
        if len(tiles) == 1:
            return tiles[0]
        return next((t for t in tiles if layout.half(i, t) == half), tiles[0])

    def belt_ring(i: int, half: int, start: bool) -> List[Tuple[int, int]]:
        """Behind a belt start / in front of a belt end, then beside it."""
        e = ents[i]
        dx, dy = _DIRS[e.direction]
        lx, ly = dy, -dx  # left of the direction of travel
        x, y = tile_of(i, half)
        s = -1 if start else 1
        behind = (x + s * dx, y + s * dy)
        return [behind, (x + lx, y + ly), (x - lx, y - ly), (behind[0] + lx, behind[1] + ly),
                (behind[0] - lx, behind[1] - ly)]

    def label_belt(i: int, half: int, by_lane: Dict[int, Set[str]], start: bool, mixed: bool) -> bool:
        """
        One label in line with the belt end. Where the lanes carry different items, one label per
        lane, each with the virtual signal L or R (left / right lane in the direction of travel): one
        label with two items reads as both on either lane (the in-game review of 28.10), and a label
        beside a belt reads as the label of the next belt. Returns whether the lanes got labels of
        their own.
        """
        items = set().union(*by_lane.values())
        whole = belt_ring(i, half, start)
        home = tile_of(i, half)
        # both lanes carry the same: L / R tell nothing, even on a mixed line; a single lane on a
        # mixed line still needs its side
        same = len(by_lane) == 2 and len({frozenset(v) for v in by_lane.values()}) == 1
        if same or (len(by_lane) < 2 and not mixed):
            put(whole, [f"item:{x}" for x in items], home)
            return False
        for lane, v in sorted(by_lane.items()):
            put(whole, [f"item:{x}" for x in v] + ["virtual:signal-" + ("L" if lane % 2 == 0 else "R")], home)
        return True

    inputs: Dict[Tuple[int, int], Dict[int, Set[str]]] = {}
    outputs: Dict[Tuple[int, int], Dict[int, Set[str]]] = {}
    for p in flow.ports:
        if p.kind == "lane":
            group = inputs if p.role == "input" else outputs
            group.setdefault((p.entity, p.lane // 2), {}).setdefault(p.lane, set()).update(p.items)
    # spare goods get no label: what machines take off the line on its way to the output is left over
    # there (spare gears a module of the author's book hands on to the next one); goods nobody takes
    # on the way are made for the output (iron plates of a smelting block that also makes steel)
    before = layout.taken_before()
    for key in list(outputs):
        i = key[0]
        lanes_ = {lane: v - (before.get((i, lane)) or set()) for lane, v in outputs[key].items()}
        lanes_ = {lane: v for lane, v in lanes_.items() if v}
        if lanes_:
            outputs[key] = lanes_
        else:
            del outputs[key]
    port_tiles ={tile_of(i, half) for i, half in list(inputs) + list(outputs)} | \
        {p.tile for p in flow.ports if p.kind == "fluid"} | {(ents[p.entity].x, ents[p.entity].y) for p in flow.ports
                                                             if p.kind == "chest"}
    # a mixed line carries products next to its inputs; a line that passes its inputs through to the
    # next block (coal past boilers, science past labs) is not mixed
    line_items: Dict[Any, Dict[str, Set[str]]] = {}
    for group, role in ((inputs, "in"), (outputs, "out")):
        for (i, _), by_lane in group.items():
            line_items.setdefault(lines.get(ents[i]), {"in": set(), "out": set()})[role].update(*by_lane.values())
    mixed = {line for line, v in line_items.items() if v["in"] and v["out"] - v["in"]}
    split: Set[Any] = set()  # lines whose input lanes matter: their pass-through outputs are labelled per lane too
    for (i, half), by_lane in sorted(inputs.items(), key=lambda kv: (ents[kv[0][0]].y, ents[kv[0][0]].x)):
        if label_belt(i, half, by_lane, True, mixed=lines.get(ents[i]) in mixed):
            split.add(lines.get(ents[i]))
    for (i, half), by_lane in sorted(outputs.items(), key=lambda kv: (ents[kv[0][0]].y, ents[kv[0][0]].x)):
        line = lines.get(ents[i])
        label_belt(i, half, by_lane, False, mixed=line in mixed or line in split)
    for p in flow.ports:
        if p.kind == "chest":
            e = ents[p.entity]
            put([(e.x + dx, e.y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))], [f"item:{x}" for x in p.items],
                (e.x, e.y))
        if p.kind != "fluid":
            continue
        ox, oy = p.tile
        dx, dy = _DIRS[p.direction]
        inside = (ox - dx, oy - dy)
        put([(inside[0] + dy, inside[1] - dx), (inside[0] - dy, inside[1] + dx), (ox + dy, oy - dx), (ox - dy, oy + dx)],
            [f"fluid:{x}" for x in p.items], p.tile)
    return ir.with_entities(ents + labels)
