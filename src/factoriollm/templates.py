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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from factoriollm.blueprint_ir import IRBlueprint, IREntity, _prototype, entity_type, footprint, normalized, rotated
from factoriollm.game_data import SMELTING, recipe
from factoriollm.validator import _center, _rotate, is_electric_consumer, validate

# Entities whose count defines "how much" a design produces; used for density metrics.
MACHINE_TYPES = frozenset({
    "furnace", "assembling-machine", "lab", "mining-drill", "solar-panel", "accumulator", "boiler",
    "generator", "reactor", "rocket-silo", "beacon",
})
# Cosmetic entities: not compared when looking for the repeating part. Lamps become an optional
# layer of the template; author labels and radars are removed by prepare().
DECOR = frozenset({"small-lamp", "constant-combinator", "display-panel"})
LAMP = "small-lamp"
AUTHOR_ONLY = frozenset({"radar", "constant-combinator", "display-panel"})
MIN_RUN_COVERAGE = 0.4  # the repeating part must hold at least this share of the entities
# Belts further than this from the machines are the author's surroundings (bus taps, mains):
# inserter reach (2 tiles for long-handed) plus a belt row, or a lane-loading junction in front.
CROP_MARGIN = 4
BELT_TIERS = (("transport-belt", "underground-belt", "splitter"),
              ("fast-transport-belt", "fast-underground-belt", "fast-splitter"),
              ("express-transport-belt", "express-underground-belt", "express-splitter"))
_TIER = {name: (tier, kind) for tier, names in enumerate(BELT_TIERS) for kind, name in enumerate(names)}
_DIRS = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}
MAX_PORT_LABELS = 12
# Validator categories that are never allowed in a template, whatever the source had.
POWER_CATEGORIES = frozenset({"pole_islands", "unpowered", "no_poles"})


def _key(e: IREntity) -> Tuple[Any, ...]:
    return (e.name, e.direction, e.extras)


def _span(e: IREntity) -> Tuple[int, int]:
    """[start, end) of the entity footprint along x."""
    return e.x, e.x + footprint(e.name, e.direction)[0]


def _structural(ir: IRBlueprint) -> List[IREntity]:
    return [e for e in ir.entities if e.name not in DECOR]


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

    def _place(self, parts: Dict[str, List[IREntity]], segments: int) -> List[IREntity]:
        ents = list(parts["head"])
        for k in range(segments):
            ents += [IREntity(e.name, e.x + k * self.period, e.y, e.direction, e.extras) for e in parts["segment"]]
        shift = (segments - self.repeats) * self.period
        ents += [IREntity(e.name, e.x + shift, e.y, e.direction, e.extras) for e in parts["tail"]]
        return list(dict.fromkeys(ents))  # overlapping copies may share identical entities

    def build(self, segments: int, lamps: bool = False, belt: Optional[str] = None) -> IRBlueprint:
        """
        The design with `segments` periods (head and tail kept), at the template's belt tier or at
        `belt`, optionally with the author's lamps (only on tiles left free).
        """
        if segments < 1:
            raise ValueError("a template needs at least one segment")
        ents = self._place({"head": self.head, "segment": self.segment, "tail": self.tail}, segments)
        if belt or self.belt:
            ents = retier(ents, belt or self.belt)
        if lamps:
            taken = {t for e in ents for t in _tiles(e)}
            ents += [e for e in self._place(self.lamps, segments) if (e.x, e.y) not in taken]
        return normalized(IRBlueprint(ents))

    def build_segment(self, belt: Optional[str] = None) -> IRBlueprint:
        """One segment on its own grid (relative snapping, cell = period x segment height), to tile in game."""
        ents = retier(list(self.segment), belt or self.belt) if belt or self.belt else list(self.segment)
        y0 = min(e.y for e in ents)
        return IRBlueprint([IREntity(e.name, e.x, e.y - y0, e.direction, e.extras) for e in ents], snap=self.snap)

    def to_json(self) -> Dict[str, Any]:
        def pack(ents):
            return [[e.name, e.x, e.y, e.direction, dict(e.extras)] for e in ents]
        return {"category": self.category, "period": self.period, "repeats": self.repeats,
                "snap": list(self.snap) if self.snap else None, "source": self.source, "metrics": self.metrics,
                "belt": self.belt, "head": pack(self.head), "segment": pack(self.segment), "tail": pack(self.tail),
                "lamps": {part: pack(ents) for part, ents in self.lamps.items()}}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Template":
        def unpack(rows):
            return [IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in rows]
        lamps = {part: unpack(rows) for part, rows in (data.get("lamps") or {}).items()}
        return Template(data["category"], data["period"], unpack(data["head"]), unpack(data["segment"]),
                        unpack(data["tail"]), data["repeats"], tuple(data["snap"]) if data.get("snap") else None,
                        data.get("source") or {}, data.get("metrics") or {}, belt=data.get("belt"),
                        lamps={"head": [], "segment": [], "tail": [], **lamps})


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
    kept = normalized(ir.with_entities([e for e in solid if e.name not in DECOR or e.x < start or e.x >= end]))
    rebuilt = template.build(len(run)).canonical_set()
    if not kept.canonical_set() <= rebuilt or any(not _is_belt(e) for e in rebuilt - kept.canonical_set()):
        return None  # the cut must rebuild the source; only belt tiles closing a line may be added
    return template


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


def tiling_check(template: Template, segments: Tuple[int, ...] = (1, 2, 5)) -> Dict[str, Any]:
    """
    Builds the design with several segment counts and validates each build: no errors, and no
    warning categories beyond those the source blueprint already has (a belt end the author left
    on purpose stays allowed). Power is always strict: a module must not depend on poles outside it
    (POWER_CATEGORIES are never allowed, even if the author had them). `power_only` tells whether
    power was the only problem, i.e. whether repair_power may help.
    """
    source = validate(template.build(template.repeats))
    allowed = set(source.category_counts()) - POWER_CATEGORIES
    problems, categories = [], set()
    for n in segments:
        report = validate(template.build(n))
        if report.errors:
            problems.append(f"{n} segments: {report.errors[0].message}")
            categories |= {issue.category for issue in report.errors}
        new = set(report.category_counts()) - allowed - {issue.category for issue in report.errors}
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


def _uncovered(ents: List[IREntity]) -> List[int]:
    """Indices of electric consumers outside every pole's supply area (as the validator counts them)."""
    areas = []
    for e in ents:
        if entity_type(e.name) == "electric-pole":
            d = float((_prototype(e.name) or {}).get("supply_area_distance") or 0)
            cx, cy = _center(e)
            areas.append((cx - d, cy - d, cx + d, cy + d))
    out = []
    for i, e in enumerate(ents):
        if is_electric_consumer(e.name):
            w, h = footprint(e.name, e.direction)
            if not any(e.x < x2 and e.x + w > x1 and e.y < y2 and e.y + h > y1 for x1, y1, x2, y2 in areas):
                out.append(i)
    return out


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
    underground of the same name and direction ahead within reach; if that one is another
    entrance, it does not connect (same-tier pairs cannot be woven). `names` overrides entity names
    (to test another belt tier without building new entities).
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
            if j is not None and names[j] == names[i] and ents[j].direction == e.direction:
                if ents[j].extras_dict.get("io_type") == "output":
                    pairs[i] = j
                break
    return pairs


def has_weaving(ents: List[IREntity]) -> bool:
    """Belt weaving: an underground pair passes under another underground on the same line."""
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
        lines.setdefault((e.direction, e.x if vertical else e.y), []).append((lo, hi, unit))
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


def prepare(ir: IRBlueprint) -> IRBlueprint:
    """
    A self-contained copy of a player's design: no radars or author labels (new labels come from
    the recipes, see add_port_labels), and no belts more than CROP_MARGIN tiles away from the
    machines (bus taps and mains differ per base). An underground whose partner is cut goes too.
    Lamps stay; templates keep them as an optional layer.
    """
    ents = [e for e in ir.entities if e.name not in AUTHOR_ONLY]
    machines = [e for e in ents if entity_type(e.name) in MACHINE_TYPES]
    if machines:
        x0 = min(e.x for e in machines) - CROP_MARGIN
        y0 = min(e.y for e in machines) - CROP_MARGIN
        x1 = max(e.x + footprint(e.name, e.direction)[0] for e in machines) + CROP_MARGIN
        y1 = max(e.y + footprint(e.name, e.direction)[1] for e in machines) + CROP_MARGIN
        pairs = underground_pairs(ents)
        near = [e.name not in _TIER or any(x0 <= x < x1 and y0 <= y < y1 for x, y in _tiles(e)) for e in ents]
        for i, j in pairs.items():
            if near[i] != near[j]:
                near[i] = near[j] = False
        ents = [e for e, keep in zip(ents, near) if keep]
    return ir.with_entities(ents)


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
                    inside_only: bool = False) -> IRBlueprint:
    """
    Labels every input and output belt of a design with a constant combinator showing what it
    carries, from recipe data rather than from the author:
    - an input line (inserters only take from it) is labelled at its start with the ingredients of
      the machines it feeds, minus what the design makes itself (cable inside a circuit block);
    - an output line (machines put items on it) is labelled at its end with what is put on it
      and not taken off again inside.
    Furnaces have no recipe: they get `smelt` (ore, stone or iron plates for steel) and, if they
    burn fuel, `fuel`; drills put `resource` on their belt. Labels go next to the line end, behind
    a start or in front of an end first; with inside_only the bounding box is not left.
    """
    ents = list(ir.entities)
    tiles = {t: i for i, e in enumerate(ents) for t in _tiles(e)}
    made = {p for e in ents if recipe(e.extras_dict.get("recipe") or "") for p in recipe(e.extras_dict["recipe"]).results}
    if any(entity_type(e.name) == "furnace" for e in ents) and smelt in SMELTING:
        made.add(SMELTING[smelt])

    def burner(e: IREntity) -> bool:
        return ((_prototype(e.name) or {}).get("energy_source") or {}).get("type") == "burner"

    def takes(e: IREntity) -> Set[str]:  # what a machine fed by an inserter takes
        r = recipe(e.extras_dict.get("recipe") or "")
        if r:
            solid = set(r.ingredients) - r.fluids
            return solid - made or solid
        etype = entity_type(e.name)
        if etype == "furnace":
            return {smelt} | ({fuel} if burner(e) else set())
        return {fuel} if burner(e) and etype in ("boiler", "mining-drill") else set()

    def gives(e: IREntity) -> Set[str]:  # what a machine puts out
        r = recipe(e.extras_dict.get("recipe") or "")
        if r:
            return set(r.results) - r.fluids
        etype = entity_type(e.name)
        if etype == "furnace":
            return {SMELTING[smelt]} if smelt in SMELTING else set()
        return {resource} if etype == "mining-drill" else set()

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
    taken: Dict[int, Set[str]] = {}
    put: Dict[int, Set[str]] = {}
    for e in ents:
        etype = entity_type(e.name)
        if etype == "inserter":
            src, dst = (tiles.get(t) for t in (_target_tile(e, "pickup_position"), _target_tile(e, "insert_position")))
            if src in parent and dst is not None and dst not in parent:
                taken.setdefault(root(src), set()).update(takes(ents[dst]))
            if dst in parent and src is not None and src not in parent:
                put.setdefault(root(dst), set()).update(gives(ents[src]))
        elif etype == "mining-drill":
            dst = tiles.get(_target_tile(e, "vector_to_place_result"))
            if dst in parent:
                put.setdefault(root(dst), set()).update(gives(e))
    incoming = {j for out in edges.values() for j in out}
    xs, ys = [t[0] for t in tiles], [t[1] for t in tiles]
    labels: List[IREntity] = []
    used = set(tiles)

    def place(i: int, items: Set[str], start: bool) -> None:
        e = ents[i]
        dx, dy = _DIRS[e.direction]
        x, y = e.x, e.y
        s = -1 if start else 1  # behind a line start, in front of a line end first
        ring = [(x + s * dx, y + s * dy), (x - dy, y + dx), (x + dy, y - dx),
                (x + s * dx - dy, y + s * dy + dx), (x + s * dx + dy, y + s * dy - dx)]
        for t in ring:
            if t in used or (inside_only and not (min(xs) <= t[0] <= max(xs) and min(ys) <= t[1] <= max(ys))):
                continue
            used.add(t)
            labels.append(IREntity("constant-combinator", t[0], t[1], 0,
                                   (("signals", ";".join(f"item:{item}=1" for item in sorted(items))),)))
            return

    for i in sorted(belts, key=lambda k: (ents[k].y, ents[k].x)):
        if len(labels) >= MAX_PORT_LABELS:
            break
        c = root(i)
        out_items = put.get(c, set()) - taken.get(c, set())
        if c in put and not edges[i] and out_items:
            place(i, out_items, start=False)
        elif c not in put and i not in incoming and taken.get(c):
            place(i, taken[c], start=True)
    return ir.with_entities(ents + labels)
