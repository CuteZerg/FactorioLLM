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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from factoriollm.blueprint_ir import IRBlueprint, IREntity, entity_type, footprint, normalized, rotated
from factoriollm.validator import validate

# Entities whose count defines "how much" a design produces; used for density metrics.
MACHINE_TYPES = frozenset({
    "furnace", "assembling-machine", "lab", "mining-drill", "solar-panel", "accumulator", "boiler",
    "generator", "reactor", "rocket-silo", "beacon",
})
# Cosmetic entities: not compared when looking for the repeating part, dropped inside it.
DECOR = frozenset({"small-lamp", "constant-combinator", "display-panel"})
MIN_RUN_COVERAGE = 0.4  # the repeating part must hold at least this share of the entities


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
    snap: Optional[Tuple[int, int]] = None
    source: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_width: int = 0  # width of a whole-cell design (its copies are placed `period` apart)

    def build(self, segments: int) -> IRBlueprint:
        """The design with `segments` periods (head and tail kept)."""
        if segments < 1:
            raise ValueError("a template needs at least one segment")
        ents = list(self.head)
        for k in range(segments):
            ents += [IREntity(e.name, e.x + k * self.period, e.y, e.direction, e.extras) for e in self.segment]
        shift = (segments - self.repeats) * self.period
        ents += [IREntity(e.name, e.x + shift, e.y, e.direction, e.extras) for e in self.tail]
        unique = list(dict.fromkeys(ents))  # overlapping copies may share identical entities
        return normalized(IRBlueprint(unique, snap=self.snap))

    def to_json(self) -> Dict[str, Any]:
        def pack(ents):
            return [[e.name, e.x, e.y, e.direction, dict(e.extras)] for e in ents]
        return {"category": self.category, "period": self.period, "repeats": self.repeats,
                "snap": list(self.snap) if self.snap else None, "source": self.source, "metrics": self.metrics,
                "head": pack(self.head), "segment": pack(self.segment), "tail": pack(self.tail)}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Template":
        def unpack(rows):
            return [IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in rows]
        return Template(data["category"], data["period"], unpack(data["head"]), unpack(data["segment"]),
                        unpack(data["tail"]), data["repeats"], tuple(data["snap"]) if data.get("snap") else None,
                        data.get("source") or {}, data.get("metrics") or {})


def extract(ir: IRBlueprint, category: str, period: Optional[int] = None) -> Optional[Template]:
    """
    Cuts a design that repeats along x into head, segment and tail. Cosmetic entities inside the
    repeating part are dropped; the cut must rebuild the rest of the source exactly.
    """
    ir = normalized(ir)
    period = period or find_period(ir)
    if not period:
        return None
    offset, common, run = _best_run(_structural(ir), period)
    if len(run) < 2:
        return None
    start, end = offset + run[0] * period, offset + (run[-1] + 1) * period
    segment = sorted((IREntity(k[0], u, y, k[1], k[2]) for k, u, y in common), key=lambda e: (e.y, e.x, e.name))
    head = [IREntity(e.name, e.x - start, e.y, e.direction, e.extras) for e in ir.entities if e.x < start]
    tail = [IREntity(e.name, e.x - start, e.y, e.direction, e.extras) for e in ir.entities if e.x >= end]
    template = Template(category, period, head, segment, tail, len(run), ir.snap)
    kept = normalized(ir.with_entities([e for e in ir.entities if e.name not in DECOR or e.x < start or e.x >= end]))
    rebuilt = template.build(len(run)).canonical_set()
    if not kept.canonical_set() <= rebuilt or any(not _is_belt(e) for e in rebuilt - kept.canonical_set()):
        return None  # the cut must rebuild the source; only belt tiles closing a line may be added
    return template


def extract_cell(ir: IRBlueprint, category: str) -> Template:
    """
    A design without an inner repeat that is tiled by placing whole copies side by side: the
    copy pitch is the grid snapping size when the author set one, otherwise the width.
    """
    ir = normalized(ir)
    width = ir.size[0]
    pitch = ir.snap[0] if ir.snap and ir.snap[0] <= width else width
    template = Template(category, pitch, [], sorted(ir.entities, key=lambda e: (e.y, e.x, e.name)), [], 1, ir.snap)
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
    warning categories beyond those the source blueprint already has (a belt end or an unpowered
    entity the author left on purpose stays allowed).
    """
    source = validate(template.build(template.repeats))
    allowed = set(source.category_counts())
    problems = []
    for n in segments:
        report = validate(template.build(n))
        if report.errors:
            problems.append(f"{n} segments: {report.errors[0].message}")
        new = set(report.category_counts()) - allowed - {issue.category for issue in report.errors}
        if new:
            problems.append(f"{n} segments: new warnings {sorted(new)}")
    return {"ok": not problems, "problems": problems[:5], "source_warnings": sorted(allowed)}


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
        "belt_tier": belts.most_common(1)[0][0] if belts else None,
        "poles": dict(poles),
        "head_entities": len(template.head),
        "tail_entities": len(template.tail),
    }


def signature(template: Template) -> Tuple[Any, ...]:
    """Identity of a design for de-duplication: period and segment content."""
    return (template.period, frozenset((_key(e), e.x, e.y) for e in template.segment))
