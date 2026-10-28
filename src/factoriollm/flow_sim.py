"""
Flow simulator: which items can end up on every belt lane, in every machine and in every pipe.

It answers "does this layout work on its own?" without running the game. The state is sets of
items, grown until nothing changes; speeds are not simulated (except to rank alternatives).

Belts have two lanes, 0 = left and 1 = right in the direction of travel. Items move:
- straight on and around curves keeping their lane; side-loading puts everything on the target
  lane on the feeding side (into an underground belt only through its open half, so only one
  lane of the feeding belt passes);
- through underground pairs and splitters keeping their lane (a splitter filter sends its item to
  the priority side and everything else to the other side);
- by inserters from a pickup tile (belt lanes, a machine's products, a chest, items on the ground)
  to a drop tile (the far lane of a crossing belt, the right lane of a belt along the inserter,
  the inside lane of a curve; a machine; a chest; the ground). An inserter only picks what the
  target accepts. Drills drop their resource the same way.
Fluids: pipes, pipe-to-ground pairs, tanks, pumps and machine fluid boxes form networks; a network
holds one fluid.

What enters from outside is an assignment of items to the input ports: belt lanes nobody feeds,
chests nobody fills, pipe networks with an open end and no producer. `solve()` searches the
assignment that makes the layout work (`check()` lists the problems of one assignment):
- starved: a machine lacks an ingredient (or fuel);
- no_output: nobody takes a machine's product;
- clog: an item reaches a belt stretch where nobody downstream takes it and that does not lead
  to an output, while other items still have to pass;
- output_mixed: an input item leaves through an output belt together with the products;
- dangling_inserter: an inserter with nothing to pick, or dropping on the ground nobody picks from;
- fluid_mix: two fluids in one pipe network.
"""

from __future__ import annotations

import itertools
from collections import Counter
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from factoriollm.blueprint_ir import IRBlueprint, IREntity, _prototype, entity_type, footprint
from factoriollm.game_data import SMELTING, recipe
from factoriollm.validator import _DIRECTION_VECTORS, _center, _occupied_tiles as _tiles, _rotate

Tile = Tuple[int, int]
Lane = Tuple[int, int]  # (entity index, 0 = left / 1 = right)

BELT_LIKE = frozenset({"transport-belt", "underground-belt", "splitter"})
CONTAINERS = frozenset({"container", "logistic-container"})
FUELS = ("coal", "wood", "solid-fuel", "rocket-fuel", "nuclear-fuel")
SCIENCE_PACKS = ("automation-science-pack", "logistic-science-pack", "military-science-pack", "chemical-science-pack",
                 "production-science-pack", "utility-science-pack", "space-science-pack")
AMMO = {"ammo-turret": ("firearm-magazine", "piercing-rounds-magazine", "uranium-rounds-magazine"),
        "artillery-turret": ("artillery-shell",)}
FLAMETHROWER_FLUIDS = ("light-oil", "heavy-oil", "crude-oil")
ROBOPORT_ITEMS = ("logistic-robot", "construction-robot", "repair-pack")
# Items per second of one inserter moving single items between a belt and a machine (rough, for ranking only).
INSERTER_RATE = {"burner-inserter": 0.6, "inserter": 0.83, "long-handed-inserter": 1.2, "fast-inserter": 2.31,
                 "bulk-inserter": 4.6}
ANY = None  # "accepts any item" (chests, belts, the ground)
# Entities an inserter takes items back out of, to hand them on along a chain of inserters: labs
# pass science packs on, boilers fuel, turrets ammunition. (Reactors give out spent fuel cells.)
PASS_ON = {"lab": frozenset(SCIENCE_PACKS), "boiler": frozenset(FUELS),
           "ammo-turret": frozenset(AMMO["ammo-turret"]), "artillery-turret": frozenset(AMMO["artillery-turret"])}
GIVES_OUT = frozenset(PASS_ON) | {"reactor"}  # inserters take from these whether or not they "work"
MAX_LEAVES = 400      # full simulations per solve before giving up as ambiguous
MAX_SOLUTIONS = 12    # working assignments compared by score
MAX_EXTRA_LEAVES = 40  # simulations spent on alternatives after the first working assignment
MAX_STEPS = 20        # search nodes per leaf budget (each node scans every constraint)


@dataclass(frozen=True)
class Problem:
    kind: str
    message: str
    position: Tile


@dataclass(frozen=True)
class Port:
    """Where flow crosses the layout's border: a belt lane, a chest or a pipe end."""
    kind: str          # "lane", "chest" or "fluid"
    role: str          # "input" or "output"
    entity: int        # entity index (belt, chest; for fluids the entity with the open connection)
    lane: Optional[int]
    items: FrozenSet[str]
    tile: Tile         # for fluids: the free tile the open connection points to
    direction: int = 0  # belt direction / direction of the open pipe connection


@dataclass
class FlowResult:
    problems: List[Problem]
    lanes: Dict[Lane, FrozenSet[str]]
    machines: Dict[int, FrozenSet[str]]      # what each machine receives
    working: FrozenSet[int]
    ports: List[Port]
    inputs: Dict[Any, Optional[str]] = field(default_factory=dict)
    status: str = "ok"  # ok / inputs_unsolved / inputs_ambiguous
    score: Tuple[float, ...] = ()
    fluids: Dict[int, FrozenSet[str]] = field(default_factory=dict)  # entity -> fluids in its pipe networks

    @property
    def ok(self) -> bool:
        return self.status == "ok" and not self.problems


def _vec(d: int) -> Tuple[int, int]:
    return _DIRECTION_VECTORS[d % 16]


def _left(d: int) -> Tuple[int, int]:
    dx, dy = _vec(d)
    return dy, -dx


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


@lru_cache(maxsize=None)
def _type(name: str) -> str:
    return entity_type(name)


def _burner(name: str) -> bool:
    return ((_prototype(name) or {}).get("energy_source") or {}).get("type") == "burner"


def _point(e: IREntity, key: str) -> Optional[Tuple[float, float]]:
    vec = (_prototype(e.name) or {}).get(key)
    if not vec:
        return None
    cx, cy = _center(e)
    vx, vy = _rotate((float(vec[0]), float(vec[1])), e.direction)
    return cx + vx, cy + vy


def _tile_of(p: Tuple[float, float]) -> Tile:
    return math.floor(p[0]), math.floor(p[1])


class Layout:
    """Static structure of a blueprint: belt lane graph, inserters, machines, fluid networks."""

    def __init__(self, ir: IRBlueprint, resource: str = "iron-ore"):
        self.ents = list(ir.entities)
        self.resource = resource
        self.tiles: Dict[Tile, int] = {t: i for i, e in enumerate(self.ents) for t in _tiles(e)}
        self.types = [_type(e.name) for e in self.ents]
        self.belts = [i for i, t in enumerate(self.types) if t in BELT_LIKE]
        self._pairs = self._underground_pairs()
        self.into: Dict[Tile, List[int]] = {}
        for i in self.belts:
            for t in self._out_tiles(i):
                self.into.setdefault(t, []).append(i)
        self._curves: Dict[int, Optional[int]] = {}
        # a splitter has two lanes per half: 0/1 on its left half, 2/3 on its right half
        self.lanes_of: Dict[int, Tuple[int, ...]] = {i: (0, 1, 2, 3) if self.types[i] == "splitter" else (0, 1)
                                                     for i in self.belts}
        self.edges: Dict[Lane, List[Tuple[Lane, Optional[Tuple[str, str]]]]] = {
            (i, l): [] for i in self.belts for l in self.lanes_of[i]}
        self._build_lanes()
        self.incoming: Dict[Lane, int] = {}
        self.reverse: Dict[Lane, List[Tuple[Lane, Optional[Tuple[str, str]]]]] = {k: [] for k in self.edges}
        for src, outs in self.edges.items():
            for dst, rule in outs:
                self.incoming[dst] = self.incoming.get(dst, 0) + 1
                self.reverse[dst].append((src, rule))
        self._need: Optional[Dict[Lane, Optional[Set[str]]]] = None
        self.inserters = [i for i, t in enumerate(self.types) if t == "inserter"]
        self.links = [self._inserter(i) for i in self.inserters]
        # item filters per inserter: (whitelist?, items) or None
        self.filters: List[Optional[Tuple[bool, FrozenSet[str]]]] = []
        for i in self.inserters:
            ex = self.ents[i].extras_dict
            items = frozenset(x for x in (ex.get("filters") or "").split(";") if x)
            self.filters.append((ex.get("filter_mode") != "blacklist", items) if items else None)
        self.drills = [(i, self._drop(self.ents[i], _point(self.ents[i], "vector_to_place_result")))
                       for i, t in enumerate(self.types) if t == "mining-drill" and self.ents[i].name != "pumpjack"]
        self._build_fluids()

    # --- belts ---------------------------------------------------------------------------

    def _underground_pairs(self) -> Dict[int, int]:
        from factoriollm.templates import underground_pairs
        return underground_pairs(self.ents)

    def _io(self, i: int) -> str:
        return self.ents[i].extras_dict.get("io_type", "input")

    def _out_tiles(self, i: int) -> List[Tile]:
        e, t = self.ents[i], self.types[i]
        dx, dy = _vec(e.direction)
        if t == "underground-belt" and self._io(i) == "input":
            return []
        return [(x + dx, y + dy) for x, y in _tiles(e)]

    def _feeders(self, j: int) -> List[int]:
        dj = self.ents[j].direction
        found = {i for t in _tiles(self.ents[j]) for i in self.into.get(t, [])}
        return [i for i in found if i != j and (self.ents[i].direction - dj) % 16 != 8]

    def is_curve(self, j: int) -> Optional[int]:
        """For a transport belt fed from exactly one side and not from behind: the feeding direction."""
        if j in self._curves:
            return self._curves[j]
        out = None
        if self.types[j] == "transport-belt":
            dj = self.ents[j].direction
            feeders = self._feeders(j)
            side = [i for i in feeders if (self.ents[i].direction - dj) % 8 == 4]
            if not any(self.ents[i].direction == dj for i in feeders) and len(side) == 1:
                out = self.ents[side[0]].direction
        self._curves[j] = out
        return out

    def lane_need(self) -> Dict[Lane, Optional[Set[str]]]:
        """
        Per lane: the items some inserter takes from it or from a lane it flows into (None: anything,
        a chest or the ground takes all). Independent of the inputs, so computed once per layout.
        """
        if self._need is not None:
            return self._need
        picks = _picks(self)
        need: Dict[Lane, Optional[Set[str]]] = {}
        for k in self.edges:
            p = picks.get(k[0], frozenset())
            need[k] = None if p is ANY else set(p)
        stack = list(self.edges)
        while stack:  # push what a lane needs back to the lanes feeding it
            dst = stack.pop()
            for src, rule in self.reverse[dst]:
                if need[src] is None:
                    continue
                if need[dst] is None:
                    need[src] = None
                    stack.append(src)
                    continue
                add = {x for x in need[dst] if _passes(rule, x)} - need[src]
                if add:
                    need[src] |= add
                    stack.append(src)
        self._need = need
        return need

    def upstream(self, lanes: Iterable[Lane]) -> Set[Lane]:
        """These lanes and every lane that flows into them."""
        seen = set(lanes)
        stack = list(seen)
        while stack:
            for src, _ in self.reverse[stack.pop()]:
                if src not in seen:
                    seen.add(src)
                    stack.append(src)
        return seen

    def half(self, j: int, tile: Tile) -> int:
        """0 for the left half of splitter j (in its direction), 1 for the right half."""
        cx, cy = _center(self.ents[j])
        return 0 if _dot((tile[0] + 0.5 - cx, tile[1] + 0.5 - cy), _left(self.ents[j].direction)) > 0 else 1

    def belt_lanes(self, j: int) -> Tuple[Lane, ...]:
        return tuple((j, l) for l in self.lanes_of[j])

    def _feed(self, i: int, tile: Tile, d: int) -> Dict[int, int]:
        """Lane mapping {lane of i: lane of the entity at tile} for items leaving i at tile in direction d."""
        j = self.tiles.get(tile)
        if j is None or j == i or self.types[j] not in BELT_LIKE:
            return {}
        e = self.ents[j]
        dj, tj = e.direction, self.types[j]
        if (dj - d) % 16 == 8:
            return {}
        if dj == d:
            if tj == "underground-belt" and self._io(j) == "output":
                return {}  # the tunnel side of an exit
            if tj == "splitter":
                h = 2 * self.half(j, tile)
                return {0: h, 1: h + 1}
            return {0: 0, 1: 1}
        if tj == "splitter":
            return {}
        side = 0 if _dot((-_vec(d)[0], -_vec(d)[1]), _left(dj)) > 0 else 1  # i stands on j's left or right
        if tj == "transport-belt":
            if self.is_curve(j) is not None:
                return {0: 0, 1: 1}
            return {0: side, 1: side}
        # underground belt: only its belt half (behind an entrance's hood, in front of an exit's) takes side-loading
        open_dir = _vec(dj + 8) if self._io(j) == "input" else _vec(dj)
        passing = 0 if _dot(_left(d), open_dir) > 0 else 1
        return {passing: side}

    def _build_lanes(self) -> None:
        for i in self.belts:
            e, t = self.ents[i], self.types[i]
            if t == "underground-belt" and i in self._pairs:
                for l in (0, 1):
                    self.edges[(i, l)].append(((self._pairs[i], l), None))
                continue
            if t == "splitter":
                flt = e.extras_dict.get("filter")
                prio = e.extras_dict.get("output_priority")
                left = _left(e.direction)
                cx, cy = _center(e)
                for tile in self._out_tiles(i):
                    src = (tile[0] - _vec(e.direction)[0], tile[1] - _vec(e.direction)[1])
                    side = "left" if _dot((src[0] + 0.5 - cx, src[1] + 0.5 - cy), left) > 0 else "right"
                    rule = None
                    if flt and prio in ("left", "right"):
                        rule = ("only", flt) if side == prio else ("except", flt)
                    mapping = self._feed(i, tile, e.direction)
                    for l in self.lanes_of[i]:  # either half's lane goes to both output halves
                        if l % 2 in mapping:
                            self.edges[(i, l)].append(((self.tiles[tile], mapping[l % 2]), rule))
                continue
            for tile in self._out_tiles(i):
                for l, m in self._feed(i, tile, e.direction).items():
                    self.edges[(i, l)].append(((self.tiles[tile], m), None))

    def drop_lane(self, j: int, p: Tuple[float, float], u: Optional[Tuple[float, float]]) -> int:
        """Lane of belt j an item dropped at point p lands on (u: direction the dropping arm points)."""
        dj = self.ents[j].direction
        incoming = self.is_curve(j)
        if incoming is not None:  # inside lane of the curve
            a, b = _vec(incoming), _vec(dj)
            return 1 if a[0] * b[1] - a[1] * b[0] > 0 else 0
        if u is not None and abs(_dot(u, _vec(dj))) > 0.5:
            return 1 + (2 * self.half(j, _tile_of(p)) if self.types[j] == "splitter" else 0)  # along: right lane
        tile = _tile_of(p)
        off = (p[0] - (tile[0] + 0.5), p[1] - (tile[1] + 0.5))
        lane = 0 if _dot(off, _left(dj)) > 0 else 1
        return lane + 2 * self.half(j, tile) if self.types[j] == "splitter" else lane

    # --- inserters ------------------------------------------------------------------------

    def _drop(self, e: IREntity, p: Optional[Tuple[float, float]], u=None) -> Tuple[str, Any]:
        if p is None:
            return ("none", None)
        tile = _tile_of(p)
        j = self.tiles.get(tile)
        if j is None:
            return ("ground", tile)
        if self.types[j] in BELT_LIKE:
            return ("lane", (j, self.drop_lane(j, p, u)))
        return ("entity", j)

    def _inserter(self, i: int) -> Tuple[Tuple[str, Any], Tuple[str, Any]]:
        e = self.ents[i]
        pick, drop = _point(e, "pickup_position"), _point(e, "insert_position")
        u = None
        if pick and drop:
            n = math.hypot(drop[0] - pick[0], drop[1] - pick[1]) or 1.0
            u = ((drop[0] - pick[0]) / n, (drop[1] - pick[1]) / n)
        src_tile = _tile_of(pick) if pick else None
        j = self.tiles.get(src_tile) if src_tile else None
        if src_tile is None:
            src = ("none", None)
        elif j is None:
            src = ("ground", src_tile)
        elif self.types[j] in BELT_LIKE:
            src = ("belt", j)
        else:
            src = ("entity", j)
        return src, self._drop(e, drop, u)

    # --- machines -------------------------------------------------------------------------

    def moves(self, idx: int, items: Iterable[str]) -> Set[str]:
        """What inserter number idx (in self.links) moves out of these items: its filter, then what its target accepts."""
        out = set(items)
        flt = self.filters[idx]
        if flt is not None:
            out = out & flt[1] if flt[0] else out - flt[1]
        dst = self.links[idx][1]
        if dst[0] == "entity":
            acc = self.accepts(dst[1])
            if acc is not ANY:
                out &= acc
        return out

    def accepts(self, j: int) -> Optional[FrozenSet[str]]:
        """Items an inserter may put into entity j (ANY for chests)."""
        e, t = self.ents[j], self.types[j]
        if t in CONTAINERS or t in ("car", "cargo-wagon"):
            return ANY
        r = recipe(e.extras_dict.get("recipe") or "")
        if t == "assembling-machine":
            return frozenset(set(r.ingredients) - r.fluids) if r else frozenset()
        if t == "rocket-silo":
            part = recipe("rocket-part")
            return frozenset(set(part.ingredients) | {"satellite"}) if part else frozenset()
        fuel = frozenset(FUELS) if _burner(e.name) else frozenset()
        if t == "furnace":
            return frozenset(SMELTING) | fuel
        if t == "lab":
            return frozenset(SCIENCE_PACKS)
        if t == "reactor":
            return frozenset({"uranium-fuel-cell"})
        if t in AMMO:
            return frozenset(AMMO[t])
        if t == "roboport":
            return frozenset(ROBOPORT_ITEMS)
        if t in ("boiler", "mining-drill"):
            return fuel
        return frozenset()

    def needs(self, j: int) -> List[FrozenSet[str]]:
        """Item requirements of a machine: each set must be met by at least one of its items."""
        e, t = self.ents[j], self.types[j]
        r = recipe(e.extras_dict.get("recipe") or "")
        fuel = [frozenset(FUELS)] if _burner(e.name) and t != "reactor" else []
        if t == "assembling-machine":
            return [frozenset({i}) for i in sorted(set(r.ingredients) - r.fluids)] if r else [frozenset({"<recipe>"})]
        if t == "rocket-silo":
            part = recipe("rocket-part")
            return [frozenset({i}) for i in sorted(part.ingredients)] if part else []
        if t == "furnace":
            return [frozenset(SMELTING)] + fuel
        if t == "lab":
            return [frozenset(SCIENCE_PACKS)]
        if t in ("boiler", "mining-drill"):
            return fuel
        return []

    def products(self, j: int, received: FrozenSet[str]) -> FrozenSet[str]:
        e, t = self.ents[j], self.types[j]
        r = recipe(e.extras_dict.get("recipe") or "")
        if t == "assembling-machine" and r:
            return frozenset(set(r.results) - r.fluids)
        if t == "furnace":
            return frozenset(SMELTING[i] for i in received if i in SMELTING)
        if t in PASS_ON:
            return frozenset(received & PASS_ON[t])
        if t == "reactor":
            return frozenset({"depleted-uranium-fuel-cell"}) if "uranium-fuel-cell" in received else frozenset()
        if t in CONTAINERS:
            return received
        return frozenset()

    def is_machine(self, j: int) -> bool:
        t = self.types[j]
        if t == "mining-drill":
            return _burner(self.ents[j].name)
        return t in ("assembling-machine", "furnace", "lab", "rocket-silo", "boiler") and not (
            t == "boiler" and not _burner(self.ents[j].name))

    # --- fluids ---------------------------------------------------------------------------

    def _fluid_boxes(self, j: int) -> List[Tuple[str, Optional[str], List[Tuple[Tile, int, bool, int]]]]:
        """[(production type, filter, [(inside tile, outward direction, underground, max distance)])]."""
        e, proto = self.ents[j], _prototype(self.ents[j].name) or {}
        t = self.types[j]
        if t == "assembling-machine":
            r = recipe(e.extras_dict.get("recipe") or "")
            if not r or not r.fluids:
                return []
        boxes = list(proto.get("fluid_boxes") or [])
        for key in ("fluid_box", "input_fluid_box", "output_fluid_box"):
            if isinstance(proto.get(key), dict):
                box = dict(proto[key])
                if key == "output_fluid_box" and not box.get("production_type"):
                    box["production_type"] = "output"
                if key == "input_fluid_box" and t == "mining-drill":
                    continue  # mining fluid (uranium) is out of scope
                boxes.append(box)
        if t == "pump" and boxes:
            # a pump passes fluid one way: its input and output sides are boxes of two networks
            conns = boxes[0].get("pipe_connections") or []
            boxes = [{"production_type": f"pump-{side}", "pipe_connections": [c for c in conns
                                                                               if c.get("flow_direction") == side]}
                     for side in ("input", "output")]
        out = []
        cx, cy = _center(e)
        mirror = e.extras_dict.get("mirror") == "true"  # 2.0: flipped left-right before rotating
        for box in boxes:
            conns = []
            for c in box.get("pipe_connections") or []:
                if c.get("connection_type") == "linked":
                    continue
                pos = c.get("position")
                d0 = int(c.get("direction", 0))
                if pos is None and c.get("positions"):
                    pos = c["positions"][(e.direction // 4) % 4]
                    vx, vy = float(pos[0]), float(pos[1])
                else:
                    if pos is None:
                        continue
                    px = -float(pos[0]) if mirror else float(pos[0])
                    if mirror and d0 % 8 == 4:
                        d0 = (d0 + 8) % 16  # east and west swap
                    vx, vy = _rotate((px, float(pos[1])), e.direction)
                d = (d0 + e.direction) % 16
                inside = _tile_of((cx + vx, cy + vy))
                conns.append((inside, d, c.get("connection_type") == "underground",
                              int(c.get("max_underground_distance") or 0)))
            out.append((box.get("production_type") or "none", box.get("filter"), conns))
        return out

    def _build_fluids(self) -> None:
        self.boxes: List[Tuple[int, int, str, Optional[str]]] = []  # (entity, box index, type, filter)
        conns: List[Tuple[int, Tile, int, bool, int]] = []          # (box id, inside, dir, underground, max)
        for j, t in enumerate(self.types):
            if t in BELT_LIKE or t == "inserter":
                continue
            for k, (ptype, flt, cs) in enumerate(self._fluid_boxes(j)):
                b = len(self.boxes)
                self.boxes.append((j, k, ptype, flt))
                conns += [(b, inside, d, ug, dist) for inside, d, ug, dist in cs]
        parent = list(range(len(self.boxes)))

        def root(b: int) -> int:
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        at: Dict[Tuple[Tile, int, bool], List[int]] = {}
        for b, inside, d, ug, dist in conns:
            at.setdefault((inside, d, ug), []).append(b)
        self.open_ends: List[Tuple[int, Tile, int]] = []  # (box, free tile, direction)
        self.pipe_links: List[Tuple[int, int]] = []        # entities joined by a fluid connection
        # an underground pipe has one partner: the first one facing it within reach, and only if it
        # is that one's first too (a pipe-to-ground between them facing the same way does not count;
        # a pair two tiles apart leaves a farther one facing it unpaired, as authors use to keep two
        # lines in one row apart)
        ug_at = {(inside, d): n for n, (_, inside, d, ug, _) in enumerate(conns) if ug}
        first: Dict[int, Optional[int]] = {}
        for n, (b, inside, d, ug, dist) in enumerate(conns):
            if ug:
                dx, dy = _vec(d)
                first[n] = next((ug_at[t] for t in (((inside[0] + k * dx, inside[1] + k * dy), (d + 8) % 16)
                                                    for k in range(1, dist + 1)) if t in ug_at), None)
        for n, (b, inside, d, ug, dist) in enumerate(conns):
            dx, dy = _vec(d)
            if ug:
                m = first[n]
                if m is not None and first.get(m) == n:
                    if n < m:
                        parent[root(b)] = root(conns[m][0])
                        self.pipe_links.append((self.boxes[b][0], self.boxes[conns[m][0]][0]))
                elif m is None:  # an underground pipe without a partner brings fluid from outside
                    self.open_ends.append((b, (inside[0] + dx, inside[1] + dy), d))
                continue
            nbr = (inside[0] + dx, inside[1] + dy)
            hit = at.get((nbr, (d + 8) % 16, False))
            if hit:
                for h in hit:
                    parent[root(b)] = root(h)
                    self.pipe_links.append((self.boxes[b][0], self.boxes[h][0]))
            elif nbr not in self.tiles:
                self.open_ends.append((b, nbr, d))
        self.network = [root(b) for b in range(len(self.boxes))]
        # pumps: (network on the input side, network on the output side)
        sides: Dict[int, Dict[str, int]] = {}
        for b, (j, _, ptype, _) in enumerate(self.boxes):
            if ptype.startswith("pump-"):
                sides.setdefault(j, {})[ptype] = self.network[b]
        self.pumped = [(s["pump-input"], s["pump-output"]) for s in sides.values() if len(s) == 2]

    def downstream_nets(self, net: int) -> Set[int]:
        """This network and every network pumps carry its fluid on to."""
        seen, stack = {net}, [net]
        while stack:
            n = stack.pop()
            for a, b in self.pumped:
                if a == n and b not in seen:
                    seen.add(b)
                    stack.append(b)
        return seen


# --- Simulation -----------------------------------------------------------------------------

def _passes(rule: Optional[Tuple[str, str]], item: str) -> bool:
    if rule is None:
        return True
    kind, x = rule
    return item == x if kind == "only" else item != x


FLAMETHROWER = "<flamethrower>"  # any of FLAMETHROWER_FLUIDS


def _fluid_roles(L: Layout) -> Tuple[Dict[int, List[Tuple[List[int], str]]], Dict[int, List[Tuple[int, str]]]]:
    """
    Per machine: [(input boxes, fluid)] the fluid may come through and [(output box, fluid)] it puts
    out. A fluid uses the box its recipe's fluidbox_index names (1-based among the machine's input or
    output boxes). Otherwise an ingredient may use any free input box, several products take the
    output boxes in order, and a single product the box something is connected to: the game's rule
    for fluids without an index is not documented, and these choices match the corpus (see the
    review of 27.10). Where the boxes are follows the machine's mirror flag (Layout._fluid_boxes).
    """
    if getattr(L, "_roles", None) is not None:  # independent of the inputs: once per layout
        return L._roles
    needs: Dict[int, List[Tuple[List[int], str]]] = {}
    makes: Dict[int, List[Tuple[int, str]]] = {}
    by_entity: Dict[int, List[int]] = {}
    for b, (j, k, ptype, flt) in enumerate(L.boxes):
        by_entity.setdefault(j, []).append(b)
    net_size = Counter(L.network)
    open_boxes = {b for b, _, _ in L.open_ends}
    for j, bs in by_entity.items():
        e, t = L.ents[j], L.types[j]
        r = recipe(e.extras_dict.get("recipe") or "")
        ins = [b for b in bs if L.boxes[b][2] == "input"]
        outs = [b for b in bs if L.boxes[b][2] == "output"]
        if t == "assembling-machine" and r:
            # a fluid ingredient may come through any input box the recipe does not give to another one
            # (popular 0.16 / 1.x designs feed lubricant's heavy oil through the second input)
            named_in = {f: ins[i - 1] for f, i in r.box_in.items() if 0 < i <= len(ins)}
            free_in = [b for b in ins if b not in named_in.values()]
            needs[j] = [([named_in[f]] if f in named_in else free_in, f) for f in r.ingredients if f in r.fluids]
            fout = [f for f in r.results if f in r.fluids]
            if len(fout) == 1 and fout[0] not in r.box_out and len(outs) > 1:
                # one fluid product: out of the box something is connected to (else the first)
                linked = [b for b in outs if net_size[L.network[b]] > 1 or b in open_boxes]
                makes[j] = [((linked or outs)[0], fout[0])]
            else:
                makes[j] = _boxes_for(fout, outs, r.box_out, lambda b, f: (b, f))
        elif t == "boiler":
            needs[j] = [(ins, L.boxes[ins[0]][3] or "water")] if ins else []
            makes[j] = [(b, L.boxes[b][3] or "steam") for b in outs]
        elif t == "generator":
            needs[j] = [(bs, "steam")]
        elif t == "offshore-pump":
            makes[j] = [(b, "water") for b in bs]
        elif t == "mining-drill" and e.name == "pumpjack":
            makes[j] = [(b, "crude-oil") for b in bs]
        elif t == "fluid-turret":
            needs[j] = [(bs, FLAMETHROWER)]
    L._roles = (needs, makes)
    return needs, makes


def _boxes_for(fluids: List[str], boxes: List[int], index: Dict[str, int], make) -> list:
    """Each fluid on its box: the recipe's fluidbox_index, else the next box no index claims."""
    named = {f: boxes[i - 1] for f, i in index.items() if 0 < i <= len(boxes)}
    unclaimed = [b for b in boxes if b not in named.values()]
    out = []
    for f in fluids:
        b = named[f] if f in named else (unclaimed.pop(0) if unclaimed else None)  # box 0 is a valid id
        if b is not None:
            out.append(make(b, f))
    return out


def _has_fluid(fluid: str, have: Set[str]) -> bool:
    return bool(have & set(FLAMETHROWER_FLUIDS)) if fluid == FLAMETHROWER else fluid in have


def check(ir: IRBlueprint, inputs: Dict[Any, Optional[str]], layout: Optional[Layout] = None,
          resource: str = "iron-ore") -> FlowResult:
    """Simulates one assignment of input items ({('lane', i, l) | ('chest', j) | ('fluid', net): item})."""
    L = layout or Layout(ir, resource)
    result = _simulate(L, inputs)
    result.inputs = dict(inputs)
    return result


def _simulate(L: Layout, inputs: Dict[Any, Optional[str]]) -> FlowResult:
    """
    The flow from these inputs. A fluid loop that feeds itself (coal liquefaction needs the heavy
    oil it makes) works once primed, as players do in game: a fluid a stopped machine would put into
    a network where it is needed is assumed present, and assumptions whose makers still do not work
    are dropped until the rest holds up.
    """
    result = _fixpoint(L, inputs, frozenset())
    if getattr(L, "_priming", None) is None:
        needs_f, makes_f = _fluid_roles(L)
        needed = {(L.network[b], f) for ns in needs_f.values() for bs, f in ns for b in bs if f != FLAMETHROWER}
        makers: Dict[Tuple[int, str], Set[int]] = {}
        for j, made in makes_f.items():
            for b, f in made:
                if any((n, f) in needed for n in L.downstream_nets(L.network[b])):  # there or past pumps
                    makers.setdefault((L.network[b], f), set()).add(j)
        L._priming = makers
    makers = L._priming
    assumed = frozenset(nf for nf, js in makers.items() if not js & result.working)
    while assumed:
        trial = _fixpoint(L, inputs, assumed)
        kept = frozenset(nf for nf in assumed if makers[nf] & trial.working)
        if kept == assumed:
            return trial
        assumed = kept
    return result


def _fixpoint(L: Layout, inputs: Dict[Any, Optional[str]], primed: FrozenSet[Tuple[int, str]]) -> FlowResult:
    """Items and fluids spread from the inputs (and the primed (network, fluid) pairs) until nothing changes."""
    ents = L.ents
    lanes: Dict[Lane, Set[str]] = {k: set() for k in L.edges}
    received: Dict[int, Set[str]] = {}
    ground: Dict[Tile, Set[str]] = {}
    chests: Dict[int, Set[str]] = {}
    for j, t in enumerate(L.types):
        if t in CONTAINERS:
            req = ents[j].extras_dict.get("requests") or ""
            chests[j] = {p.split("=")[0] for p in req.split(";") if p}
    for key, item in inputs.items():
        if not item:
            continue
        if key[0] == "lane":
            lanes[(key[1], key[2])].add(item)
        elif key[0] == "chest":
            chests.setdefault(key[1], set()).add(item)
    needs_f, makes_f = _fluid_roles(L)
    net_fluids: Dict[int, Set[str]] = {}
    for key, item in inputs.items():
        if key[0] == "fluid" and item:
            net_fluids.setdefault(key[1], set()).add(item)
    for net, fluid in primed:
        net_fluids.setdefault(net, set()).add(fluid)
    working: Set[int] = set()

    def works(j: int) -> bool:
        got = received.get(j, set())
        if L.types[j] in ("offshore-pump",) or (L.types[j] == "mining-drill" and not _burner(ents[j].name)):
            return True
        if any(not (need & got) for need in L.needs(j)):
            return False
        for bs, fluid in needs_f.get(j, []):
            have = set().union(*(net_fluids.get(L.network[b], set()) for b in bs)) if bs else set()
            if not _has_fluid(fluid, have):
                return False
        if L.types[j] == "assembling-machine":
            r = recipe(ents[j].extras_dict.get("recipe") or "")
            if r and r.fluids & set(r.ingredients) and j not in needs_f:
                return False  # a fluid recipe in a machine without fluid boxes
        return True

    def source_items(src: Tuple[str, Any]) -> Set[str]:
        kind, ref = src
        if kind == "belt":
            return set().union(*(lanes[k] for k in L.belt_lanes(ref)))
        if kind == "ground":
            return ground.get(ref, set())
        if kind == "entity":
            if L.types[ref] in CONTAINERS:
                return chests.get(ref, set())
            return set(L.products(ref, frozenset(received.get(ref, set())))) if ref in working or L.types[ref] in GIVES_OUT \
                else set()
        return set()

    changed = True
    rounds = 0
    while changed and rounds < 200:
        changed = False
        rounds += 1
        # belts
        stack = [k for k, v in lanes.items() if v]
        while stack:
            k = stack.pop()
            for dst, rule in L.edges[k]:
                new = {x for x in lanes[k] if _passes(rule, x)} - lanes[dst]
                if new:
                    lanes[dst] |= new
                    stack.append(dst)
        # drills
        for j, (kind, ref) in L.drills:
            if j not in working and works(j):
                working.add(j)
                changed = True
            if j in working:
                changed |= _put(L, kind, ref, {L.resource}, lanes, received, ground, chests)
        # inserters
        for idx, (src, dst) in enumerate(L.links):
            items = source_items(src)
            if not items:
                continue
            items = L.moves(idx, items)
            if items:
                changed |= _put(L, dst[0], dst[1], items, lanes, received, ground, chests)
        # machines and fluids
        for j in range(len(ents)):
            if j not in working and (L.is_machine(j) or j in makes_f or j in needs_f) and works(j):
                working.add(j)
                changed = True
                for b, fluid in makes_f.get(j, []):
                    net_fluids.setdefault(L.network[b], set()).add(fluid)
        for a, b in L.pumped:  # one way, from the input side to the output side
            new = net_fluids.get(a, set()) - net_fluids.get(b, set())
            if new:
                net_fluids.setdefault(b, set()).update(new)
                changed = True
    return _verdict(L, lanes, received, ground, chests, working, inputs, needs_f, makes_f, net_fluids)


def _put(L: Layout, kind: str, ref: Any, items: Set[str], lanes, received, ground, chests) -> bool:
    if kind == "lane":
        target = lanes[ref]
    elif kind == "ground":
        target = ground.setdefault(ref, set())
    elif kind == "entity":
        target = chests.setdefault(ref, set()) if L.types[ref] in CONTAINERS else received.setdefault(ref, set())
    else:
        return False
    new = items - target
    target |= new
    return bool(new)


def _picks(L: Layout) -> Dict[int, Optional[FrozenSet[str]]]:
    """Per belt entity: the items inserters take from it (ANY if a chest or the ground takes all)."""
    out: Dict[int, Optional[FrozenSet[str]]] = {}
    for idx, (src, dst) in enumerate(L.links):
        if src[0] != "belt":
            continue
        acc = ANY if dst[0] in ("ground", "lane") else L.accepts(dst[1]) if dst[0] == "entity" else frozenset()
        flt = L.filters[idx]
        if flt is not None and flt[0]:  # a whitelist takes only its items
            acc = flt[1] if acc is ANY else acc & flt[1]
        prev = out.get(src[1], frozenset())
        out[src[1]] = ANY if acc is ANY or prev is ANY else prev | acc
    return out


def _verdict(L: Layout, lanes, received, ground, chests, working, inputs, needs_f, makes_f, net_fluids) -> FlowResult:
    ents = L.ents
    problems: List[Problem] = []

    def pos(j: int) -> Tile:
        return (ents[j].x, ents[j].y)

    # what is produced inside (machine products that reach somewhere)
    products: Set[str] = set()
    for j in working:
        products |= set(L.products(j, frozenset(received.get(j, set())))) if L.types[j] not in CONTAINERS else set()
    for j, _ in L.drills:
        if j in working:
            products.add(L.resource)
    input_items = {item for item in inputs.values() if item}

    # machines
    for j in range(len(ents)):
        if not L.is_machine(j) and j not in needs_f:
            continue
        if L.types[j] == "generator" and j not in working:
            problems.append(Problem("starved", f"{ents[j].name} gets no steam", pos(j)))
            continue
        if j not in working:
            got = received.get(j, set())
            missing = [sorted(n)[0] if len(n) == 1 else "/".join(sorted(n)[:2]) for n in L.needs(j) if not n & got]
            missing += ["oil" if f == FLAMETHROWER else f for bs, f in needs_f.get(j, [])
                        if not _has_fluid(f, set().union(*(net_fluids.get(L.network[b], set()) for b in bs)) if bs else set())]
            problems.append(Problem("starved", f"{ents[j].name} at {pos(j)} lacks {', '.join(missing) or 'fluid'}", pos(j)))
    # research needs every pack in every lab: a lab without a pack the other labs get idles
    labs = [j for j in working if L.types[j] == "lab"]
    every_pack = set().union(*(received.get(j, set()) for j in labs)) if labs else set()
    for j in labs:
        missing = sorted(every_pack - received.get(j, set()))
        if missing:
            problems.append(Problem("starved", f"lab at {pos(j)} lacks {', '.join(missing)}", pos(j)))
    taken_from = {src[1] for src, _ in L.links if src[0] == "entity"}
    for j in working:
        if L.types[j] in ("assembling-machine", "furnace") and L.products(j, frozenset(received.get(j, set()))) \
                and j not in taken_from:
            problems.append(Problem("no_output", f"nobody takes the products of {ents[j].name} at {pos(j)}", pos(j)))

    # inserters
    for idx, (k, (src, dst)) in enumerate(zip(L.inserters, L.links)):
        e = ents[k]
        kind, ref = src
        if kind == "belt":
            items = set().union(*(lanes[k] for k in L.belt_lanes(ref)))
        elif kind == "ground":
            items = ground.get(ref, set())
        elif kind == "entity":
            items = chests.get(ref, set()) if L.types[ref] in CONTAINERS else \
                set(L.products(ref, frozenset(received.get(ref, set())))) if ref in working or L.types[ref] in GIVES_OUT else set()
        else:
            items = set()
        items = L.moves(idx, items)
        if not items:
            problems.append(Problem("dangling_inserter", f"{e.name} at {(e.x, e.y)} has nothing to pick", (e.x, e.y)))
        elif dst[0] == "ground" and not any(s == ("ground", dst[1]) for s, _ in L.links):
            problems.append(Problem("dangling_inserter", f"{e.name} at {(e.x, e.y)} drops on the ground", (e.x, e.y)))
        elif dst[0] == "none" or (dst[0] == "entity" and not L.accepts(dst[1]) and L.accepts(dst[1]) is not ANY):
            problems.append(Problem("dangling_inserter", f"{e.name} at {(e.x, e.y)} drops into nothing", (e.x, e.y)))

    # belts: clogs and mixed outputs
    need = L.lane_need()
    terminals = [k for k, outs in L.edges.items() if not outs and lanes[k]]
    outputs = {k for k in terminals if lanes[k] & products}
    leads_out = L.upstream(outputs)
    reported: Set[Tuple[str, int]] = set()
    for k, items in lanes.items():
        n = need[k]
        if not items or n is None or k in leads_out:
            continue
        stuck = items - n
        if stuck and items & n:
            for x in sorted(stuck):
                if (x, k[0]) not in reported:
                    reported.add((x, k[0]))
                    problems.append(Problem("clog", f"{x} blocks the belt at {pos(k[0])}", pos(k[0])))
    for k in outputs:
        mixed = (lanes[k] & input_items) - products
        if mixed:
            problems.append(Problem("output_mixed", f"{', '.join(sorted(mixed))} leaves with the products at {pos(k[0])}",
                                    pos(k[0])))

    # fluids
    for net in set(L.network):
        fluids = net_fluids.get(net, set())
        if len(fluids) > 1:
            b = L.network.index(net)
            problems.append(Problem("fluid_mix", f"{' and '.join(sorted(fluids))} in one pipe network",
                                    pos(L.boxes[b][0])))

    # ports
    ports: List[Port] = []
    chest_items: Dict[int, Set[str]] = {}  # a chest filled by hand may hold several items
    for key, item in inputs.items():
        if not item:
            continue
        if key[0] == "lane":
            e = ents[key[1]]
            ports.append(Port("lane", "input", key[1], key[2], frozenset({item}), (e.x, e.y), e.direction))
        elif key[0] == "chest":
            chest_items.setdefault(key[1], set()).add(item)
    for j, items in sorted(chest_items.items()):
        ports.append(Port("chest", "input", j, None, frozenset(items), pos(j)))
    for k in sorted(outputs):
        e = ents[k[0]]
        ports.append(Port("lane", "output", k[0], k[1], frozenset(lanes[k] & products), (e.x, e.y), e.direction))
    producing = {L.network[b] for j, made in makes_f.items() if j in working for b, _ in made}
    consuming = {L.network[b] for ns in needs_f.values() for bs, _ in ns for b in bs}
    labelled_nets: Set[int] = set()
    for b, tile, d in L.open_ends:  # one port per pipe network; fluid used up inside is no output
        net = L.network[b]
        fluids = net_fluids.get(net, set())
        if not fluids or net in labelled_nets or (net in producing and net in consuming):
            continue
        labelled_nets.add(net)
        ports.append(Port("fluid", "output" if net in producing else "input", L.boxes[b][0], None,
                          frozenset(fluids), tile, d))
    entity_fluids: Dict[int, Set[str]] = {}
    for b, (j, _, _, _) in enumerate(L.boxes):
        entity_fluids.setdefault(j, set()).update(net_fluids.get(L.network[b], set()))
    return FlowResult(problems, {k: frozenset(v) for k, v in lanes.items()},
                      {j: frozenset(v) for j, v in received.items()}, frozenset(working), ports,
                      fluids={j: frozenset(v) for j, v in entity_fluids.items()})


# --- Input assignment -----------------------------------------------------------------------

def input_ports(L: Layout) -> List[Tuple[Any, ...]]:
    """Belt lanes nobody feeds (no belt, inserter or drill puts anything on them), unfilled chests, open pipe networks."""
    dropped: Set[Lane] = {dst[1] for _, dst in L.links if dst[0] == "lane"} | \
                         {ref for _, (kind, ref) in L.drills if kind == "lane"}
    ports: List[Tuple[Any, ...]] = []
    fed = {k[0] for k in L.incoming} | {k[0] for k in dropped}
    for k in L.edges:
        if L.incoming.get(k) or k in dropped:
            continue
        if L.types[k[0]] == "splitter" and k[0] in fed:
            continue  # the free half of a splitter that a belt feeds is not an extra input
        ports.append(("lane", k[0], k[1]))
    filled = {dst[1] for _, dst in L.links if dst[0] == "entity"}
    for j, t in enumerate(L.types):
        if t in CONTAINERS and j not in filled and not L.ents[j].extras_dict.get("requests") \
                and any(src == ("entity", j) for src, _ in L.links):
            ports.append(("chest", j))
    return ports


def _fluid_domains(L: Layout) -> Dict[Tuple[Any, ...], List[Optional[str]]]:
    """
    Pipe networks with an open end and no producer inside are input ports: candidates are the fluids
    the machines on the network, or past the pumps it feeds, need (the most wanted first;
    flamethrowers take light oil). A network a pump fills is not an input.
    """
    needs_f, makes_f = _fluid_roles(L)
    produced = {L.network[b] for made in makes_f.values() for b, _ in made} | {b for _, b in L.pumped}
    wanted: Dict[int, Dict[str, int]] = {}
    for j, ns in needs_f.items():
        for bs, fluid in ns:
            for net in {L.network[b] for b in bs}:
                f = "light-oil" if fluid == FLAMETHROWER else fluid
                for up in [n for n in set(L.network) if net in L.downstream_nets(n)]:
                    wanted.setdefault(up, {})[f] = wanted.setdefault(up, {}).get(f, 0) + 1
    out: Dict[Tuple[Any, ...], List[Optional[str]]] = {}
    for b, _, _ in L.open_ends:
        net = L.network[b]
        if net in produced or net not in wanted or ("fluid", net) in out:
            continue
        out[("fluid", net)] = sorted(wanted[net], key=lambda f: (-wanted[net][f], f)) + [None]
    return out


def _reach_all(L: Layout, starts: Iterable[Lane]) -> Set[Lane]:
    """Lanes reachable from any of these lanes."""
    seen = set(starts)
    stack = list(seen)
    while stack:
        for dst, _ in L.edges[stack.pop()]:
            if dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return seen


def _reach(L: Layout, k: Lane, item: Optional[str] = None) -> Set[Lane]:
    """Lanes reachable from lane k (for an item: through the splitter filters it passes)."""
    seen, stack = {k}, [k]
    while stack:
        for dst, rule in L.edges[stack.pop()]:
            if dst not in seen and (item is None or _passes(rule, item)):
                seen.add(dst)
                stack.append(dst)
    return seen


class _Static:
    """Assignment-independent facts used to prune the search: what is taken where, inner lanes, outputs."""

    def __init__(self, L: Layout):
        self.L = L
        self.need = L.lane_need()
        # lanes that machines, drills or other inner sources put items on, and where those lead
        dropped = {dst[1] for src, dst in L.links if dst[0] == "lane" and src[0] != "belt"} | \
                  {ref for _, (kind, ref) in L.drills if kind == "lane"}
        self.inner: Set[Lane] = _reach_all(L, dropped)
        out_terms = {k for k in self.inner if not L.edges[k]}
        self.leads_out: Set[Lane] = L.upstream(out_terms)
        self.crafted: Set[str] = set()  # what recipes here make (never an input from outside)
        for j in range(len(L.ents)):
            r = recipe(L.ents[j].extras_dict.get("recipe") or "")
            if r:
                self.crafted |= set(r.results)
        self.made: Set[str] = set(self.crafted)  # anything a machine here could make
        if any(t == "furnace" for t in L.types):
            self.made |= set(SMELTING.values())
        if L.drills:
            self.made.add(L.resource)

    def safe(self, k: Lane, item: str) -> bool:
        """
        Whether item on lane k can work at all: it must not get stuck on a lane that products still
        have to pass, nor leave through an output with the products. (Whether it blocks other
        inputs depends on what else shares the lane: the simulation checks that.)
        """
        for n in _reach(self.L, k, item):
            need = self.need[n]
            if need is None:
                continue
            if item not in need and n in self.inner and n not in self.leads_out:
                return False
            if not self.L.edges[n] and n in self.inner and item not in self.made:
                return False  # would leave with the products
        return True


def _wants(L: Layout, dst: Tuple[str, Any], smelt: str, fuel: str) -> List[str]:
    """Items an inserter's target needs from outside, in order of preference."""
    if dst[0] != "entity":
        return []
    j = dst[1]
    out: List[str] = []
    for need in L.needs(j):
        if need == frozenset(SMELTING):
            out += [smelt] + sorted(set(SMELTING) - {smelt})
        elif need == frozenset(FUELS):
            out += [fuel] + [f for f in FUELS if f != fuel]
        elif need == frozenset(SCIENCE_PACKS):
            out += [x for x in SCIENCE_PACKS if x != "space-science-pack"]  # rocket launches bring that one
        else:
            out += sorted(need)
    if L.types[j] in AMMO:
        out += list(AMMO[L.types[j]])
    acc = L.accepts(j)
    if not out and acc:  # reactors, roboports: what they take
        out = sorted(acc)
    return out


def _demand(L: Layout, smelt: str, fuel: str) -> Dict[Tuple[Any, ...], Dict[str, int]]:
    """
    What could be of use arriving at each place of the flow graph, with a preference rank (lower is
    better): ('lane', lane), ('entity', j) and ('ground', tile). Machines want their ingredients; a
    chest, the ground or an entity that passes items on (PASS_ON) wants what the inserters taking
    from it deliver onward; a lane wants what is taken from it or from the lanes it flows into. So a
    manual chest feeding a belt, an inserter moving coal from one belt to another or ammunition
    passed from turret to turret all get their candidates.
    """
    demand: Dict[Tuple[Any, ...], Dict[str, int]] = {}

    def add(node: Tuple[Any, ...], items: Dict[str, int]) -> bool:
        have = demand.setdefault(node, {})
        grew = False
        for x, rank in items.items():
            if x not in have or rank < have[x]:
                have[x] = rank
                grew = True
        return grew

    for j in range(len(L.ents)):
        own = _wants(L, ("entity", j), smelt, fuel)
        if own:
            add(("entity", j), {x: r for r, x in enumerate(own)})
    into: Dict[Tuple[Any, ...], List[int]] = {}  # node -> inserters putting items there
    for idx, (_, dst) in enumerate(L.links):
        into.setdefault(("lane", dst[1]) if dst[0] == "lane" else dst, []).append(idx)
    stack = list(demand)
    while stack:  # what a node wants spreads back to what feeds it
        node = stack.pop()
        have = demand.get(node, {})
        if node[0] == "lane":
            for src_lane, rule in L.reverse[node[1]]:
                if add(("lane", src_lane), {x: r for x, r in have.items() if _passes(rule, x)}):
                    stack.append(("lane", src_lane))
        for idx in into.get(node, []):
            src = L.links[idx][0]
            wanted = {x: r for x, r in have.items() if L.moves(idx, {x})}
            if not wanted:
                continue
            if src[0] == "belt":
                targets = [("lane", lane) for lane in L.belt_lanes(src[1])]
            elif src[0] == "ground" or (src[0] == "entity" and (L.types[src[1]] in CONTAINERS or L.types[src[1]] in PASS_ON)):
                acc = L.accepts(src[1]) if src[0] == "entity" and L.types[src[1]] in PASS_ON else ANY
                wanted = {x: r for x, r in wanted.items() if acc is ANY or x in acc}
                targets = [src]
            else:
                targets = []
            for target in targets:
                if add(target, wanted):
                    stack.append(target)
    return demand


def _domains(L: Layout, ports: List[Tuple[Any, ...]], smelt: str, fuel: str, static: _Static
             ) -> Dict[Tuple[Any, ...], List[Optional[str]]]:
    """
    Candidate items per port: what is of use downstream of it (_demand), best ranked first, if safe
    for the lanes; what a machine here makes does not come from outside (cable in a circuit block).
    """
    demand = _demand(L, smelt, fuel)
    domains: Dict[Tuple[Any, ...], List[Optional[str]]] = {}
    for port in ports:
        node = ("lane", (port[1], port[2])) if port[0] == "lane" else ("entity", port[1])
        ranked = sorted(demand.get(node, {}).items(), key=lambda kv: (kv[1], kv[0]))
        cands = [x for x, _ in ranked if x not in static.crafted]
        if port[0] == "lane":
            cands = [x for x in cands if static.safe((port[1], port[2]), x)]
        domains[port] = cands + [None]
    return domains


def _constraints(L: Layout, ports: List[Tuple[Any, ...]], domains, static: _Static
                 ) -> List[List[Tuple[Tuple[Any, ...], str]]]:
    """
    'At least one of these (port, item) choices' for: every inserter that only the outside feeds
    (from belts or chests), and every need of a machine that nothing inside can meet.
    """
    serves: Dict[int, List[Tuple[Tuple[Any, ...], str]]] = {}
    for port in ports:
        for item in domains[port]:
            if item is None:
                continue
            if port[0] == "lane":
                lanes = _reach(L, (port[1], port[2]), item)
                hit = {idx for idx, (src, _) in enumerate(L.links) if src[0] == "belt"
                       and any(k in lanes for k in L.belt_lanes(src[1]))}
            else:
                hit = {idx for idx, (src, _) in enumerate(L.links) if src == ("entity", port[1])}
            for idx in hit:
                if L.moves(idx, {item}):
                    serves.setdefault(idx, []).append((port, item))
    inner_fed: Set[int] = set()  # inserters that may get something from inside
    for idx, (src, dst) in enumerate(L.links):
        if src[0] == "belt" and any(k in static.inner for k in L.belt_lanes(src[1])):
            inner_fed.add(idx)
        elif src[0] == "ground" or (src[0] == "entity" and L.types[src[1]] not in CONTAINERS):
            inner_fed.add(idx)
        elif src[0] == "entity" and any(d == ("entity", src[1]) for _, d in L.links):
            inner_fed.add(idx)
    out: List[List[Tuple[Tuple[Any, ...], str]]] = []
    for idx, (src, dst) in enumerate(L.links):
        if idx not in inner_fed and src[0] in ("belt", "entity"):
            out.append(serves.get(idx, []))
    feeding: Dict[int, List[int]] = {}
    for idx, (src, dst) in enumerate(L.links):
        if dst[0] == "entity":
            feeding.setdefault(dst[1], []).append(idx)
    for j, idxs in feeding.items():
        if not L.is_machine(j):
            continue
        for need in L.needs(j):
            if any(i in inner_fed for i in idxs) and need & static.made:
                continue
            out.append(list(dict.fromkeys((p, x) for i in idxs for p, x in serves.get(i, []) if x in need)))
    return out


def _doomed(L: Layout, ports: List[Tuple[Any, ...]]) -> bool:
    """
    Whether the layout has a problem whatever comes in: an inserter dropping into nothing (or on
    ground nobody picks from), a machine whose products nobody takes, or a machine or inserter that
    no flow reaches from any input or inner source (a machine missing its input inserter).
    """
    picked_ground = {src for src, _ in L.links if src[0] == "ground"}
    taken_from = {src[1] for src, _ in L.links if src[0] == "entity"}
    for src, dst in L.links:
        if dst[0] == "none" or (dst[0] == "ground" and dst not in picked_ground) or (
                dst[0] == "entity" and L.accepts(dst[1]) is not ANY and not L.accepts(dst[1])):
            return True
    for j, t in enumerate(L.types):
        if t in ("assembling-machine", "furnace") and L.is_machine(j) and j not in taken_from:
            r = recipe(L.ents[j].extras_dict.get("recipe") or "")
            if t == "furnace" or (r and set(r.results) - r.fluids):
                return True
    # what can reach where: from the input ports, request chests, drills and fluid sources
    needs_f, makes_f = _fluid_roles(L)
    out_of: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
    for idx, (src, dst) in enumerate(L.links):
        node = ("entity", src[1]) if src[0] == "belt" else src
        out_of.setdefault(node, []).append(("inserter", idx))
        out_of.setdefault(("inserter", idx), []).append(("entity", dst[1][0]) if dst[0] == "lane" else dst)
    for (i, _), outs in L.edges.items():
        out_of.setdefault(("entity", i), []).extend(("entity", j) for (j, _), _ in outs)
    for b, (j, _, _, _) in enumerate(L.boxes):
        out_of.setdefault(("net", L.network[b]), []).append(("entity", j))
    for j, made in makes_f.items():
        out_of.setdefault(("entity", j), []).extend(("net", L.network[b]) for b, _ in made)
    for a, b in L.pumped:
        out_of.setdefault(("net", a), []).append(("net", b))
    starts = [("entity", p[1]) if p[0] != "fluid" else ("net", p[1]) for p in ports]
    starts += [("entity", j) for j, t in enumerate(L.types) if t in CONTAINERS and L.ents[j].extras_dict.get("requests")]
    starts += [("entity", j) for j, _ in L.drills] + [("entity", j) for j in makes_f if j not in needs_f]
    seen = set(starts)
    stack = list(starts)
    while stack:
        for n in out_of.get(stack.pop(), []):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    if any(("inserter", idx) not in seen for idx in range(len(L.links))):
        return True
    return any(("entity", j) not in seen for j in range(len(L.ents))
               if L.is_machine(j) and (L.needs(j) or needs_f.get(j)))


def _same_fuels(L: Layout, constraints: List[List[Tuple[Tuple[Any, ...], str]]], hints: Dict[Tile, Set[str]]
                ) -> List[List[Tuple[Tuple[Any, ...], str]]]:
    """
    Fuels nothing in the layout names (no inserter or splitter filter, chest request, recipe or
    author label) behave the same everywhere: each port keeps only the first of them (the preferred
    fuel). Otherwise every need of fuel branches five ways and the search runs out of budget.
    """
    named: Set[str] = set().union(*hints.values()) if hints else set()
    for flt in L.filters:
        if flt is not None:
            named |= flt[1]
    for e in L.ents:
        ex = e.extras_dict
        named |= {ex["filter"]} if ex.get("filter") else set()
        named |= {p.split("=")[0] for p in (ex.get("requests") or "").split(";") if p}
        r = recipe(ex.get("recipe") or "")
        if r:
            named |= set(r.ingredients) | set(r.results)
    out = []
    for options in constraints:
        seen: Set[Tuple[Any, ...]] = set()
        kept = []
        for p, x in options:
            key = (p[:2] if p[0] == "chest" else p, "<fuel>") if x in FUELS and x not in named else (p, x)
            if key not in seen:
                seen.add(key)
                kept.append((p, x))
        out.append(kept)
    return out


def _fluid_constraints(L: Layout, fluid_domains) -> List[List[Tuple[Tuple[Any, ...], str]]]:
    """
    Every fluid a machine needs must come through one of its input networks, or one that pumps
    feed them from (unless made inside).
    """
    needs_f, makes_f = _fluid_roles(L)
    made = {(L.network[b], f) for ms in makes_f.values() for b, f in ms}
    out = []
    for j, ns in needs_f.items():
        for bs, fluid in ns:
            own = {L.network[b] for b in bs}
            nets = {n for n in set(L.network) if L.downstream_nets(n) & own}
            if any((n, fluid) in made for n in nets) or (fluid == FLAMETHROWER and any(n in nets for n, _ in made)):
                continue
            f = "light-oil" if fluid == FLAMETHROWER else fluid
            out.append([(("fluid", n), f) for n in sorted(nets) if f in fluid_domains.get(("fluid", n), [])])
    return out


def _score(L: Layout, result: FlowResult, smelt: str, fuel: str, hints: Dict[Tile, Set[str]]) -> Tuple[float, ...]:
    """
    Ranks working assignments: agreement with the author's labels, the preferred smelting input and
    fuel, whole belts (one item on both lanes) over lane-split ones, then supply against demand (an
    item that machines use fast should be on the lane more or faster inserters take from: green
    circuits on the near lane), then fewer used lanes.
    """
    agree = 0
    for key, item in result.inputs.items():
        if key[0] in ("lane", "chest") and item:
            e = L.ents[key[1]]
            agree += any(item in items for (x, y), items in hints.items() if abs(x - e.x) <= 3 and abs(y - e.y) <= 3)
    prefer = sum(item in (smelt, fuel) for item in result.inputs.values())
    used = sum(1 for key, item in result.inputs.items() if item and key[0] != "fluid")
    supply: Dict[str, float] = {}
    demand: Dict[str, float] = {}
    for idx, (k, (src, dst)) in enumerate(zip(L.inserters, L.links)):
        if src[0] != "belt" or dst[0] != "entity":
            continue
        for x in L.moves(idx, set().union(*(result.lanes[lane] for lane in L.belt_lanes(src[1])))):
            supply[x] = supply.get(x, 0.0) + INSERTER_RATE.get(L.ents[k].name, 1.0)
    for j in result.working:
        r = recipe(L.ents[j].extras_dict.get("recipe") or "")
        if r and r.time:
            speed = float((_prototype(L.ents[j].name) or {}).get("crafting_speed") or 1)
            for x, n in r.ingredients.items():
                demand[x] = demand.get(x, 0.0) + n * speed / r.time
    ratios = [supply[x] / demand[x] for x in demand if x in supply and demand[x] > 0]
    # a belt with a different item on each lane only when needed: whole belts are easier to feed
    by_belt: Dict[int, Set[str]] = {}
    for key, item in result.inputs.items():
        if key[0] == "lane" and item:
            by_belt.setdefault((key[1], key[2] // 2), set()).add(item)
    mixed = sum(1 for items in by_belt.values() if len(items) > 1)
    return (agree, prefer, -mixed, round(min(ratios), 3) if ratios else 0.0, -used)


MAX_IMPROVE = 120  # simulations spent on improving the chosen assignment


def _improve(ir: IRBlueprint, L: Layout, best: FlowResult, domains, smelt: str, fuel: str,
             hints: Dict[Tile, Set[str]]) -> FlowResult:
    """
    Local search on a working assignment: other candidates on one port at a time, kept if the
    layout still works and more different items come in (labs fed all their science packs, not one
    pack on every lane; alternatives such as fuels do not count), or the score improves otherwise.
    """
    # fuels, ammunition and smelting inputs are alternatives: one is enough, more is no better
    alternatives = set(FUELS) | {a for kinds in AMMO.values() for a in kinds} | set(SMELTING)

    labs = [j for j, t in enumerate(L.types) if t == "lab"]

    def key(r: FlowResult) -> Tuple[Any, ...]:
        distinct = len({x for k, x in r.inputs.items() if x and k[0] != "fluid" and x not in alternatives})
        # research needs every pack in every lab: the worst-fed lab first (a copy left with red only)
        poorest = min((len(r.machines.get(j, ())) for j in labs), default=0)
        return (r.score[0], r.score[1], poorest, distinct) + tuple(r.score[2:])  # then whole belts, supply, fewer lanes

    budget = MAX_IMPROVE
    improved = True
    while improved and budget > 0:
        improved = False
        for port, current in list(best.inputs.items()):
            if port[0] == "fluid":
                continue
            for alt in domains.get(port, []):
                if alt == current or not alt or budget <= 0:
                    continue
                budget -= 1
                trial = check(ir, {**best.inputs, port: alt}, L)
                if trial.problems:
                    continue
                trial.score = _score(L, trial, smelt, fuel, hints)
                if key(trial) > key(best):
                    best, improved = trial, True
                    break
    return best


def _parts(L: Layout) -> List[List[int]]:
    """
    Groups of entities that share flow (lanes, inserters with what they pick from and drop into,
    drills, pipe connections); no item or fluid passes between two groups, so each can get its
    inputs on its own. Entities without any flow (poles, beacons, lamps) are left out.
    """
    parent: Dict[Any, Any] = {}

    def root(a: Any) -> Any:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a: Any, b: Any) -> None:
        parent[root(a)] = root(b)

    def node(end: Tuple[str, Any]) -> Optional[Any]:
        kind, ref = end
        if kind == "lane":
            return ref[0]
        if kind in ("belt", "entity"):
            return ref
        return ("ground", ref) if kind == "ground" else None

    for (i, _), outs in L.edges.items():
        root(i)
        for (j, _), _ in outs:
            join(i, j)
    for i, (src, dst) in zip(L.inserters, L.links):
        for end in (src, dst):
            n = node(end)
            join(i, i) if n is None else join(i, n)
    for j, drop in L.drills:
        n = node(drop)
        join(j, j) if n is None else join(j, n)
    for a, b in L.pipe_links:
        join(a, b)
    for j, _, _, _ in L.boxes:
        root(j)
    groups: Dict[Any, List[int]] = {}
    for a in list(parent):
        if not isinstance(a, tuple):
            groups.setdefault(root(a), []).append(a)
    return [sorted(g) for g in groups.values()]


def _part_blueprint(ir: IRBlueprint, L: Layout, part: List[int], hints: Dict[Tile, Set[str]]
                    ) -> Tuple[IRBlueprint, List[int], Dict[Tile, Set[str]], Tile]:
    """
    One part as a blueprint of its own, moved to the origin and in a fixed order (identical parts
    give identical blueprints): the part's entities, then a wall on every neighbouring tile another
    part occupies (a pipe that ends against another machine is not an open end). Returns the
    blueprint, the full index of each of its entities, the author's labels moved with it, the shift.
    """
    ox = min(L.ents[i].x for i in part)
    oy = min(L.ents[i].y for i in part)
    order = sorted(part, key=lambda i: (L.ents[i].y, L.ents[i].x, L.ents[i].name, L.ents[i].direction, L.ents[i].extras))
    own = {t for i in part for t in _tiles(L.ents[i])}
    blocked = sorted({(x + dx, y + dy) for x, y in own for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if (x + dx, y + dy) in L.tiles and (x + dx, y + dy) not in own}, key=lambda t: (t[1], t[0]))
    ents = [IREntity(L.ents[i].name, L.ents[i].x - ox, L.ents[i].y - oy, L.ents[i].direction, L.ents[i].extras)
            for i in order] + [IREntity("stone-wall", x - ox, y - oy) for x, y in blocked]
    xs = [t[0] for t in own]
    ys = [t[1] for t in own]
    near = {(x - ox, y - oy): v for (x, y), v in hints.items()
            if min(xs) - 3 <= x <= max(xs) + 3 and min(ys) - 3 <= y <= max(ys) + 3}
    return ir.with_entities(ents), order, near, (ox, oy)


def solve(ir: IRBlueprint, smelt: str = "iron-ore", fuel: str = "coal", resource: str = "iron-ore",
          hints: Optional[Dict[Tile, Set[str]]] = None, max_leaves: int = MAX_LEAVES) -> FlowResult:
    """
    Finds the input assignment under which the layout works (no problems), see _solve_whole. A
    layout made of parts that share no flow (furnace columns side by side, separate lines of a
    book page) is solved part by part, each with the full search budget, and identical parts only
    once: one search over all of them at once runs out of budget long before it tries every
    combination. The joined assignment is simulated on the whole layout.
    """
    L = Layout(ir, resource)
    hints = hints or {}
    fluid_nets = {port[1] for port in _fluid_domains(L)}
    at_ports = {port[1] for port in input_ports(L)} | {j for b, (j, _, _, _) in enumerate(L.boxes)
                                                       if L.network[b] in fluid_nets}
    parts = [p for p in _parts(L) if at_ports.intersection(p)]
    if len(parts) <= 1:
        return _solve_whole(ir, L, smelt, fuel, hints, max_leaves)
    box_of = {(j, k): b for b, (j, k, _, _) in enumerate(L.boxes)}
    solved: Dict[Any, Tuple[Layout, FlowResult]] = {}
    inputs: Dict[Any, Optional[str]] = {}
    ambiguous = False
    for part in parts:
        sub, order, near, _ = _part_blueprint(ir, L, part, hints)
        key = (tuple(sub.entities), tuple(sorted((t, tuple(sorted(v))) for t, v in near.items())))
        if key not in solved:
            S = Layout(sub, resource)
            solved[key] = (S, _solve_whole(sub, S, smelt, fuel, near, max_leaves))
        S, res = solved[key]
        ambiguous |= res.status == "inputs_ambiguous"
        for port, item in res.inputs.items():
            if port[0] == "fluid":
                b = S.network.index(port[1])
                j, k = S.boxes[b][0], S.boxes[b][1]
                inputs[("fluid", L.network[box_of[(order[j], k)]])] = item
            else:
                inputs[(port[0], order[port[1]]) + tuple(port[2:])] = item
    result = check(ir, inputs, L)
    if result.problems:
        result.status = "inputs_ambiguous" if ambiguous else "inputs_unsolved"
    else:
        result.score = _score(L, result, smelt, fuel, hints)
    return result


def _solve_whole(ir: IRBlueprint, L: Layout, smelt: str, fuel: str, hints: Dict[Tile, Set[str]],
                 max_leaves: int) -> FlowResult:
    """
    The search for one layout: it picks the open constraint with the fewest options (see
    _constraints), tries each (port, item) choice and simulates the layout once all constraints
    are met. Among working assignments the best by _score wins. Without one, the result with the
    fewest problems is returned with status 'inputs_unsolved', or 'inputs_ambiguous' when the
    search ran out of MAX_LEAVES simulations.
    """
    static = _Static(L)
    ports = input_ports(L)
    domains = _domains(L, ports, smelt, fuel, static)
    # a chest filled by hand holds what the player puts in, several items too (the hub chest of a
    # hand-fed starter base): one yes / no port ('chest', j, item) per candidate item
    for p in [p for p in ports if p[0] == "chest"]:
        ports.remove(p)
        for x in domains.pop(p):
            if x:
                ports.append(("chest", p[1], x))
                domains[("chest", p[1], x)] = [x, None]
    fluid_domains = _fluid_domains(L)
    constraints = _same_fuels(L, _constraints(L, ports, domains, static), hints) + _fluid_constraints(L, fluid_domains)
    ports += list(fluid_domains)
    domains.update(fluid_domains)
    # no assignment can make the whole layout work: running out of budget is then not the reason
    # (the search still runs: copies made to dock are judged on their middle one only)
    doomed = _doomed(L, ports)
    solutions: List[FlowResult] = []
    tried: Set[FrozenSet[Tuple[Any, Optional[str]]]] = set()
    best_fail: List[Optional[FlowResult]] = [None]
    leaves = [0]
    first: List[Optional[int]] = [None]  # leaf count when the first working assignment was found

    port_set = set(ports)

    def simulate(assign: Dict[Tuple[Any, ...], str]) -> bool:
        key = frozenset(assign.items())
        if key in tried:
            return False
        tried.add(key)
        leaves[0] += 1
        result = check(ir, {p: assign.get(p) for p in ports}, L)
        if not result.problems:
            solutions.append(result)
            first[0] = first[0] if first[0] is not None else leaves[0]
            return True
        if best_fail[0] is None or len(result.problems) < len(best_fail[0].problems):
            best_fail[0] = result
        return False

    def leaf(assign: Dict[Tuple[Any, ...], str]) -> None:
        # an input belt is a whole belt: a lane next to a loaded input lane carries the same item
        # (or another one the search chose), never nothing - unless products are put on that lane
        # further on (a mixed belt: one lane in, the other out)
        assign = dict(assign)
        filled = []
        for p, x in list(assign.items()):
            other = ("lane", p[1], p[2] ^ 1) if p[0] == "lane" else None
            if x and other in port_set and other not in assign and not _reach(L, (p[1], p[2] ^ 1)) & static.inner:
                assign[other] = x
                filled.append(other)
        for p in fluid_domains:  # a pipe network the search did not need still gets its fluid
            assign.setdefault(p, fluid_domains[p][0])
        if simulate(assign) or not filled:
            return
        # the same item on both lanes does not work: try other items on the filled lanes (lane-split belts)
        options = [[x for x in domains[p] if x] for p in filled]
        for combo in itertools.islice(itertools.product(*options), 16):
            if done():
                return
            if simulate({**assign, **dict(zip(filled, combo))}):
                return

    visited: Set[FrozenSet[Tuple[Any, str]]] = set()  # partial assignments already searched (in any order)
    steps = [0]

    def done() -> bool:
        return leaves[0] >= max_leaves or len(solutions) >= MAX_SOLUTIONS or steps[0] >= MAX_STEPS * max_leaves or (
            first[0] is not None and leaves[0] - first[0] >= MAX_EXTRA_LEAVES)

    def search(assign: Dict[Tuple[Any, ...], str]) -> None:
        if done():
            return
        state = frozenset(assign.items())
        if state in visited:
            return
        visited.add(state)
        steps[0] += 1
        open_ = []
        for options in constraints:
            if not any(assign.get(p) == x for p, x in options):
                live = [(p, x) for p, x in options if p not in assign]
                if live:  # a constraint nothing can meet any more is left to the simulation to report
                    open_.append(live)
        if not open_:
            leaf(assign)
            return
        for p, x in min(open_, key=len):
            search({**assign, p: x})
            if done():
                return

    search({})
    if solutions:
        for s in solutions:
            s.score = _score(L, s, smelt, fuel, hints)
        return _improve(ir, L, max(solutions, key=lambda s: s.score), domains, smelt, fuel, hints)
    result = best_fail[0] or check(ir, {p: (fluid_domains[p][0] if p in fluid_domains else None) for p in ports}, L)
    ran_out = leaves[0] >= max_leaves or steps[0] >= MAX_STEPS * max_leaves
    result.status = "inputs_ambiguous" if ran_out and not doomed else "inputs_unsolved"
    return result
