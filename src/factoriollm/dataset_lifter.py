"""
Macro-Lifter for FactorioLLM.
Turns a blueprint IR (integer tile grid, see blueprint_ir.py) into concise, idiomatic draftsman
code using draftsman_helpers (add_belt_line, add_underground_pair, add_entity_row) and Python loops.

The output is deterministic: one blueprint always yields the same script. Every value embedded
into code is either an int, a validated prototype name, or passed through repr(), so scraped data
cannot inject code.
"""

from __future__ import annotations

import contextlib
import io
import sys
from typing import Dict, List, Optional, Set, Tuple

from factoriollm.blueprint_ir import (
    IRBlueprint,
    IREntity,
    direction_name,
    entity_type,
    parse_blueprint_string,
)
from factoriollm.blueprint_ir import _prototype  # prototype lookup for underground reach


HELPER_NAMES = ("add_belt_line", "add_underground_pair", "add_entity_row", "add_power_poles")

MIN_RUN = 3          # minimal belt line / entity row length worth a macro
MIN_LOOP_GROUP = 3   # minimal number of same-kind calls worth a for-loop


def find_arithmetic_runs(values: List[int], min_len: int = MIN_RUN) -> List[Tuple[int, int, int]]:
    """
    Greedily covers integer positions with arithmetic progressions, longest first.
    Returns (start, step, count) for every progression of at least `min_len` members;
    e.g. [1, 2, 5, 6, 9, 10] -> [(1, 4, 3), (2, 4, 3)].
    """
    remaining = sorted(set(values))
    runs = []
    while len(remaining) >= min_len:
        present = set(remaining)
        best = None
        for i, start in enumerate(remaining):
            for nxt in remaining[i + 1:]:
                step = nxt - start
                if best is not None and (remaining[-1] - start) // step + 1 <= best[2]:
                    break  # larger steps cannot beat the current best
                count = 2
                while start + count * step in present:
                    count += 1
                if best is None or count > best[2]:
                    best = (start, step, count)
        if best is None or best[2] < min_len:
            break
        runs.append(best)
        members = {best[0] + k * best[1] for k in range(best[2])}
        remaining = [v for v in remaining if v not in members]
    return runs


def _kwargs_str(extras: Tuple[Tuple[str, str], ...]) -> str:
    return "".join(f", {k}={v!r}" for k, v in extras)


def _tile(x: int, y: int) -> str:
    return f"({x}, {y})"


def _direction_arg(e: IREntity) -> str:
    return f", direction=Direction.{direction_name(e.direction).upper()}" if e.directional and e.direction else ""


def _extract_underground_pairs(ents: List[IREntity], used: Set[int]) -> List[Tuple[IREntity, IREntity]]:
    pairs = []
    for i, e1 in enumerate(ents):
        if i in used or entity_type(e1.name) != "underground-belt" or e1.extras_dict.get("io_type", "input") != "input":
            continue
        reach = int((_prototype(e1.name) or {}).get("max_distance") or 5)
        dx, dy = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}[e1.direction]
        best_j, best_dist = None, None
        for j, e2 in enumerate(ents):
            if j in used or j == i or e2.name != e1.name or e2.direction != e1.direction:
                continue
            if e2.extras_dict.get("io_type") != "output":
                continue
            ox, oy = e2.x - e1.x, e2.y - e1.y
            dist = ox * dx + oy * dy
            if dist <= 0 or dist > reach or (ox * dy != 0) or (oy * dx != 0):
                continue
            if best_dist is None or dist < best_dist:
                best_j, best_dist = j, dist
        if best_j is not None:
            used.update((i, best_j))
            pairs.append((e1, ents[best_j]))
    return pairs


def _extract_belt_lines(ents: List[IREntity], used: Set[int]) -> List[Tuple[IREntity, int]]:
    by_kind: Dict[Tuple[str, int], List[Tuple[int, IREntity]]] = {}
    for i, e in enumerate(ents):
        if i not in used and entity_type(e.name) == "transport-belt":
            by_kind.setdefault((e.name, e.direction), []).append((i, e))

    lines = []
    for (_, direction), items in by_kind.items():
        dx, dy = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}[direction]
        positions = {(e.x, e.y): (i, e) for i, e in items}
        for (x, y), (i, e) in sorted(positions.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if i in used or (x - dx, y - dy) in positions:
                continue  # not the head of a line
            run = [(i, e)]
            while (x + dx * len(run), y + dy * len(run)) in positions:
                run.append(positions[(x + dx * len(run), y + dy * len(run))])
            if len(run) >= MIN_RUN:
                used.update(idx for idx, _ in run)
                lines.append((e, len(run)))
    return lines


def _extract_rows(ents: List[IREntity], used: Set[int]) -> List[Tuple[IREntity, int, Tuple[int, int]]]:
    by_kind: Dict[Tuple[str, int, tuple], List[int]] = {}
    for i, e in enumerate(ents):
        if i not in used:
            by_kind.setdefault((e.name, e.direction, e.extras), []).append(i)

    rows = []
    for indices in by_kind.values():
        for horizontal in (True, False):  # rows along x (same y), then columns along y (same x)
            lines: Dict[int, Dict[int, int]] = {}
            for i in indices:
                if i not in used:
                    e = ents[i]
                    fixed, moving = (e.y, e.x) if horizontal else (e.x, e.y)
                    lines.setdefault(fixed, {})[moving] = i
            for fixed, members in sorted(lines.items()):
                for start, step, count in find_arithmetic_runs(list(members)):
                    used.update(members[start + k * step] for k in range(count))
                    rows.append((ents[members[start]], count, (step, 0) if horizontal else (0, step)))
    return rows


def lift_ir_to_code(ir: IRBlueprint, use_macros: bool = True) -> str:
    """Generates a deterministic draftsman script that rebuilds the IR blueprint exactly."""
    ents = ir.entities
    used: Set[int] = set()
    lines: List[str] = []
    helpers: Set[str] = set()

    pairs = _extract_underground_pairs(ents, used) if use_macros else []
    belts = _extract_belt_lines(ents, used) if use_macros else []
    rows = _extract_rows(ents, used) if use_macros else []

    # Belt lines, grouped by (type, direction) into loops
    belt_groups: Dict[Tuple[str, int], List[Tuple[IREntity, int]]] = {}
    for e, length in belts:
        belt_groups.setdefault((e.name, e.direction), []).append((e, length))
    if belt_groups:
        helpers.add("add_belt_line")
        lines.append("# Belt lines")
        for (name, direction), group in sorted(belt_groups.items()):
            d = direction_name(direction)
            if len(group) >= MIN_LOOP_GROUP:
                specs = ", ".join(f"({_tile(e.x, e.y)}, {n})" for e, n in group)
                lines.append(f"for start, length in [{specs}]:")
                lines.append(f"    add_belt_line(bp, start, length, direction={d!r}, belt_type={name!r})")
            else:
                for e, n in group:
                    lines.append(f"add_belt_line(bp, start={_tile(e.x, e.y)}, length={n}, direction={d!r}, belt_type={name!r})")
        lines.append("")

    # Underground pairs, grouped by (type, direction)
    pair_groups: Dict[Tuple[str, int], List[Tuple[IREntity, IREntity]]] = {}
    for a, b in pairs:
        pair_groups.setdefault((a.name, a.direction), []).append((a, b))
    if pair_groups:
        helpers.add("add_underground_pair")
        lines.append("# Underground belts")
        for (name, direction), group in sorted(pair_groups.items()):
            d = direction_name(direction)
            if len(group) >= MIN_LOOP_GROUP:
                specs = ", ".join(f"({_tile(a.x, a.y)}, {_tile(b.x, b.y)})" for a, b in group)
                lines.append(f"for start, end in [{specs}]:")
                lines.append(f"    add_underground_pair(bp, start, end, direction={d!r}, belt_type={name!r})")
            else:
                for a, b in group:
                    lines.append(f"add_underground_pair(bp, start={_tile(a.x, a.y)}, end={_tile(b.x, b.y)}, direction={d!r}, belt_type={name!r})")
        lines.append("")

    # Uniform rows / columns; identical rows (same kind, count and step) are grouped into loops
    row_groups: Dict[Tuple[str, int, tuple, int, Tuple[int, int]], List[IREntity]] = {}
    for e, count, step in rows:
        row_groups.setdefault((e.name, e.direction, e.extras, count, step), []).append(e)
    if row_groups:
        helpers.add("add_entity_row")
        lines.append("# Rows of identical entities")
        for (name, _, extras, count, step), group in sorted(row_groups.items(), key=lambda kv: (kv[1][0].y, kv[1][0].x, kv[0][0])):
            e = group[0]
            dir_kw = f", direction={direction_name(e.direction)!r}" if e.directional and e.direction else ""
            tail = f"count={count}, step={_tile(*step)}{dir_kw}{_kwargs_str(extras)})"
            if len(group) >= MIN_LOOP_GROUP:
                specs = ", ".join(_tile(g.x, g.y) for g in group)
                lines.append(f"for start in [{specs}]:")
                lines.append(f"    add_entity_row(bp, {name!r}, start=start, {tail}")
            else:
                for g in group:
                    lines.append(f"add_entity_row(bp, {name!r}, start={_tile(g.x, g.y)}, {tail}")
        lines.append("")

    # Remaining entities, grouped by kind into loops over positions
    singles: Dict[Tuple[str, int, tuple], List[IREntity]] = {}
    for i, e in enumerate(ents):
        if i not in used:
            singles.setdefault((e.name, e.direction, e.extras), []).append(e)
    if singles:
        lines.append("# Individual entities")
        for (name, _, extras), group in sorted(singles.items(), key=lambda kv: (kv[1][0].y, kv[1][0].x, kv[0][0])):
            tail = f"{_direction_arg(group[0])}{_kwargs_str(extras)}"
            if use_macros and len(group) >= MIN_LOOP_GROUP:
                specs = ", ".join(_tile(e.x, e.y) for e in group)
                lines.append(f"for x, y in [{specs}]:")
                lines.append(f"    bp.entities.append({name!r}, tile_position=(x, y){tail})")
            else:
                for e in group:
                    lines.append(f"bp.entities.append({name!r}, tile_position={_tile(e.x, e.y)}{tail})")
        lines.append("")

    header = ["from draftsman.blueprintable import Blueprint"]
    if any("Direction." in line for line in lines):
        header.append("from draftsman.constants import Direction")
    if helpers:
        header.append(f"from draftsman_helpers import {', '.join(h for h in HELPER_NAMES if h in helpers)}")
    width, height = ir.size
    return "\n".join(
        header + ["", f"bp = Blueprint()  # {width}x{height} tiles", ""] + lines + ["print(bp.to_string())"]
    ) + "\n"


# --- Trusted local execution (only for code produced by this module) ----------------------

def run_trusted_script(code: str) -> str:
    """
    Executes a lifter-generated script on the host and returns the printed blueprint string.
    NEVER call this with model- or teacher-generated code: use DockerSandbox for that.
    """
    import factoriollm.helpers as helpers

    sys.modules.setdefault("draftsman_helpers", helpers)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<lifted>", "exec"), {"__name__": "__main__"})
    return out.getvalue().strip().splitlines()[-1]


def ir_equivalent(expected: IRBlueprint, blueprint_string: str) -> bool:
    """True if the blueprint string rebuilds exactly the expected IR (same tiles, directions, extras)."""
    try:
        actual = parse_blueprint_string(blueprint_string, allowed=None)
    except Exception:
        return False
    return actual.canonical_set() == expected.canonical_set()


def lift_and_verify_locally(ir: IRBlueprint) -> Optional[str]:
    """Lifts the IR and verifies it on the host; returns the code or None if not equivalent."""
    code = lift_ir_to_code(ir)
    try:
        bp_string = run_trusted_script(code)
    except Exception:
        return None
    return code if ir_equivalent(ir, bp_string) else None
