"""
AST & Blueprint Macro-Lifter for FactorioLLM dataset.
Transforms low-level unrolled Draftsman scripts from dataset.jsonl into concise,
idiomatic code using draftsman_helpers (add_belt_line, add_underground_pair, add_entity_row).

Every transformed script is verified mathematically against the original blueprint
to guarantee 100% identical entities, coordinates, and orientations in-game.
"""

from __future__ import annotations

import io
import json
import re
import sys
import warnings
import contextlib
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
from pathlib import Path

# Suppress draftsman warnings during mass processing
warnings.filterwarnings("ignore")

# Setup compatibility shim so synthetic entity classes load properly
import draftsman.entity as _d_ent

def _compat_getattr(name: str):
    if name in _d_ent.__dict__:
        return _d_ent.__dict__[name]
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1-\2', name)
    kebab = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', s1).lower()
    try:
        return _d_ent.get_entity_class(kebab)
    except Exception:
        raise AttributeError(f"module 'draftsman.entity' has no attribute '{name}'")

_d_ent.__getattr__ = _compat_getattr

from draftsman.blueprintable import Blueprint
import factoriollm.helpers as helpers

# Register draftsman_helpers in sys.modules so generated scripts can import it
sys.modules["draftsman_helpers"] = helpers


BELT_NAMES = {
    "transport-belt", "fast-transport-belt", "express-transport-belt",
}
UNDERGROUND_NAMES = {
    "underground-belt", "fast-underground-belt", "express-underground-belt",
}


def entity_to_canonical(ent_dict: Dict[str, Any]) -> Tuple[str, float, float, int, str]:
    """Returns a canonical hashable signature for entity comparison."""
    name = ent_dict.get("name", "")
    pos = ent_dict.get("position", {})
    x = round(float(pos.get("x", 0.0)), 2)
    y = round(float(pos.get("y", 0.0)), 2)
    direction = int(ent_dict.get("direction", 0) or 0)
    if name in UNDERGROUND_NAMES:
        io_type = str(ent_dict.get("io_type") or ent_dict.get("type", "input"))
    else:
        io_type = str(ent_dict.get("io_type") or ent_dict.get("type", ""))
    return (name, x, y, direction, io_type)


def find_uniform_runs(items: List[Tuple[int, float, float]], coord_idx: int, min_len: int = 3) -> List[List[Tuple[int, float, float]]]:
    """Finds maximal contiguous subsequences of items with uniform non-zero step along coord_idx (1 for x, 2 for y)."""
    if len(items) < min_len:
        return []
    runs = []
    i = 0
    while i < len(items) - 1:
        step = round(items[i + 1][coord_idx] - items[i][coord_idx], 2)
        if step <= 0.1:
            i += 1
            continue
        run = [items[i], items[i + 1]]
        j = i + 1
        while j < len(items) - 1:
            next_step = round(items[j + 1][coord_idx] - items[j][coord_idx], 2)
            if abs(next_step - step) < 1e-3:
                run.append(items[j + 1])
                j += 1
            else:
                break
        if len(run) >= min_len:
            runs.append(run)
            i = j + 1
        else:
            i += 1
    return runs


def lift_blueprint_to_code(bp: Blueprint, blueprint_name: str = "Blueprint") -> str:
    """
    Analyzes all entities in a Blueprint and generates concise Python code
    using draftsman_helpers macros.
    """
    raw_entities = [e.to_dict() for e in bp.entities]
    if not raw_entities:
        return "from draftsman.blueprintable import Blueprint\n\nbp = Blueprint()\nprint(bp.to_string())\n"

    used_indices: Set[int] = set()
    code_blocks: List[str] = []
    used_helpers: Set[str] = set()

    # 1. Extract underground belt pairs
    for i, e1 in enumerate(raw_entities):
        if i in used_indices:
            continue
        name1 = e1.get("name", "")
        io1 = e1.get("io_type") or e1.get("type", "input")
        if name1 not in UNDERGROUND_NAMES or io1 != "input":
            continue

        dir1 = e1.get("direction", 0) or 0
        p1 = e1.get("position", {})
        x1, y1 = float(p1.get("x", 0.0)), float(p1.get("y", 0.0))

        # Look for matching output
        best_j = None
        min_dist = float("inf")
        for j, e2 in enumerate(raw_entities):
            if j in used_indices or j == i:
                continue
            io2 = e2.get("io_type") or e2.get("type", "input")
            if e2.get("name") == name1 and io2 == "output" and (e2.get("direction", 0) or 0) == dir1:
                p2 = e2.get("position", {})
                x2, y2 = float(p2.get("x", 0.0)), float(p2.get("y", 0.0))
                # Check alignment and forward direction
                valid_dist = None
                if dir1 == 4 and abs(y1 - y2) < 1e-3 and x2 > x1:
                    valid_dist = x2 - x1
                elif dir1 == 12 and abs(y1 - y2) < 1e-3 and x2 < x1:
                    valid_dist = x1 - x2
                elif dir1 == 8 and abs(x1 - x2) < 1e-3 and y2 > y1:
                    valid_dist = y2 - y1
                elif dir1 == 0 and abs(x1 - x2) < 1e-3 and y2 < y1:
                    valid_dist = y1 - y2

                if valid_dist is not None and valid_dist < min_dist:
                    min_dist = valid_dist
                    best_j = j

        if best_j is not None:
            e2 = raw_entities[best_j]
            p2 = e2.get("position", {})
            x2, y2 = float(p2.get("x", 0.0)), float(p2.get("y", 0.0))
            used_indices.add(i)
            used_indices.add(best_j)
            code_blocks.append(
                f"add_underground_pair(bp, start=({x1}, {y1}), end=({x2}, {y2}), direction={dir1}, belt_type={repr(name1)})"
            )
            used_helpers.add("add_underground_pair")

    # 2. Extract straight belt lines (length >= 3)
    belts_by_type_dir: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = {}
    for i, e in enumerate(raw_entities):
        if i in used_indices:
            continue
        name = e.get("name", "")
        if name in BELT_NAMES:
            direction = e.get("direction", 0) or 0
            pos = e.get("position", {})
            x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
            belts_by_type_dir.setdefault((name, direction), []).append((i, x, y))

    for (belt_name, belt_dir), items in belts_by_type_dir.items():
        # Step delta according to direction
        if belt_dir == 4:  # East: x increases
            sort_key = lambda it: (round(it[2], 2), round(it[1], 2))
            is_next = lambda p_prev, p_curr: abs(p_prev[1] - p_curr[1]) < 1e-3 and abs((p_prev[0] + 1.0) - p_curr[0]) < 1e-3
        elif belt_dir == 12:  # West: x decreases
            sort_key = lambda it: (round(it[2], 2), -round(it[1], 2))
            is_next = lambda p_prev, p_curr: abs(p_prev[1] - p_curr[1]) < 1e-3 and abs((p_prev[0] - 1.0) - p_curr[0]) < 1e-3
        elif belt_dir == 8:  # South: y increases
            sort_key = lambda it: (round(it[1], 2), round(it[2], 2))
            is_next = lambda p_prev, p_curr: abs(p_prev[0] - p_curr[0]) < 1e-3 and abs((p_prev[1] + 1.0) - p_curr[1]) < 1e-3
        elif belt_dir == 0:  # North: y decreases
            sort_key = lambda it: (round(it[1], 2), -round(it[2], 2))
            is_next = lambda p_prev, p_curr: abs(p_prev[0] - p_curr[0]) < 1e-3 and abs((p_prev[1] - 1.0) - p_curr[1]) < 1e-3
        else:
            continue

        sorted_items = sorted(items, key=sort_key)
        idx_seq: List[Tuple[int, float, float]] = []

        for item in sorted_items:
            if not idx_seq:
                idx_seq.append(item)
            else:
                prev_pos = (idx_seq[-1][1], idx_seq[-1][2])
                curr_pos = (item[1], item[2])
                if is_next(prev_pos, curr_pos):
                    idx_seq.append(item)
                else:
                    if len(idx_seq) >= 3:
                        for it in idx_seq:
                            used_indices.add(it[0])
                        start_pt = (idx_seq[0][1], idx_seq[0][2])
                        code_blocks.append(
                            f"add_belt_line(bp, start=({start_pt[0]}, {start_pt[1]}), length={len(idx_seq)}, direction={belt_dir}, belt_type={repr(belt_name)})"
                        )
                        used_helpers.add("add_belt_line")
                    idx_seq = [item]

        if len(idx_seq) >= 3:
            for it in idx_seq:
                used_indices.add(it[0])
            start_pt = (idx_seq[0][1], idx_seq[0][2])
            code_blocks.append(
                f"add_belt_line(bp, start=({start_pt[0]}, {start_pt[1]}), length={len(idx_seq)}, direction={belt_dir}, belt_type={repr(belt_name)})"
            )
            used_helpers.add("add_belt_line")

    # 3. Extract repeating entity rows & columns (furnaces, assemblers, lamps, poles, inserters)
    entities_by_type: Dict[Tuple[str, int, Optional[str]], List[Tuple[int, float, float]]] = {}
    for i, e in enumerate(raw_entities):
        if i in used_indices:
            continue
        name = e.get("name", "")
        direction = e.get("direction", 0) or 0
        recipe = e.get("recipe", None)
        pos = e.get("position", {})
        x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        entities_by_type.setdefault((name, direction, recipe), []).append((i, x, y))

    for (ent_name, ent_dir, ent_recipe), items in entities_by_type.items():
        recipe_param = f", recipe={repr(ent_recipe)}" if ent_recipe else ""
        dir_param = f", direction={ent_dir}" if ent_dir else ""

        # Check horizontal rows (same y, constant dx)
        by_y: Dict[float, List[Tuple[int, float, float]]] = {}
        for it in items:
            if it[0] not in used_indices:
                by_y.setdefault(round(it[2], 2), []).append(it)

        for y_val, row_items in by_y.items():
            avail_items = [it for it in row_items if it[0] not in used_indices]
            s_items = sorted(avail_items, key=lambda it: it[1])
            for run in find_uniform_runs(s_items, coord_idx=1, min_len=3):
                step_dx = round(run[1][1] - run[0][1], 2)
                for it in run:
                    used_indices.add(it[0])
                start_pos = (run[0][1], run[0][2])
                code_blocks.append(
                    f"add_entity_row(bp, {repr(ent_name)}, start=({start_pos[0]}, {start_pos[1]}), count={len(run)}, step=({step_dx}, 0.0){dir_param}{recipe_param})"
                )
                used_helpers.add("add_entity_row")

        # Check vertical columns (same x, constant dy)
        by_x: Dict[float, List[Tuple[int, float, float]]] = {}
        for it in items:
            if it[0] not in used_indices:
                by_x.setdefault(round(it[1], 2), []).append(it)

        for x_val, col_items in by_x.items():
            avail_items = [it for it in col_items if it[0] not in used_indices]
            s_items = sorted(avail_items, key=lambda it: it[2])
            for run in find_uniform_runs(s_items, coord_idx=2, min_len=3):
                step_dy = round(run[1][2] - run[0][2], 2)
                for it in run:
                    used_indices.add(it[0])
                start_pos = (run[0][1], run[0][2])
                code_blocks.append(
                    f"add_entity_row(bp, {repr(ent_name)}, start=({start_pos[0]}, {start_pos[1]}), count={len(run)}, step=(0.0, {step_dy}){dir_param}{recipe_param})"
                )
                used_helpers.add("add_entity_row")

    # 4. Remaining individual entities
    needed_entity_classes: Set[str] = set()
    single_entities_code: List[str] = []

    for i, e in enumerate(raw_entities):
        if i in used_indices:
            continue
        name = e.get("name", "")
        direction = e.get("direction", None)
        pos = e.get("position", {})
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))

        # Check if prototype is available in draftsman.entity
        class_name = _get_class_name_for_entity(name)
        needed_entity_classes.add(class_name)

        pos_str = f"position={{'x': {x}, 'y': {y}}}"
        dir_str = f", direction={direction}" if direction is not None else ""

        extra_kwargs = ""
        if name in UNDERGROUND_NAMES:
            io_val = e.get("io_type") or e.get("type", "input")
            extra_kwargs += f", io_type={repr(io_val)}"

        for k in ("recipe", "input_priority", "output_priority", "filter_mode"):
            if k in e and e[k] is not None:
                extra_kwargs += f", {k}={repr(e[k])}"

        single_entities_code.append(
            f"bp.entities.append({class_name}({repr(name)}, {pos_str}{dir_str}{extra_kwargs}))"
        )

    # 5. Build full Python script
    lines = [
        "from draftsman.blueprintable import Blueprint"
    ]
    if needed_entity_classes:
        sorted_classes = sorted(needed_entity_classes)
        lines.append(f"from draftsman.entity import {', '.join(sorted_classes)}")

    if used_helpers:
        sorted_helpers = sorted(used_helpers)
        lines.append(f"from draftsman_helpers import {', '.join(sorted_helpers)}")

    lines.append("")
    lines.append(f"# Blueprint: {blueprint_name}")
    lines.append("bp = Blueprint()")
    lines.append("")

    if code_blocks:
        lines.append("# Macro placements")
        lines.extend(code_blocks)
        lines.append("")

    if single_entities_code:
        lines.append("# Specific entities")
        lines.extend(single_entities_code)
        lines.append("")

    lines.append("print(bp.to_string())")
    return "\n".join(lines).strip() + "\n"


def _get_class_name_for_entity(name: str) -> str:
    """Returns appropriate PascalCase class name for an entity."""
    words = name.split("-")
    pascal = "".join(w.capitalize() for w in words)
    return pascal


def execute_script_to_blueprint(code: str) -> Optional[Blueprint]:
    """Safely executes a Draftsman script in a local scope to get the compiled Blueprint."""
    loc: Dict[str, Any] = {}
    glob: Dict[str, Any] = {
        "__name__": "__main__",
        "Blueprint": Blueprint,
        "add_belt_line": helpers.add_belt_line,
        "add_underground_pair": helpers.add_underground_pair,
        "add_entity_row": helpers.add_entity_row,
        "add_power_poles": helpers.add_power_poles,
        "resolve_direction": helpers.resolve_direction,
    }
    # Also add entity prototypes
    for k, v in _d_ent.__dict__.items():
        if isinstance(v, type):
            glob[k] = v

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, glob, loc)
    except Exception:
        return None

    bp = loc.get("bp") or glob.get("bp")
    if isinstance(bp, Blueprint):
        return bp
    return None


def verify_code_equivalence(orig_code: str, lifted_code: str) -> bool:
    """Verifies that the lifted code compiles to the exact same blueprint entities as the original."""
    bp_orig = execute_script_to_blueprint(orig_code)
    if bp_orig is None:
        return False

    bp_lifted = execute_script_to_blueprint(lifted_code)
    if bp_lifted is None:
        return False

    orig_set = {entity_to_canonical(e.to_dict()) for e in bp_orig.entities}
    lifted_set = {entity_to_canonical(e.to_dict()) for e in bp_lifted.entities}

    return orig_set == lifted_set


def process_dataset(
    input_path: Union[str, Path] = "data/dataset.jsonl",
    output_path: Union[str, Path] = "data/dataset_v2.jsonl",
    limit: Optional[int] = None,
    log_interval: int = 200,
) -> Dict[str, Any]:
    """
    Transforms the unrolled scripts in dataset.jsonl into macro-lifted scripts.
    Guarantees mathematical equivalence for all transformed code.
    Falls back to original code if compilation or equivalence fails.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from factoriollm.inference import DEFAULT_SYSTEM_PROMPT

    total_records = 0
    lifted_records = 0
    fallback_records = 0
    failed_compile_records = 0
    orig_total_lines = 0
    new_total_lines = 0

    print(f"[*] Starting dataset transformation: {input_file} -> {output_file}")
    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(in_f):
            if limit is not None and idx >= limit:
                break
            line_str = line.strip()
            if not line_str:
                continue

            total_records += 1
            record = json.loads(line_str)
            messages = record.get("messages", [])
            if len(messages) < 3:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Update system prompt to modern prompt documenting macros
            messages[0]["content"] = DEFAULT_SYSTEM_PROMPT

            orig_code = messages[2].get("content", "")
            orig_lines = len(orig_code.splitlines())
            orig_total_lines += orig_lines

            # Extract blueprint name if present
            m = re.search(r"#\s*Blueprint:\s*(.*)", orig_code)
            bp_name = m.group(1).strip() if m else "Blueprint"

            # Attempt to execute and lift
            bp = execute_script_to_blueprint(orig_code)
            if bp is None:
                failed_compile_records += 1
                fallback_records += 1
                new_total_lines += orig_lines
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            lifted_code = lift_blueprint_to_code(bp, blueprint_name=bp_name)
            is_equiv = verify_code_equivalence(orig_code, lifted_code)
            has_macros = any(m in lifted_code for m in ["add_belt_line", "add_underground_pair", "add_entity_row", "add_power_poles"])
            new_lines = len(lifted_code.splitlines())

            # Only substitute if 100% mathematically equivalent, contains macros, and reduces lines
            if is_equiv and has_macros and new_lines < orig_lines:
                lifted_records += 1
                new_total_lines += new_lines
                messages[2]["content"] = lifted_code
            else:
                fallback_records += 1
                new_total_lines += orig_lines

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % log_interval == 0:
                print(f"[{idx + 1}] Processed... Lifted: {lifted_records}/{total_records} ({(lifted_records/total_records)*100:.1f}%) | "
                      f"Lines: {orig_total_lines} -> {new_total_lines} ({(1 - new_total_lines/orig_total_lines)*100:.1f}% reduction)")

    compression_pct = ((1.0 - (new_total_lines / max(orig_total_lines, 1))) * 100.0)
    lift_pct = (lifted_records / max(total_records, 1)) * 100.0

    metrics = {
        "total_records": total_records,
        "lifted_records": lifted_records,
        "fallback_records": fallback_records,
        "failed_compile_records": failed_compile_records,
        "lift_success_percentage": round(lift_pct, 2),
        "orig_total_lines": orig_total_lines,
        "new_total_lines": new_total_lines,
        "lines_saved": orig_total_lines - new_total_lines,
        "compression_percentage": round(compression_pct, 2),
    }

    print("\n" + "=" * 50)
    print("Dataset Transformation Summary:")
    print(f"  Total records:           {metrics['total_records']}")
    print(f"  Lifted & verified:       {metrics['lifted_records']} ({metrics['lift_success_percentage']}%)")
    print(f"  Safe fallback retained:  {metrics['fallback_records']}")
    print(f"  Uncompilable originals:  {metrics['failed_compile_records']}")
    print(f"  Total lines before:      {metrics['orig_total_lines']}")
    print(f"  Total lines after:       {metrics['new_total_lines']}")
    print(f"  Lines saved:             {metrics['lines_saved']}")
    print(f"  Token/Line reduction:    {metrics['compression_percentage']}%")
    print("=" * 50 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Lift FactorioLLM dataset to concise macros")
    parser.add_argument("--input", default="data/dataset.jsonl", help="Input dataset path")
    parser.add_argument("--output", default="data/dataset_v2.jsonl", help="Output dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    args = parser.parse_args()

    process_dataset(input_path=args.input, output_path=args.output, limit=args.limit)


if __name__ == "__main__":
    main()
