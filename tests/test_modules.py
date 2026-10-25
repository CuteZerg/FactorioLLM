import importlib.util
import itertools
import math
import unittest
import warnings
from collections import Counter
from pathlib import Path

from draftsman.blueprintable import Blueprint
from draftsman.data import entities as entity_data

from factoriollm.blueprint_ir import parse_blueprint_string
from factoriollm.helpers import (
    _BALANCER_TEMPLATES,
    BBox,
    add_belt_line,
    add_entity_row,
    assembler_row,
    balancer,
    lab_array,
    mining_field,
    place_below,
    place_right_of,
    resolve_direction,
    smelting_column,
    solar_field,
    steam_power,
)
from factoriollm.validator import validate_string

DIRECTIONS = ("north", "east", "south", "west")


def entity_tiles(e):
    return [(e.tile_position.x + dx, e.tile_position.y + dy) for dx in range(e.tile_width) for dy in range(e.tile_height)]


def pole_networks(bp):
    """Number of wire-connected pole networks (poles within maximum_wire_distance of each other)."""
    poles = [e for e in bp.entities if entity_data.raw[e.name].get("type") == "electric-pole"]
    if not poles:
        return 0
    centers = [(e.tile_position.x + e.tile_width / 2, e.tile_position.y + e.tile_height / 2) for e in poles]
    reach = [float(entity_data.raw[e.name]["maximum_wire_distance"]) for e in poles]
    seen, networks = set(), 0
    for start in range(len(poles)):
        if start in seen:
            continue
        networks += 1
        stack = [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            for j in range(len(poles)):
                if j not in seen and math.dist(centers[i], centers[j]) <= min(reach[i], reach[j]) + 1e-9:
                    seen.add(j)
                    stack.append(j)
    return networks


class ModuleTestCase(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def assert_clean(self, bp, box, label=""):
        """No validator issues at all, one pole network, and the bbox covers exactly the entities."""
        report = validate_string(bp.to_string())
        self.assertEqual(report.errors, [], label)
        self.assertEqual(report.warnings, [], label)
        self.assertLessEqual(pole_networks(bp), 1, label)
        tiles = [t for e in bp.entities for t in entity_tiles(e)]
        self.assertEqual(min(t[0] for t in tiles), box.x, label)
        self.assertEqual(min(t[1] for t in tiles), box.y, label)
        self.assertLessEqual(max(t[0] for t in tiles), box.right - 1, label)
        self.assertLessEqual(max(t[1] for t in tiles), box.bottom - 1, label)

    def counts(self, bp):
        return Counter(e.name for e in bp.entities)


class TestProductionModules(ModuleTestCase):
    def test_smelting_column_grid(self):
        for furnace, count, rows, direction, pole in itertools.product(
                ("stone-furnace", "steel-furnace", "electric-furnace"), (1, 7, 24), (1, 2), DIRECTIONS,
                ("small-electric-pole", "medium-electric-pole")):
            label = f"{furnace} x{count} rows={rows} {direction} {pole}"
            bp = Blueprint()
            box = smelting_column(bp, (3, -2), count=count, furnace=furnace, rows=rows, direction=direction, pole=pole)
            self.assert_clean(bp, box, label)
            c = self.counts(bp)
            self.assertEqual(c[furnace], count, label)
            self.assertEqual(c["inserter"], 2 * count, label)

    def test_smelting_column_shape(self):
        bp = Blueprint()
        box = smelting_column(bp, count=24, rows=2, direction="south")
        self.assertEqual((box.width, box.height), (11, 24))  # column: belts run north-south

    def test_burner_inserters_need_no_poles(self):
        bp = Blueprint()
        box = smelting_column(bp, count=8, inserter="burner-inserter", pole=None)
        self.assert_clean(bp, box)
        self.assertFalse(any("pole" in e.name for e in bp.entities))

    def test_assembler_row_grid(self):
        for machine, count, inputs, rows, direction in itertools.product(
                ("assembling-machine-1", "assembling-machine-2", "assembling-machine-3"), (1, 6, 11), (1, 2), (1, 2), DIRECTIONS):
            label = f"{machine} x{count} inputs={inputs} rows={rows} {direction}"
            bp = Blueprint()
            box = assembler_row(bp, count=count, recipe="electronic-circuit", machine=machine, inputs=inputs,
                                rows=rows, direction=direction)
            self.assert_clean(bp, box, label)
            c = self.counts(bp)
            self.assertEqual(c[machine], count, label)
            self.assertEqual(c["long-handed-inserter"], count if inputs == 2 else 0, label)
            self.assertEqual({e.recipe for e in bp.entities if e.name == machine}, {"electronic-circuit"})

    def test_assembler_row_rejects_bad_recipes(self):
        with self.assertRaisesRegex(ValueError, "electronic-circuit"):
            assembler_row(Blueprint(), recipe="electronic-circuits")
        with self.assertRaisesRegex(ValueError, "fluids"):
            assembler_row(Blueprint(), recipe="plastic-bar")

    def test_lab_array_grid(self):
        for count, inputs, rows, direction in itertools.product((1, 10, 16), (1, 2), (1, 2), DIRECTIONS):
            label = f"labs x{count} inputs={inputs} rows={rows} {direction}"
            bp = Blueprint()
            box = lab_array(bp, count=count, inputs=inputs, rows=rows, direction=direction)
            self.assert_clean(bp, box, label)
            self.assertEqual(self.counts(bp)["lab"], count, label)


class TestFieldModules(ModuleTestCase):
    def test_mining_field_grid(self):
        for count, rows, direction, pole in itertools.product(
                (1, 2, 7, 16), (1, 2), DIRECTIONS, ("small-electric-pole", "medium-electric-pole")):
            label = f"drills x{count} rows={rows} {direction} {pole}"
            bp = Blueprint()
            box = mining_field(bp, count=count, rows=rows, direction=direction, pole=pole)
            self.assert_clean(bp, box, label)
            self.assertEqual(self.counts(bp)["electric-mining-drill"], count, label)

    def test_solar_field_exact_counts(self):
        for panels, accumulators in ((84, None), (104, 88), (1, 0), (0, 5), (300, 252)):
            bp = Blueprint()
            box = solar_field(bp, (5, 5), panels=panels, accumulators=accumulators)
            self.assert_clean(bp, box, f"{panels}/{accumulators}")
            c = self.counts(bp)
            self.assertEqual(c["solar-panel"], panels)
            self.assertEqual(c["accumulator"], round(panels * 0.84) if accumulators is None else accumulators)

    def test_steam_power_grid(self):
        for boilers, engines, direction in itertools.product((1, 2, 5, 20), (1, 2), DIRECTIONS):
            label = f"boilers x{boilers} engines={engines} {direction}"
            bp = Blueprint()
            box = steam_power(bp, boilers=boilers, engines_per_boiler=engines, direction=direction)
            self.assert_clean(bp, box, label)
            c = self.counts(bp)
            self.assertEqual((c["boiler"], c["steam-engine"], c["offshore-pump"]), (boilers, boilers * engines, 1), label)

    def test_steam_power_matches_real_blueprint(self):
        # "Bored Bobs Power Book": pump, boiler, two engines and a burner inserter on the fuel side
        bp = Blueprint()
        steam_power(bp, boilers=1, belt="transport-belt", pole=None)
        got = {(e.name, e.tile_position.x, e.tile_position.y, int(e.direction)) for e in bp.entities if e.name != "transport-belt"}
        self.assertEqual(got, {("steam-engine", 1, 0, 0), ("steam-engine", 1, 5, 0), ("boiler", 1, 10, 0),
                               ("offshore-pump", 0, 11, 12), ("burner-inserter", 2, 12, 8)})

    def test_steam_power_limits_boilers_per_pump(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            steam_power(Blueprint(), boilers=21)


def load_balancer_tool():
    spec = importlib.util.spec_from_file_location("extract_balancers", Path(__file__).resolve().parent.parent / "tools" / "extract_balancers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBalancer(ModuleTestCase):
    def test_every_template_balances_in_every_direction(self):
        tool = load_balancer_tool()
        belts = ("transport-belt", "fast-transport-belt", "express-transport-belt")
        for (n_in, n_out, tier), direction in itertools.product(_BALANCER_TEMPLATES, DIRECTIONS):
            label = f"{n_in}->{n_out} tier {tier} {direction}"
            bp = Blueprint()
            box = balancer(bp, (2, 3), inputs=n_in, outputs=n_out, belt=belts[tier - 1], direction=direction)
            self.assert_clean(bp, box, label)
            ir = parse_blueprint_string(bp.to_string(), allowed=None)
            info = tool.Template([(e.name, e.x, e.y, e.direction, e.extras_dict) for e in ir.entities]).analyse()
            self.assertIsNotNone(info, label)
            self.assertEqual((len(info["inputs"]), len(info["outputs"]), info["flow"]),
                             (n_in, n_out, resolve_direction(direction)), label)

    def test_belt_tier_is_applied(self):
        bp = Blueprint()
        balancer(bp, inputs=4, outputs=4, belt="fast-transport-belt")
        self.assertEqual({e.name.split("-")[0] for e in bp.entities}, {"fast"})

    def test_unknown_combination_lists_available(self):
        with self.assertRaisesRegex(ValueError, "available: .*4->4"):
            balancer(Blueprint(), inputs=7, outputs=3)


class TestPlacement(ModuleTestCase):
    def test_modules_compose_without_overlap(self):
        bp = Blueprint()
        mines = mining_field(bp, count=8)
        furnaces = smelting_column(bp, place_right_of(mines), count=12, direction="east", rows=1)
        labs = lab_array(bp, place_below(furnaces, gap=2), count=6)
        self.assertIsInstance(labs, BBox)
        report = validate_string(bp.to_string())
        self.assertEqual([i for i in report.errors], [])
        self.assertLessEqual(pole_networks(bp), 3)

    def test_overlap_is_reported_with_tiles(self):
        bp = Blueprint()
        add_entity_row(bp, "stone-furnace", start=(0, 0), count=2, step=(2, 0))
        with self.assertRaisesRegex(ValueError, r"inserter at \(0, 1\): tile \(0, 1\) is already occupied by stone-furnace at \(0, 0\)"):
            add_entity_row(bp, "inserter", start=(0, 1), count=2, step=(1, 0))

    def test_failed_module_leaves_blueprint_untouched(self):
        bp = Blueprint()
        add_belt_line(bp, start=(5, 5), length=3)
        with self.assertRaises(ValueError):
            smelting_column(bp, (0, 0), count=12)
        self.assertEqual(len(bp.entities), 3)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "count"):
            smelting_column(Blueprint(), count=0)
        with self.assertRaisesRegex(ValueError, "rows"):
            lab_array(Blueprint(), rows=3)
        with self.assertRaisesRegex(ValueError, "Unknown belt"):
            mining_field(Blueprint(), belt="red-belt")
        with self.assertRaisesRegex(ValueError, "furnace"):
            smelting_column(Blueprint(), furnace="assembling-machine-2")


if __name__ == "__main__":
    unittest.main()
