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
    add_power_poles,
    connect_power,
    green_circuits,
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

    def assert_clean(self, bp, box, label="", allowed_warnings=frozenset()):
        """No validator issues at all, one pole network, and the bbox covers exactly the entities."""
        report = validate_string(bp.to_string())
        self.assertEqual(report.errors, [], label)
        self.assertEqual([w for w in report.warnings if w.category not in allowed_warnings], [], label)
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
        box = smelting_column(bp, count=24, rows=2, direction="south", feed="lanes")
        self.assertEqual((box.width, box.height), (11, 24))  # column: belts run north-south
        bp = Blueprint()
        box = smelting_column(bp, count=24, rows=2, direction="south")  # + ore/coal junction
        self.assertEqual((box.width, box.height), (13, 30))

    def test_coal_furnaces_take_input_outside_and_output_in_the_middle(self):
        bp = Blueprint()
        smelting_column(bp, count=12, direction="east", feed="lanes")
        rows = {}
        for e in bp.entities:
            rows.setdefault(e.tile_position.y, set()).add(e.name)
        belt_rows = sorted(y for y, names in rows.items() if names == {"transport-belt"})
        self.assertEqual(belt_rows, [0, 5, 10])  # two outer inputs, one shared output
        drops = [e for e in bp.entities if e.name == "inserter" and e.tile_position.y in (4, 6)]
        self.assertEqual(len(drops), 12)  # every furnace outputs onto the middle belt

    def test_inserters_of_a_furnace_share_one_column(self):
        for furnace in ("stone-furnace", "electric-furnace"):
            bp = Blueprint()
            smelting_column(bp, count=8, furnace=furnace, direction="east", feed="lanes")
            columns = Counter(e.tile_position.x for e in bp.entities if e.name == "inserter")
            self.assertEqual(set(columns.values()), {4}, furnace)  # 2 furnaces x (in + out) per column

    def test_fuel_junction_puts_ore_and_coal_on_separate_lanes(self):
        bp = Blueprint()
        smelting_column(bp, count=12, direction="east")
        transport = [e for e in bp.entities if "belt" in e.name or "splitter" in e.name]
        at = {t: e for e in transport for t in entity_tiles(e)}
        vec = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}

        def source(tile):
            """Follows belts upstream to the west edge; returns the row where the item enters."""
            for _ in range(60):
                if tile[0] == 0:
                    return tile[1]
                here = at[tile]
                if here.name == "underground-belt" and here.io_type == "output":  # jump to the entrance
                    tile = next(t for t, e in at.items() if e.name == here.name and e.io_type == "input"
                                and t[0] == tile[0] and t[1] < tile[1])
                    continue
                feeders = [t for e in transport for t in entity_tiles(e)
                           if not (e.name == "underground-belt" and e.io_type == "input")
                           and (t[0] + vec[int(e.direction)][0], t[1] + vec[int(e.direction)][1]) == tile]
                if here.name == "splitter":  # a splitter mixes both halves: follow the fed one
                    feeders += [t for half in entity_tiles(here) for t, e in at.items()
                                if (t[0] + vec[int(e.direction)][0], t[1] + vec[int(e.direction)][1]) == half]
                tile = feeders[0]
            raise AssertionError("no source found")

        # input belt starts (T-junctions) are fed from north and south; one side ore, the other coal
        for start in ((5, 1), (5, 11)):
            north, south = (start[0], start[1] - 1), (start[0], start[1] + 1)
            self.assertEqual({source(north), source(south)}, {0, 6}, start)  # coal row 0, ore row 6

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


    def test_lab_chain_grid(self):
        for count, rows, depth, inputs, direction in itertools.product(
                (3, 11, 16, 30), (1, 2), (1, 2, 4), (1, 2), DIRECTIONS):
            label = f"labs x{count} rows={rows} depth={depth} inputs={inputs} {direction}"
            bp = Blueprint()
            box = lab_array(bp, count=count, rows=rows, layout="chain", chain_depth=depth, inputs=inputs,
                            direction=direction)
            self.assert_clean(bp, box, label)
            self.assertEqual(self.counts(bp)["lab"], count, label)

    def test_lab_chain_feeds_each_lab_once(self):
        bp = Blueprint()
        lab_array(bp, count=12, rows=1, layout="chain", chain_depth=3, direction="east")
        inserters = [e for e in bp.entities if e.name == "inserter"]
        self.assertEqual(len(inserters), 12)  # 4 from the belt + 8 between labs
        self.assertEqual(sum(e.tile_position.y == 1 for e in inserters), 4)

    def test_large_research_blocks_chain_by_default(self):
        small, large = Blueprint(), Blueprint()
        lab_array(small, count=10)
        lab_array(large, count=24)
        self.assertEqual(sum(e.name == "transport-belt" for e in small.entities), 15)  # 5 labs per side
        self.assertEqual(sum(e.name == "transport-belt" for e in large.entities), 11)  # 3 columns of 2 x 4 labs

    def test_green_circuits_ratio(self):
        for count, machine, direction in itertools.product((1, 2, 4, 5), ("assembling-machine-1", "assembling-machine-2"),
                                                          DIRECTIONS):
            label = f"circuits x{count} {machine} {direction}"
            bp = Blueprint()
            box = green_circuits(bp, count=count, machine=machine, direction=direction)
            self.assert_clean(bp, box, label)
            recipes = Counter(e.recipe for e in bp.entities if e.name == machine)
            self.assertEqual(recipes["electronic-circuit"], count, label)
            self.assertEqual(recipes["copper-cable"], count + (count + 1) // 2, label)  # 3 per 2 circuits
            link = "fast-inserter" if machine == "assembling-machine-1" else "bulk-inserter"
            self.assertEqual(self.counts(bp)[link], 2 * count, label)


class TestFieldModules(ModuleTestCase):
    def test_mining_field_grid(self):
        for count, rows, direction, pole, layout, belts in itertools.product(
                (2, 7, 16, 30), (1, 2), DIRECTIONS, ("small-electric-pole", "medium-electric-pole"),
                ("pole_gaps", "belt_poles", "pole_row"), (1, 2)):
            if count < belts * rows * 2:
                continue  # small poles cannot wire pole rows 8 tiles apart
            label = f"drills x{count} rows={rows} {direction} {pole} {layout} belts={belts}"
            bp = Blueprint()
            box = mining_field(bp, count=count, rows=rows, direction=direction, pole=pole, layout=layout, belts=belts)
            self.assert_clean(bp, box, label)
            self.assertEqual(self.counts(bp)["electric-mining-drill"], count, label)

    def test_belt_poles_keep_the_belt_continuous(self):
        bp = Blueprint()
        mining_field(bp, count=12, layout="belt_poles", direction="east")
        line = sorted((e.tile_position.x, e.name, getattr(e, "io_type", None)) for e in bp.entities if e.tile_position.y == 3)
        self.assertEqual([x for x, _, _ in line], list(range(18)))  # belt, undergrounds or pole on every tile
        poles = [x for x, name, _ in line if "pole" in name]
        for x in poles:
            self.assertEqual(line[x - 1][2], "input")
            self.assertEqual(line[x + 1][2], "output")

    def test_single_drill_rows_refuse_belt_poles(self):
        with self.assertRaisesRegex(ValueError, "pole_gaps"):
            mining_field(Blueprint(), count=1, rows=1, layout="belt_poles")

    def test_solar_field_exact_counts(self):
        for (panels, accumulators), pole in itertools.product(
                ((84, None), (104, 88), (1, 0), (0, 5), (300, 252), (7, 3)), ("substation", "medium-electric-pole")):
            label = f"{panels}/{accumulators} {pole}"
            bp = Blueprint()
            box = solar_field(bp, (5, 5), panels=panels, accumulators=accumulators, pole=pole)
            self.assert_clean(bp, box, label)
            c = self.counts(bp)
            self.assertEqual(c["solar-panel"], panels, label)
            self.assertEqual(c["accumulator"], round(panels * 0.84) if accumulators is None else accumulators, label)

    def test_solar_cells_have_no_gaps(self):
        for panels, accumulators, cells in ((28, 17, 1), (24, 26, 1), (52, 43, 2), (32, 8, 1)):
            bp = Blueprint()
            box = solar_field(bp, panels=panels, accumulators=accumulators)
            tiles = sum(e.tile_width * e.tile_height for e in bp.entities)
            self.assertEqual(tiles, 324 * cells, (panels, accumulators))
            self.assertEqual(box.width * box.height, 324 * cells)

    def test_steam_power_grid(self):
        for boilers, engines, direction, layout, side, flow in itertools.product(
                (1, 2, 5, 20), (1, 2), DIRECTIONS, ("gapless_row", "gapped_row", "mirrored_pair"),
                ("left", "right"), ("forward", "backward")):
            if layout == "mirrored_pair" and boilers < 2:
                continue
            label = f"boilers x{boilers} engines={engines} {direction} {layout} {side} {flow}"
            bp = Blueprint()
            box = steam_power(bp, boilers=boilers, engines_per_boiler=engines, direction=direction, layout=layout,
                              engines_side=side, fuel_flow=flow)
            self.assert_clean(bp, box, label)
            c = self.counts(bp)
            pumps = 2 if layout == "mirrored_pair" else 1
            self.assertEqual((c["boiler"], c["steam-engine"], c["offshore-pump"]), (boilers, boilers * engines, pumps), label)

    def test_gapless_row_matches_the_player_layout(self):
        # boilers side by side, a pipe row between the two engines of each stack, poles in that row
        bp = Blueprint()
        steam_power(bp, boilers=5, direction="east")
        pipes = sorted((e.tile_position.x, e.tile_position.y) for e in bp.entities if e.name == "pipe")
        self.assertEqual(pipes, [(x, 5) for x in (2, 5, 8, 11, 14)])
        self.assertEqual({e.tile_position.y for e in bp.entities if "pole" in e.name}, {5})
        boilers = sorted(e.tile_position.x for e in bp.entities if e.name == "boiler")
        self.assertEqual(boilers, [1, 4, 7, 10, 13])

    def test_steam_power_matches_real_blueprint(self):
        # "Bored Bobs Power Book": pump, boiler, two engines and a burner inserter on the fuel side
        bp = Blueprint()
        steam_power(bp, boilers=1, belt="transport-belt", pole=None, layout="gapped_row")
        got = {(e.name, e.tile_position.x, e.tile_position.y, int(e.direction)) for e in bp.entities if e.name != "transport-belt"}
        self.assertEqual(got, {("steam-engine", 1, 0, 0), ("steam-engine", 1, 5, 0), ("boiler", 1, 10, 0),
                               ("offshore-pump", 0, 11, 12), ("burner-inserter", 2, 12, 8)})

    def test_steam_power_limits_boilers(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            steam_power(Blueprint(), boilers=101)


def load_balancer_tool():
    spec = importlib.util.spec_from_file_location("extract_balancers", Path(__file__).resolve().parent.parent / "tools" / "extract_balancers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBalancer(ModuleTestCase):
    def test_every_template_balances_in_every_direction(self):
        tool = load_balancer_tool()
        belts = ("transport-belt", "fast-transport-belt", "express-transport-belt")
        for (n_in, n_out, tier, tu), direction in itertools.product(_BALANCER_TEMPLATES, DIRECTIONS):
            label = f"{n_in}->{n_out} tier {tier} tu={tu} {direction}"
            bp = Blueprint()
            box = balancer(bp, (2, 3), inputs=n_in, outputs=n_out, belt=belts[tier - 1], direction=direction,
                           throughput_unlimited=tu)
            # some book designs block a splitter side with an underground entrance on purpose
            self.assert_clean(bp, box, label, allowed_warnings={"belt_head_on"})
            ir = parse_blueprint_string(bp.to_string(), allowed=None)
            info = tool.Template([(e.name, e.x, e.y, e.direction, e.extras_dict) for e in ir.entities]).analyse()
            self.assertIsNotNone(info, label)
            self.assertEqual((len(info["inputs"]), len(info["outputs"]), info["flow"]),
                             (n_in, n_out, resolve_direction(direction)), label)

    def test_standard_designs_are_the_defaults(self):
        # Raynquist's 4-4 TU: 6 splitters; the 8-8 for red belts is not the yellow downgrade
        bp = Blueprint()
        balancer(bp, inputs=4, outputs=4)
        self.assertEqual(sum("splitter" in e.name for e in bp.entities), 6)
        red, yellow = Blueprint(), Blueprint()
        balancer(red, inputs=8, outputs=8, belt="fast-transport-belt")
        balancer(yellow, inputs=8, outputs=8)
        self.assertLess(len(red.entities), len(yellow.entities))

    def test_belt_tier_is_applied(self):
        bp = Blueprint()
        balancer(bp, inputs=4, outputs=4, belt="fast-transport-belt")
        self.assertEqual({e.name.split("-")[0] for e in bp.entities}, {"fast"})

    def test_unknown_combination_lists_available(self):
        with self.assertRaisesRegex(ValueError, "available: .*4->4"):
            balancer(Blueprint(), inputs=7, outputs=3)


class TestWiring(ModuleTestCase):
    def wires(self, bp):
        exported = bp.to_dict()["blueprint"]
        ents = exported["entities"]
        return ents, [(a, b) for a, ca, b, cb in exported.get("wires", []) if ca == 5 and cb == 5]

    def test_modules_get_a_spanning_tree(self):
        for build in (lambda bp: smelting_column(bp, count=24), lambda bp: solar_field(bp, panels=300),
                      lambda bp: mining_field(bp, count=16, pole="small-electric-pole"),
                      lambda bp: assembler_row(bp, count=10, recipe="electronic-circuit", inputs=2, rows=2)):
            bp = Blueprint()
            build(bp)
            n = connect_power(bp)
            ents, wires = self.wires(bp)
            poles = [i + 1 for i, e in enumerate(ents) if entity_data.raw[e["name"]]["type"] == "electric-pole"]
            self.assertEqual(n, len(poles) - 1)
            self.assertEqual(len(wires), len(poles) - 1)
            degree = Counter(x for w in wires for x in w)
            self.assertLessEqual(max(degree.values()), 5)
            for a, b in wires:
                pa, pb = ents[a - 1]["position"], ents[b - 1]["position"]
                reach = min(entity_data.raw[ents[k - 1]["name"]]["maximum_wire_distance"] for k in (a, b))
                self.assertLessEqual(math.dist((pa["x"], pa["y"]), (pb["x"], pb["y"])), reach + 1e-9)

    def test_rewiring_replaces_old_wires(self):
        bp = Blueprint()
        add_power_poles(bp, start=(0, 0), count=4, step=(5, 0))
        bp.add_power_connection(bp.entities[0], bp.entities[1])
        connect_power(bp)
        connect_power(bp)
        self.assertEqual(len(self.wires(bp)[1]), 3)

    def test_out_of_reach_poles_are_reported(self):
        bp = Blueprint()
        add_power_poles(bp, start=(0, 0), count=2, step=(20, 0))
        self.assertEqual(connect_power(bp), 0)
        self.assertIn("pole_islands", validate_string(bp.to_string()).category_counts())


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
