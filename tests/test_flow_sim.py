import json
import unittest
from pathlib import Path

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.flow_sim import Layout, check, input_ports, solve

DATA = Path(__file__).parent / "fixtures" / "flow"


def E(name, x, y, d=0, **extras):
    return IREntity(name, x, y, d, tuple(sorted((k, str(v)) for k, v in extras.items())))


def belts(x0, x1, y, d=4, name="transport-belt"):
    return [E(name, x, y, d) for x in range(x0, x1 + 1)]


def fixture(name):
    data = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    return IRBlueprint([IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]])


def gear_line(extra=()):
    """Input belt (row 0) -> inserter -> gear assembler -> inserter -> output belt (row 6)."""
    return IRBlueprint(belts(0, 3, 0) + [E("inserter", 1, 1, 0), E("assembling-machine-1", 0, 2, recipe="iron-gear-wheel"),
                                         E("inserter", 1, 5, 0)] + belts(0, 3, 6) + list(extra))


class FlowSimTest(unittest.TestCase):
    def test_simple_line_works(self):
        r = solve(gear_line())
        self.assertTrue(r.ok, r.problems)
        self.assertEqual({v for v in r.inputs.values() if v}, {"iron-plate"})
        out = [p for p in r.ports if p.role == "output"]
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].items, frozenset({"iron-gear-wheel"}))
        self.assertEqual(out[0].lane, 1)  # dropped from the north onto an eastward belt: the far (right) lane

    def test_starved_and_dangling(self):
        ir = IRBlueprint([E("inserter", 1, 1, 0), E("assembling-machine-1", 0, 2, recipe="iron-gear-wheel")])
        r = solve(ir)
        self.assertFalse(r.ok)
        self.assertEqual({p.kind for p in r.problems}, {"starved", "dangling_inserter"})

    def test_side_load_lane(self):
        # a belt going east side-loads onto a belt going north from its left (west) side: left lane
        ir = IRBlueprint(belts(0, 1, 2) + [E("transport-belt", 2, y, 0) for y in range(0, 4)])
        L = Layout(ir)
        r = check(ir, {("lane", 0, 0): "coal", ("lane", 0, 1): "coal"}, L)
        north = next(i for i, e in enumerate(L.ents) if (e.x, e.y) == (2, 0))
        self.assertEqual(r.lanes[(north, 0)], frozenset({"coal"}))
        self.assertEqual(r.lanes[(north, 1)], frozenset())

    def test_underground_side_load_one_lane(self):
        # side-loading into an underground entrance: only the feeding lane on its open (back) half passes
        ir = IRBlueprint([E("transport-belt", 2, 1, 0), E("underground-belt", 2, 0, 12), E("underground-belt", 0, 0, 12, io_type="output")])
        L = Layout(ir)
        r = check(ir, {("lane", 0, 0): "coal", ("lane", 0, 1): "iron-ore"}, L)
        exit_ = next(i for i, e in enumerate(L.ents) if e.extras_dict.get("io_type") == "output")
        self.assertEqual(r.lanes[(exit_, 0)] | r.lanes[(exit_, 1)], frozenset({"iron-ore"}))

    def test_splitter_filter(self):
        ir = IRBlueprint(belts(0, 0, 0) + [E("splitter", 1, 0, 4, filter="coal", output_priority="left")]
                         + belts(2, 3, 0) + belts(2, 3, 1))
        L = Layout(ir)
        r = check(ir, {("lane", 0, 0): "coal", ("lane", 0, 1): "iron-ore"}, L)
        top = next(i for i, e in enumerate(L.ents) if (e.x, e.y) == (3, 0))
        bottom = next(i for i, e in enumerate(L.ents) if (e.x, e.y) == (3, 1))
        self.assertEqual(r.lanes[(top, 0)] | r.lanes[(top, 1)], frozenset({"coal"}))  # left of an eastward splitter is north
        self.assertEqual(r.lanes[(bottom, 0)] | r.lanes[(bottom, 1)], frozenset({"iron-ore"}))

    def test_inserter_chain_over_ground(self):
        # gear machine -> inserter drops on the ground -> next inserter picks it up into a chest
        # (inserter direction 12: picks up from the west, drops to the east)
        ir = gear_line([E("inserter", 3, 3, 12), E("inserter", 5, 3, 12), E("wooden-chest", 6, 3)])
        r = solve(ir)
        self.assertTrue(r.ok, r.problems)
        lone = gear_line([E("inserter", 3, 3, 12)])  # drops on the ground, nobody picks it up
        self.assertIn("dangling_inserter", {p.kind for p in solve(lone).problems})

    def test_clog_on_shared_lane(self):
        # a second input side-loads copper onto the iron line before the gear machine, which never takes it
        ir = gear_line([E("transport-belt", -1, 0, 4), E("transport-belt", 0, -1, 8), E("transport-belt", 0, -2, 8)])
        L = Layout(ir)
        ports = input_ports(L)
        inputs = {p: ("copper-plate" if L.ents[p[1]].y < 0 else "iron-plate" if L.ents[p[1]].y == 0 else None)
                  for p in ports}
        r = check(ir, inputs, L)
        self.assertIn("clog", {p.kind for p in r.problems})

    def test_fluid_mix(self):
        # one pipe joins the offshore pump's water and the pumpjack's crude oil
        ir = IRBlueprint([E("offshore-pump", 0, 2, 0), E("pipe", 0, 3), E("pipe", 1, 3), E("pipe", 2, 3),
                          E("pumpjack", 0, 4, 0)])
        r = check(ir, {}, Layout(ir))
        self.assertIn("fluid_mix", {p.kind for p in r.problems})


class FlowSimRealDesignsTest(unittest.TestCase):
    """Player designs from the second in-game review of the template book."""

    def test_broken_gear_factory(self):
        r = solve(fixture("gear_factory_mk1_broken"))
        self.assertFalse(r.ok)
        self.assertIn("starved", {p.kind for p in r.problems})

    def test_cable_passed_over_the_ground(self):
        self.assertTrue(solve(fixture("red_circuits_ground_chain")).ok)

    def test_mixed_belt_input_and_output_lane(self):
        r = solve(fixture("processing_units_mixed_belt"))
        self.assertTrue(r.ok, r.problems)
        self.assertIn("electronic-circuit", set(r.inputs.values()))

    def test_red_and_green_come_apart(self):
        # the author splits red and green over lanes; whole belts work too and are preferred
        r = solve(fixture("processing_units_lanes"))
        self.assertTrue(r.ok, r.problems)
        by_belt = {}
        for key, item in r.inputs.items():
            if key[0] == "lane" and item:
                by_belt.setdefault(key[1], set()).add(item)
        self.assertEqual(set().union(*by_belt.values()), {"electronic-circuit", "advanced-circuit"})
        self.assertTrue(all(len(items) == 1 for items in by_belt.values()))

    def test_blue_science_without_coal(self):
        r = solve(fixture("blue_science_no_coal"))
        self.assertFalse(r.ok)
        self.assertTrue(any("coal" in p.message for p in r.problems if p.kind == "starved"))

    def test_ore_to_steel(self):
        r = solve(fixture("ore_to_steel_column"))
        self.assertTrue(r.ok, r.problems[:3])
        self.assertIn("steel-plate", {x for p in r.ports if p.role == "output" for x in p.items})


if __name__ == "__main__":
    unittest.main()
