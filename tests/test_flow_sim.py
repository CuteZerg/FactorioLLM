import json
import unittest
from pathlib import Path

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.flow_sim import Layout, _fluid_roles, check, input_ports, solve

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


class FlowSimReviewTest(unittest.TestCase):
    """Mistakes found in the review of step 2d."""

    def test_filter_inserter_takes_only_its_item(self):
        copper = gear_line([])
        line = list(copper.entities)
        line[4] = E("inserter", 1, 1, 0, filters="copper-plate")  # feeds the gear machine copper only
        self.assertIn("dangling_inserter", {p.kind for p in solve(IRBlueprint(line)).problems})
        line[4] = E("inserter", 1, 1, 0, filters="iron-plate")
        self.assertTrue(solve(IRBlueprint(line)).ok)

    def test_manual_chest_feeding_a_belt(self):
        # a chest filled by hand, an inserter puts its items on the belt the gear machine takes from
        ir = IRBlueprint([E("wooden-chest", 0, -2), E("inserter", 0, -1, 0)] + list(gear_line().entities))
        r = solve(ir)
        self.assertTrue(r.ok, r.problems)
        self.assertIn("iron-plate", {v for k, v in r.inputs.items() if k[0] == "chest"})

    def test_ammunition_passed_from_turret_to_turret(self):
        ir = IRBlueprint(belts(0, 3, 0) + [E("inserter", 1, 1, 0), E("gun-turret", 1, 2), E("inserter", 3, 2, 12),
                                           E("gun-turret", 4, 2)])
        r = solve(ir)
        self.assertTrue(r.ok, r.problems)

    def test_basic_oil_processing_uses_its_fluid_boxes(self):
        L = Layout(IRBlueprint([E("oil-refinery", 0, 0, 0, recipe="basic-oil-processing")]))
        needs, makes = _fluid_roles(L)
        (j, made), = makes.items()
        self.assertEqual([(L.boxes[b][1], f) for b, f in made], [(4, "petroleum-gas")])  # the third output box
        self.assertEqual([(L.boxes[bs[0]][1], f) for bs, f in needs[j]], [(1, "crude-oil")])  # the second input box

    def test_mirror_flips_fluid_connections(self):
        plain = Layout(IRBlueprint([E("chemical-plant", 0, 0, 0, recipe="plastic-bar")]))
        flipped = Layout(IRBlueprint([E("chemical-plant", 0, 0, 0, recipe="plastic-bar", mirror="true")]))
        first = lambda L: L._fluid_boxes(0)[0][2][0][0]  # noqa: E731  inside tile of the first input box
        self.assertEqual(first(plain), (0, 0))
        self.assertEqual(first(flipped), (2, 0))

    def test_pump_passes_fluid_one_way(self):
        # pipe -> pump facing south -> pipe: water put in above reaches below, not the other way
        ir = IRBlueprint([E("pipe", 0, 0), E("pump", 0, 1, 8), E("pipe", 0, 3)])
        L = Layout(ir)
        (up, down), = L.pumped
        top, bottom = 0, 2
        self.assertIn("water", check(ir, {("fluid", up): "water"}, L).fluids[bottom])
        self.assertNotIn("water", check(ir, {("fluid", down): "water"}, L).fluids.get(top, frozenset()))

    def test_underground_pipe_has_one_partner(self):
        # two pipes-to-ground face east, one faces west two tiles past the second: the near two pair,
        # the far one is a dead end (not joined, not an input from outside)
        ir = IRBlueprint([E("pipe-to-ground", 0, 0, 12), E("pipe-to-ground", 5, 0, 12), E("pipe-to-ground", 7, 0, 4)])
        L = Layout(ir)
        net = {L.ents[j].x: L.network[b] for b, (j, _, _, _) in enumerate(L.boxes)}
        self.assertEqual(net[5], net[7])
        self.assertNotEqual(net[0], net[7])
        self.assertNotIn((1, 0), {t for _, t, _ in L.open_ends})

    def test_lab_without_a_pack_the_others_get_idles(self):
        # two labs on one belt; only the second lab's inserter also reaches a second belt with green
        ir = IRBlueprint(belts(0, 7, 0) + belts(0, 7, 6) + [
            E("inserter", 1, 1, 0), E("lab", 0, 2), E("inserter", 5, 1, 0), E("lab", 4, 2), E("inserter", 5, 5, 8)])
        L = Layout(ir)
        top = [p for p in input_ports(L) if L.ents[p[1]].y == 0]
        bottom = [p for p in input_ports(L) if L.ents[p[1]].y == 6]
        inputs = {**{p: "automation-science-pack" for p in top}, **{p: "logistic-science-pack" for p in bottom}}
        r = check(ir, inputs, L)
        self.assertTrue(any(p.kind == "starved" and "logistic" in p.message for p in r.problems), r.problems)
        self.assertTrue(solve(ir).ok)  # red only on both: every lab gets every pack there is

    def test_design_made_to_dock(self):
        # the machine's input inserter takes from the belt of the copy to its left
        cell = IRBlueprint([E("inserter", 0, 1, 12), E("assembling-machine-1", 1, 0, recipe="iron-gear-wheel"),
                            E("inserter", 2, 3, 0), E("wooden-chest", 2, 4)] + [E("transport-belt", 4, y, 0) for y in range(3)])
        self.assertFalse(solve(cell).ok)
        from factoriollm.template_miner import middle_flow
        docked, _ = middle_flow(cell, (5, 0))
        self.assertTrue(docked.ok, docked.problems)


class FlowSimPartsTest(unittest.TestCase):
    """Layouts of parts that share no flow are solved part by part."""

    @staticmethod
    def lines(n, dy=10):
        return IRBlueprint([IREntity(e.name, e.x, e.y + k * dy, e.direction, e.extras)
                            for k in range(n) for e in gear_line([E("small-electric-pole", 4, 3)]).entities])

    def test_parts_share_no_flow(self):
        from factoriollm.flow_sim import _parts
        L = Layout(self.lines(3))
        parts = _parts(L)
        self.assertEqual(len(parts), 3)
        self.assertFalse(any(L.ents[i].name == "small-electric-pole" for p in parts for i in p))  # no flow

    def test_identical_parts_are_solved_once(self):
        from unittest import mock
        from factoriollm import flow_sim
        with mock.patch.object(flow_sim, "_solve_whole", wraps=flow_sim._solve_whole) as whole:
            r = solve(self.lines(4))
        self.assertTrue(r.ok, r.problems)
        self.assertEqual(whole.call_count, 1)
        fed = {r.inputs.get(("lane", i, l)) for i, e in enumerate(self.lines(4).entities) if e.x == 0 and e.y % 10 == 0
               for l in (0, 1)}
        self.assertEqual(fed, {"iron-plate"})  # every line got the assignment found for the first one

    def test_a_pipe_against_another_part_is_not_an_open_end(self):
        from factoriollm.flow_sim import _part_blueprint, _parts
        # a chemical plant's input pipe ends against a belt: blocked there, open only to the sides
        ir = IRBlueprint([E("chemical-plant", 0, 2, 0, recipe="sulfur"), E("pipe", 0, 1)] + belts(-2, 3, 0, d=4))
        L = Layout(ir)
        part = next(p for p in _parts(L) if any(L.ents[i].name == "chemical-plant" for i in p))
        sub, _, _, (ox, oy) = _part_blueprint(ir, L, part, {})
        ends = lambda L, dx=0, dy=0: {(t[0] + dx, t[1] + dy) for _, t, _ in L.open_ends}  # noqa: E731
        self.assertNotIn((0, 0), ends(L))
        self.assertEqual(ends(Layout(sub), ox, oy), ends(L))


class TemplateFeedTest(unittest.TestCase):
    """A template is fed by the player: one item per belt lane anywhere, inputs only where a belt can reach."""

    def circuit_line(self):
        # main belt east along row 0; a one-tile belt facing south side-loads onto it from the north,
        # so both its lanes land on the main belt's left lane; an assembler makes green circuits
        return IRBlueprint(belts(0, 4, 0) + [E("transport-belt", 2, -1, 8), E("inserter", 3, 1, 0),
                                              E("assembling-machine-1", 2, 2, recipe="electronic-circuit"),
                                              E("inserter", 3, 5, 0)] + belts(2, 5, 6))

    def test_side_loaded_lanes_must_not_mix(self):
        ir = self.circuit_line()
        L = Layout(ir)
        stub = next(i for i, e in enumerate(L.ents) if (e.x, e.y) == (2, -1))
        mixed = check(ir, {("lane", stub, 0): "iron-plate", ("lane", stub, 1): "copper-cable"}, L, one_per_lane=True)
        self.assertIn("lane_mixed", {p.kind for p in mixed.problems})
        r = solve(ir, share_lanes=False)
        self.assertTrue(r.ok, r.problems)
        self.assertTrue(all(len(items) <= 1 for items in r.lanes.values()), r.inputs)

    def test_mixed_products_may_leave_together(self):
        # gears and pipes dropped onto the same lane of the output belt: sorted further on, no jam
        ir = IRBlueprint(belts(0, 6, 0) + [E("inserter", 1, 1, 0), E("assembling-machine-1", 0, 2, recipe="iron-gear-wheel"),
                                            E("inserter", 4, 1, 0), E("assembling-machine-1", 3, 2, recipe="pipe"),
                                            E("inserter", 1, 5, 0), E("inserter", 4, 5, 0)] + belts(0, 6, 6))
        r = solve(ir, share_lanes=False)
        self.assertTrue(r.ok, r.problems)
        self.assertIn(frozenset({"iron-gear-wheel", "pipe"}), set(r.lanes.values()))

    def test_a_bare_fluid_box_is_no_template_input(self):
        # a template's fluid comes in through a pipe: the bare box of a lone machine is no input
        from factoriollm.flow_sim import _fluid_domains
        plant = E("chemical-plant", 0, 0, 0, recipe="lubricant")
        bare = Layout(IRBlueprint([plant]), outside_only=True)
        self.assertEqual(_fluid_domains(bare), {})
        self.assertTrue(_fluid_domains(Layout(IRBlueprint([plant]))))  # an author's design may be fed so
        box_tile = next(tile for b, tile, _ in bare.open_ends if bare.boxes[b][2] == "input")
        piped = Layout(IRBlueprint([plant, E("pipe", *box_tile)]), outside_only=True)
        self.assertEqual([v[0] for v in _fluid_domains(piped).values()], ["heavy-oil"])

    def test_a_walled_belt_start_is_no_template_input(self):
        # walls behind the belt start and beside both its lanes: nothing can be brought to it
        walls = [E("stone-wall", -1, 0), E("stone-wall", 0, -1), E("stone-wall", 0, 1)]
        ir = gear_line(walls)
        self.assertTrue(solve(ir).ok)  # an author's design may be fed in ways the blueprint does not show
        r = solve(ir, share_lanes=False)
        self.assertIn("starved", {p.kind for p in r.problems})


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

    def test_reactors_pass_fuel_cells_on(self):
        # a requester chest feeds one reactor; an inserter hands fuel cells on to the next reactor
        ir = IRBlueprint([E("requester-chest", 2, 0, requests="uranium-fuel-cell=10"), E("inserter", 2, 1, 0),
                          E("nuclear-reactor", 0, 2), E("inserter", 5, 4, 12), E("nuclear-reactor", 6, 2)])
        r = check(ir, {}, Layout(ir))
        self.assertEqual(r.problems, [])
        # a boiler hands no coal onto a belt (only to the next burner)
        boiler = IRBlueprint([E("boiler", 0, 0, 0), E("inserter", 3, 0, 12)] + belts(4, 6, 0))
        r = check(boiler, {}, Layout(boiler))
        self.assertTrue(all(not v for v in r.lanes.values()))

    def test_coal_liquefaction_works_once_primed(self):
        # the refineries take back the heavy oil they make: players prime the loop once
        r = solve(fixture("coal_liquefaction_primed"))
        self.assertTrue(r.ok, r.problems[:3])
        # a refinery whose heavy oil output is not piped back to its input cannot prime itself
        lone = IRBlueprint([E("oil-refinery", 0, 0, 0, recipe="coal-liquefaction")])
        self.assertTrue(any("heavy-oil" in p.message for p in check(lone, {}).problems))

    def test_hand_fed_hub_chest_holds_two_items(self):
        # one chest by hand feeds the gear, cable, circuit and red science machines: iron and copper
        r = solve(fixture("hand_fed_hub_chest"))
        self.assertTrue(r.ok, r.problems[:3])
        hub = [p for p in r.ports if p.kind == "chest"]
        self.assertEqual([p.items for p in hub], [frozenset({"iron-plate", "copper-plate"})])

    def test_ore_to_steel(self):
        r = solve(fixture("ore_to_steel_column"))
        self.assertTrue(r.ok, r.problems[:3])
        self.assertIn("steel-plate", {x for p in r.ports if p.role == "output" for x in p.items})


if __name__ == "__main__":
    unittest.main()
