import unittest
import warnings

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.game_data import final_products, recipe
from factoriollm.flow_sim import FlowResult, Port
from factoriollm.template_miner import (
    Rejects,
    author_hints,
    classify,
    default_template,
    furnace_input,
    is_mall,
    products,
    recipe_family,
    straight_wall,
)
from factoriollm.templates import Template, normalize_modules


def assemblers(*recipes, y=0):
    return [IREntity("assembling-machine-2", 3 * i, y, 0, (("recipe", r),)) for i, r in enumerate(recipes)]


def label(x, *items):
    return IREntity("constant-combinator", x, 10, 0, (("signals", ";".join(f"item:{i}=1" for i in items)),))


def furnaces(n=6):
    return [IREntity("stone-furnace", 2 * i, 0) for i in range(n)]


class TestGameData(unittest.TestCase):
    def test_recipe_amounts_kinds_and_default_time(self):
        circuit = recipe("electronic-circuit")
        self.assertEqual(circuit.ingredients, {"iron-plate": 1.0, "copper-cable": 3.0})
        self.assertEqual(circuit.time, 0.5)
        oil = recipe("advanced-oil-processing")
        self.assertEqual(oil.fluids, {"water", "crude-oil", "heavy-oil", "light-oil", "petroleum-gas"})
        self.assertIsNone(recipe("tungsten-plate"))  # Space Age

    def test_intermediates_are_not_final_products(self):
        self.assertEqual(final_products(["copper-cable", "electronic-circuit"]), {"electronic-circuit"})
        self.assertEqual(final_products(["iron-gear-wheel", "automation-science-pack"]), {"automation-science-pack"})


class TestClassify(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def classify(self, entities):
        return classify(IRBlueprint(entities))

    def test_green_circuits_with_their_cable(self):
        self.assertEqual(self.classify(assemblers("copper-cable", "copper-cable", "copper-cable",
                                                  "electronic-circuit", "electronic-circuit")), "green_circuits")

    def test_blue_circuits_are_not_green(self):
        block = assemblers("copper-cable", "electronic-circuit", "electronic-circuit", "processing-unit")
        self.assertEqual(self.classify(block), "blue_circuits")

    def test_two_science_packs_make_a_multi_block(self):
        block = assemblers("iron-gear-wheel", "automation-science-pack", "inserter", "transport-belt",
                           "logistic-science-pack")
        ir = IRBlueprint(block)
        self.assertEqual(classify(ir), "multi")
        self.assertEqual(products(ir), ["automation-science-pack", "logistic-science-pack"])

    def test_malls_are_not_mined(self):
        mall = assemblers("iron-chest", "pipe", "stone-furnace", "burner-inserter", "small-electric-pole")
        belts = assemblers("iron-gear-wheel", "transport-belt", "underground-belt", "splitter")
        for block in (mall, belts):
            ir = IRBlueprint(block)
            self.assertIsNone(classify(ir))
            self.assertTrue(is_mall(products(ir)))

    def test_solar_powering_a_pump_station_is_not_solar(self):
        panels = [IREntity("solar-panel", 3 * i, 0) for i in range(6)]
        self.assertEqual(self.classify(panels), "solar")
        pumps = [IREntity("offshore-pump", 0, 4, 0), IREntity("pipe", 0, 6)]
        self.assertIsNone(self.classify(panels + pumps))

    def test_power_needs_most_of_the_area(self):
        engines = [IREntity("boiler", 0, 0, 0)] + [IREntity("steam-engine", 0, 2 + 5 * i, 0) for i in range(2)]
        self.assertEqual(self.classify(engines), "steam")
        walls = [IREntity("stone-wall", x, 20 + y) for x in range(20) for y in range(5)]
        self.assertIsNone(self.classify(engines + walls))


class TestDefault(unittest.TestCase):
    def test_density_first_then_popularity(self):
        def t(tiles, favorites, beacons=0):
            return Template("labs", 4, [], [], [], 1, source={"favorites": favorites, "version": [2, 0]},
                            metrics={"tiles_per_machine": tiles, "beacons_per_machine": beacons})
        sparse, dense, popular_dense, beaconed = t(20, 100000), t(10, 10), t(11, 1000), t(5, 10**6, beacons=1)
        self.assertIs(default_template([sparse, dense, popular_dense, beaconed]), popular_dense)
        moduled = t(8, 10**5)
        moduled.metrics["traits"] = {"machine_modules": "productivity-3=4"}
        self.assertIs(default_template([dense, moduled]), dense)
        steel, ore = t(8, 10**5), t(12, 10)
        steel.metrics["input"], ore.metrics["input"] = "steel", "ore"
        self.assertIs(default_template([steel, ore]), ore)
        self.assertIs(default_template([steel]), steel)  # a preference never empties the choice


class TestFamiliesAndDefense(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_gears_are_a_one_input_family(self):
        self.assertEqual(classify(IRBlueprint(assemblers(*["iron-gear-wheel"] * 4))), "assembly_1in")
        self.assertEqual(recipe_family("engine-unit"), "assembly_3in")
        self.assertEqual(recipe_family("electric-engine-unit"), "assembly_fluid")
        self.assertEqual(recipe_family("sulfur"), "chemistry")

    def test_wall_with_turrets_is_defense(self):
        wall = [IREntity("stone-wall", x, y) for x in range(20) for y in (0, 1)]
        turrets = [IREntity("laser-turret", 4 * i, 3) for i in range(5)]
        self.assertEqual(classify(IRBlueprint(wall + turrets)), "defense")


class TestStraightWall(unittest.TestCase):
    def wall(self, ys, x0=0, n=4):
        return [IREntity("stone-wall", x0 + x, y) for x in range(n) for y in ys]

    def test_straight_wall_and_ring(self):
        straight = Template("defense", 4, self.wall([0, 1], -2, 2), self.wall([0, 1]), self.wall([0, 1], 8, 2), 2)
        self.assertTrue(straight_wall(straight))
        caps = [IREntity("stone-wall", -1, y) for y in range(10)]  # a ring: the head closes the two walls
        ring = Template("defense", 4, caps, self.wall([0, 9]), [], 2)
        self.assertFalse(straight_wall(ring))
        across = Template("defense", 1, [], self.wall(range(30), n=1), [], 5)  # repeats across a long wall
        self.assertFalse(straight_wall(across))
        turrets = Template("defense", 4, [], [IREntity("gun-turret", 0, 0)], [], 3)
        self.assertTrue(straight_wall(turrets))


class TestSmeltingInput(unittest.TestCase):
    def flow(self, inputs, outputs):
        port = Port("lane", "output", 0, 0, frozenset(outputs), (0, 0))
        return FlowResult([], {}, {}, frozenset(), [port], inputs={("lane", 0, 0): x for x in inputs})

    def test_what_the_simulator_found(self):
        self.assertEqual(furnace_input(self.flow(["iron-plate"], ["steel-plate"])), "steel")
        self.assertEqual(furnace_input(self.flow(["iron-ore"], ["steel-plate"])), "ore_to_steel")
        self.assertEqual(furnace_input(self.flow(["iron-ore"], ["iron-plate"])), "ore")


class TestMinerParts(unittest.TestCase):
    def test_author_hints(self):
        hints = author_hints(IRBlueprint([label(3, "coal", "iron-ore")]))
        self.assertEqual(hints, {(3, 10): {"coal", "iron-ore"}})

    def test_modules_follow_the_group(self):
        def lab(x, modules):
            return IREntity("lab", 3 * x, 0, 0, (("modules", modules),) if modules else ())
        labs = [lab(i, "speed-module-3=2") for i in range(5)] + [lab(5, "")]
        t = normalize_modules(Template("labs", 18, [], labs, [], 1))
        self.assertEqual({e.extras_dict.get("modules") for e in t.segment}, {"speed-module-3=2"})
        self.assertEqual(t.modules, {"lab": "speed-module-3=2"})
        build = t.build(1, modules={"lab": "productivity-module-3=2"})
        self.assertEqual({e.extras_dict.get("modules") for e in build.entities}, {"productivity-module-3=2"})

    def test_modules_per_recipe_stay(self):
        def am(x, r, modules):
            return IREntity("assembling-machine-3", 3 * x, 0, 0, (("modules", modules), ("recipe", r)))
        ents = [am(i, "iron-gear-wheel", "productivity-module-3=4") for i in range(3)] + \
            [am(3 + i, "electronic-circuit", "speed-module-3=4") for i in range(4)]
        t = normalize_modules(Template("multi", 21, [], ents, [], 1))
        self.assertEqual(t.modules, {"assembling-machine-3": "speed-module-3=4"})
        self.assertEqual(t.modules_by_recipe, {"iron-gear-wheel": "productivity-module-3=4"})
        self.assertEqual(Template.from_json(t.to_json()).modules_by_recipe, t.modules_by_recipe)

    def test_rejects_keep_a_few_per_reason(self):
        rejects = Rejects(per_reason=2)
        empty = FlowResult([], {}, {}, frozenset(), [])
        for i in range(10):
            rejects.add("clog", f"design {i}", IRBlueprint([]), empty)
        self.assertEqual(rejects.seen["clog"], 10)
        self.assertEqual(len(rejects.kept["clog"]), 2)


class CropKeepsInputsAtTheBorderTest(unittest.TestCase):
    def test_bus_tap_kept_when_cutting_it_leaves_an_inner_input(self):
        # the battle bots block feeds a column from the bus through a splitter tap; cutting the tap
        # leaves the column starting in the middle of the design, so a gentler cut must be used
        import sys
        from collections import Counter
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tests.test_golden_set import GOLDEN, load
        from factoriollm.template_miner import Rejects, _mine_design, inner_inputs, run_flow
        _, ir = load(GOLDEN / "battle_bots.json")
        stats = Counter()
        t = _mine_design(ir, "assembly_2in", "battle bots", stats, Rejects())
        build = t.build(1)
        flow, _ = run_flow(build, share_lanes=False)
        self.assertTrue(flow.ok, flow.problems[:3])
        self.assertEqual(inner_inputs(build, flow), [])
        self.assertNotIn("crop step 0", stats)  # the cut of bus taps was refused


class PlainlyFedTemplatesTest(unittest.TestCase):
    def test_unfinished_weave_is_no_template(self):
        # the author did not run the steel underground to the last machine of each row (checked in
        # game): it works only with steel mixed onto the gear lane, which a template must not need
        import json
        from collections import Counter
        from pathlib import Path
        from factoriollm.template_miner import Rejects, _mine_design
        data = json.loads((Path(__file__).parent / "fixtures" / "flow" / "engines_unfinished_weave.json").read_text(encoding="utf-8"))
        ir = IRBlueprint([IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]],
                         version=tuple(data["version"]))
        rejects = Rejects()
        self.assertIsNone(_mine_design(ir, "assembly_3in", "Engines", Counter(), rejects))
        self.assertIn("template: starved", rejects.seen)


class DensityTest(unittest.TestCase):
    def test_a_long_feed_with_poles_is_too_sparse(self):
        from factoriollm.template_miner import MIN_DENSITY, density
        block = [IREntity("assembling-machine-2", 3 * i, 0) for i in range(4)]
        self.assertGreater(density(IRBlueprint(block)), MIN_DENSITY)
        feed = [IREntity("transport-belt", 12, y, 0) for y in range(3, 60)] + \
               [IREntity("medium-electric-pole", 11, y) for y in range(3, 60, 9)]
        self.assertLess(density(IRBlueprint(block + feed)), MIN_DENSITY)


class StraightTurretRowTest(unittest.TestCase):
    def test_turrets_without_walls_must_be_one_band(self):
        from factoriollm.template_miner import straight_wall
        from factoriollm.templates import Template
        row = Template("defense", 4, [], [IREntity("gun-turret", 0, 0)], [], 3)
        self.assertTrue(straight_wall(row))
        corner = Template("defense", 4, [IREntity("gun-turret", -2, y) for y in range(2, 12, 2)],
                          [IREntity("gun-turret", 0, 0)], [], 3)  # the column of the corner leaves the band
        self.assertFalse(straight_wall(corner))
        ring = Template("defense", 4, [], [IREntity("laser-turret", 0, 0), IREntity("laser-turret", 0, 20)], [], 3)
        self.assertFalse(straight_wall(ring))


if __name__ == "__main__":
    unittest.main()
