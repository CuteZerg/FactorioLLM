import unittest
import warnings

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.game_data import final_products, recipe
from factoriollm.template_miner import classify, default_template, is_mall, products, smelting_input
from factoriollm.templates import Template


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


class TestSmeltingInput(unittest.TestCase):
    def test_steel_from_labels_and_names(self):
        self.assertEqual(smelting_input(IRBlueprint(furnaces() + [label(0, "steel-plate")])), "steel")
        self.assertEqual(smelting_input(IRBlueprint(furnaces()), "Steel smelting 12"), "steel")
        self.assertEqual(smelting_input(IRBlueprint(furnaces()), "Плавка стали"), "steel")

    def test_ore_by_default_and_furnace_names_are_no_hint(self):
        self.assertEqual(smelting_input(IRBlueprint(furnaces()), "Iron smelting, steel furnaces"), "ore")
        self.assertEqual(smelting_input(IRBlueprint(furnaces()), "Стальные печи: железо"), "ore")
        self.assertEqual(smelting_input(IRBlueprint(furnaces()), "Остальное"), "ore")
        self.assertEqual(smelting_input(IRBlueprint(furnaces() + [label(0, "iron-plate")])), "ore")
        # a bus smelter labelled with ore and steel is mostly ore smelting
        self.assertEqual(smelting_input(IRBlueprint(furnaces() + [label(0, "iron-ore"), label(2, "steel-plate")])), "ore")


if __name__ == "__main__":
    unittest.main()
