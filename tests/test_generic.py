import unittest

from factoriollm.blueprint_ir import IREntity
from factoriollm.generic import Incompatible, _machine_for, instantiate, row_machines, shaped_like, throughput, with_recipe
from factoriollm.game_data import recipe
from factoriollm.templates import Template


def row(machine: str = "assembling-machine-2", recipe_name: str = "iron-gear-wheel") -> Template:
    """One machine per 3 tiles, fed from a belt north of it, products onto a belt south of it (both lanes free)."""
    ents = [IREntity(machine, 0, 2, 0, (("recipe", recipe_name),)), IREntity("inserter", 1, 1, 0),
            IREntity("inserter", 1, 5, 0), IREntity("small-electric-pole", 2, 1, 0)]
    ents += [IREntity("transport-belt", x, 0, 4) for x in range(3)] + [IREntity("transport-belt", x, 6, 4) for x in range(3)]
    return Template("assembly_1in", 3, [], ents, [], 3, (3, 7), metrics={"family": "machine_row", "source_warnings": []})


class GenericTest(unittest.TestCase):
    def test_recipe_goes_into_every_machine_of_the_row(self):
        t = with_recipe(row(), "copper-cable")
        self.assertEqual(row_machines(t), ("assembling-machine-2", "copper-cable"))

    def test_two_ingredients_on_the_two_lanes(self):
        g = instantiate(row(), "electronic-circuit", segments=2)
        self.assertTrue(g.flow.ok)
        self.assertEqual({x for x in g.flow.inputs.values() if x}, {"iron-plate", "copper-cable"})

    def test_what_does_not_fit_is_refused_with_the_reason(self):
        with self.assertRaisesRegex(Incompatible, "3 solid ingredients.*2 lanes"):
            with_recipe(row(), "engine-unit")
        with self.assertRaisesRegex(Incompatible, "fluid"):
            with_recipe(row(), "concrete")  # no pipe to the machines
        with self.assertRaisesRegex(Incompatible, "not crafted"):
            with_recipe(row(), "sulfur")  # a chemical plant is 3x3 too, but no assembler

    def test_machine_of_the_family_that_crafts_the_recipe(self):
        self.assertEqual(_machine_for(recipe("concrete"), "assembling-machine-1"), "assembling-machine-2")
        self.assertIsNone(_machine_for(recipe("sulfur"), "assembling-machine-2"))

    def test_demand_of_a_machine(self):
        # research note: gears on AM2 eat 3 iron plates per second, 2 machines per yellow lane
        g = instantiate(row(), "iron-gear-wheel", segments=2)
        tp = throughput(g)
        self.assertAlmostEqual(tp["per_machine_in"]["iron-plate"], 3.0)
        lanes = sum(1 for k, x in g.flow.inputs.items() if x == "iron-plate")
        self.assertEqual(tp["machines_per_lane_load"]["iron-plate"], int(lanes * 7.5 / 3.0))

    def test_recipes_of_the_same_shape_first(self):
        picks = shaped_like(row(), limit=3)
        self.assertTrue(all(sum(i not in recipe(p).fluids for i in recipe(p).ingredients) == 1 for p in picks), picks)
        self.assertNotIn("iron-gear-wheel", picks)


if __name__ == "__main__":
    unittest.main()
