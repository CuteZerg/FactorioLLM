import unittest

from factoriollm.base_game import FLUIDS, ITEMS, RECIPE_CHANGES, RECIPES, VIRTUAL_SIGNALS
from factoriollm.blueprint_ir import BlueprintIRError, parse_blueprint_dict
from factoriollm.game_data import recipe


def _bp(*entities):
    return {"entities": [dict(e, entity_number=i + 1) for i, e in enumerate(entities)]}


class BaseGameDataTest(unittest.TestCase):
    def test_no_space_age_ingredients(self):
        known = ITEMS | FLUIDS
        for name, r in RECIPES.items():
            for part in list(r["ingredients"]) + list(r["results"]):
                self.assertIn(part, known, f"{name}: {part}")
        self.assertNotIn("superconductor", recipe("personal-roboport-mk2-equipment").ingredients)
        self.assertEqual(set(recipe("speed-module-3").ingredients), {"speed-module-2", "advanced-circuit", "processing-unit"})

    def test_amounts_and_fluids(self):
        r = recipe("advanced-oil-processing")
        self.assertEqual(r.ingredients, {"water": 50.0, "crude-oil": 100.0})
        self.assertEqual(r.category, "oil-processing")
        self.assertIn("petroleum-gas", r.fluids)
        self.assertEqual(recipe("military-science-pack").ingredients,
                         {"piercing-rounds-magazine": 1.0, "grenade": 1.0, "stone-wall": 2.0})
        self.assertIn("water-barrel", RECIPES)
        self.assertIn("water-barrel", ITEMS)

    def test_signals(self):
        self.assertIn("signal-A", VIRTUAL_SIGNALS)
        self.assertIn("signal-each", VIRTUAL_SIGNALS)
        self.assertNotIn("tungsten-plate", ITEMS)

    def test_recipe_history(self):
        self.assertIn("military-science-pack", RECIPE_CHANGES["0.16"])
        self.assertIn("production-science-pack", RECIPE_CHANGES["0.16"])
        self.assertNotIn("military-science-pack", RECIPE_CHANGES["0.17"])
        for changes in RECIPE_CHANGES.values():
            self.assertNotIn("electronic-circuit", changes)


class BaseGamePolicyTest(unittest.TestCase):
    def test_legacy_recipe_rejected(self):
        bp = _bp({"name": "assembling-machine-2", "position": {"x": 1.5, "y": 1.5}, "recipe": "military-science-pack"})
        with self.assertRaisesRegex(BlueprintIRError, "legacy recipe"):
            parse_blueprint_dict(bp, version=(0, 16))
        self.assertEqual(len(parse_blueprint_dict(bp, version=(0, 17)).entities), 1)
        ok = _bp({"name": "assembling-machine-2", "position": {"x": 1.5, "y": 1.5}, "recipe": "electronic-circuit"})
        self.assertEqual(len(parse_blueprint_dict(ok, version=(0, 16)).entities), 1)

    def test_entity_quality(self):
        bp = _bp({"name": "assembling-machine-2", "position": {"x": 1.5, "y": 1.5}, "quality": "rare"})
        with self.assertRaisesRegex(BlueprintIRError, "quality"):
            parse_blueprint_dict(bp)
        bp["entities"][0]["quality"] = "normal"
        self.assertEqual(len(parse_blueprint_dict(bp).entities), 1)

    def test_splitter_filters(self):
        def splitter(flt):
            return _bp({"name": "splitter", "position": {"x": 1.0, "y": 0.5}, "filter": flt, "output_priority": "left"})
        for flt in ({"name": "iron-plate"}, {"name": "iron-plate", "quality": "normal", "comparator": "="}, "iron-plate"):
            ir = parse_blueprint_dict(splitter(flt))
            self.assertEqual(ir.entities[0].extras_dict["filter"], "iron-plate")
        for flt in ({"quality": "rare", "comparator": "="}, {"name": "iron-plate", "quality": "epic", "comparator": "="},
                    {"name": "scrap"}):
            with self.assertRaises(BlueprintIRError):
                parse_blueprint_dict(splitter(flt))

    def test_space_age_label_signal(self):
        bp = _bp({"name": "constant-combinator", "position": {"x": 0.5, "y": 0.5},
                  "control_behavior": {"sections": {"sections": [{"index": 1, "filters": [
                      {"index": 1, "type": "item", "name": "tungsten-plate", "comparator": "=", "count": 1}]}]}}})
        with self.assertRaisesRegex(BlueprintIRError, "label signal"):
            parse_blueprint_dict(bp)

    def test_game_string_filter_format(self):
        from draftsman.blueprintable import Blueprint
        from draftsman.entity import new_entity
        from draftsman.utils import string_to_JSON

        from factoriollm.helpers import game_string
        bp = Blueprint()
        bp.entities.append(new_entity("splitter", tile_position=(0, 0), filter="iron-plate", output_priority="left"))
        data = string_to_JSON(game_string(bp))
        self.assertEqual(data["blueprint"]["entities"][0]["filter"], {"name": "iron-plate"})


if __name__ == "__main__":
    unittest.main()
