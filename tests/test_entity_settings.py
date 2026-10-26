import json
import unittest
import warnings

from draftsman.blueprintable import Blueprint

from factoriollm.blueprint_ir import BlueprintIRError, extract_facts, parse_blueprint_dict, parse_blueprint_string
from factoriollm.dataset_lifter import lift_and_verify_locally, lift_ir_to_code
from factoriollm.helpers import add_entity, add_entity_row


def beacon(number=1, x=1.5, items=None):
    return {"entity_number": number, "name": "beacon", "position": {"x": x, "y": 1.5}, "items": items}


def modules_20(name, inventory, count, quality=None):
    ident = {"name": name, **({"quality": quality} if quality else {})}
    return {"id": ident, "items": {"in_inventory": [{"inventory": inventory, "stack": i} for i in range(count)]}}


def requester(number=1, x=0.5, request_filters=None, **extra):
    return {"entity_number": number, "name": "requester-chest", "position": {"x": x, "y": 0.5},
            "request_filters": request_filters, **extra}


def section(*filters, index=1, active=None):
    out = {"index": index, "filters": [{"index": i + 1, "name": n, "quality": "normal", "comparator": "=", "count": c}
                                       for i, (n, c) in enumerate(filters)]}
    if active is not None:
        out["active"] = active
    return out


class TestModules(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_formats_of_every_version(self):
        cases = [
            ((2, 0), [modules_20("speed-module-3", 1, 2)]),
            ((1, 1), {"speed-module-3": 2}),
            ((0, 17), [{"item": "speed-module-3", "count": 2}]),
        ]
        for version, items in cases:
            ir = parse_blueprint_dict({"entities": [beacon(items=items)]}, version=version)
            self.assertEqual(ir.entities[0].extras_dict["modules"], "speed-module-3=2", version)

    def test_mixed_modules_and_legacy_names(self):
        am = {"entity_number": 1, "name": "assembling-machine-3", "position": {"x": 1.5, "y": 1.5}, "recipe": "iron-gear-wheel",
              "items": {"productivity-module-3": 2, "effectivity-module": 1, "speed-module": 1}}
        ir = parse_blueprint_dict({"entities": [am]}, version=(1, 1))
        self.assertEqual(ir.entities[0].extras_dict["modules"],
                         "efficiency-module=1;productivity-module-3=2;speed-module=1")
        self.assertEqual(extract_facts(ir)["modules"], {"efficiency-module": 1, "productivity-module-3": 2, "speed-module": 1})

    def test_fuel_and_ingredient_requests_are_dropped(self):
        drill = {"entity_number": 1, "name": "electric-mining-drill", "position": {"x": 1.5, "y": 1.5},
                 "items": [modules_20("productivity-module", 2, 3), {"id": {"name": "coal"}, "items": {"in_inventory": [
                     {"inventory": 1, "stack": 0, "count": 50}]}}]}
        boiler = {"entity_number": 2, "name": "boiler", "position": {"x": 5.5, "y": 1}, "items": {"coal": 5}}
        ir = parse_blueprint_dict({"entities": [drill, boiler]})
        extras = {e.name: e.extras_dict for e in ir.entities}
        self.assertEqual(extras["electric-mining-drill"], {"modules": "productivity-module=3"})
        self.assertEqual(extras["boiler"], {})

    def test_modules_outside_the_module_inventory_are_ignored(self):
        # a module put into an assembler's input inventory is an ingredient, not a module
        am = {"entity_number": 1, "name": "assembling-machine-2", "position": {"x": 1.5, "y": 1.5}, "recipe": "speed-module-2",
              "items": [modules_20("speed-module", 2, 2)]}
        self.assertNotIn("modules", parse_blueprint_dict({"entities": [am]}).entities[0].extras_dict)

    def test_quality_mod_and_overfull_modules_are_rejected(self):
        bad = [
            ([modules_20("speed-module-3", 1, 2, quality="legendary")], "quality"),
            ([modules_20("quality-module-3", 1, 2)], "not in the base game"),
            ({"speed-module-4": 2}, "not in the base game"),
            ({"speed-module-3": 3}, "slots"),
        ]
        for items, message in bad:
            with self.assertRaisesRegex(BlueprintIRError, message):
                parse_blueprint_dict({"entities": [beacon(items=items)]}, version=(1, 1) if isinstance(items, dict) else (2, 0))

    def test_rows_keep_modules_and_rebuild_exactly(self):
        beacons = [beacon(i + 1, x=1.5 + 3 * i, items=[modules_20("speed-module-3", 1, 2)]) for i in range(4)]
        ir = parse_blueprint_dict({"entities": beacons})
        code = lift_and_verify_locally(ir)
        self.assertIsNotNone(code)
        self.assertIn("add_entity_row(bp, 'beacon', start=(0, 0), count=4, step=(3, 0), modules={'speed-module-3': 2})", code)


class TestRequests(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_formats_of_every_version(self):
        legacy = requester(request_filters=[{"index": 2, "name": "copper-cable", "count": 400},
                                            {"index": 1, "name": "iron-plate", "count": 200}], request_from_buffers=True)
        ir = parse_blueprint_dict({"entities": [legacy]}, version=(1, 1))
        self.assertEqual(ir.entities[0].extras_dict,
                         {"requests": "iron-plate=200;copper-cable=400", "request_from_buffers": "true"})
        new = requester(request_filters={"sections": [section(("iron-plate", 200), ("copper-cable", 400))],
                                         "request_from_buffers": True, "trash_not_requested": True})
        self.assertEqual(parse_blueprint_dict({"entities": [new]}).entities[0].extras_dict,
                         {"requests": "iron-plate=200;copper-cable=400", "request_from_buffers": "true",
                          "trash_not_requested": "true"})

    def test_sections_are_merged_and_inactive_ones_skipped(self):
        chest = requester(request_filters={"sections": [section(("iron-plate", 200)), section(("coal", 50), index=2),
                                                        section(("stone", 10), index=3, active=False)]})
        self.assertEqual(parse_blueprint_dict({"entities": [chest]}).entities[0].extras_dict["requests"],
                         "iron-plate=200;coal=50")

    def test_empty_sections_add_nothing(self):
        chest = {"entity_number": 1, "name": "passive-provider-chest", "position": {"x": 0.5, "y": 0.5},
                 "request_filters": {"sections": [{"index": 1}]}}
        empty = requester(request_filters={"sections": [{"index": 1}]})
        self.assertEqual([e.extras for e in parse_blueprint_dict({"entities": [chest, empty | {"entity_number": 2,
                          "position": {"x": 1.5, "y": 0.5}}]}).entities], [(), ()])

    def test_space_age_and_duplicate_requests_are_rejected(self):
        bad = [
            ({"sections": [section(("tungsten-plate", 10))]}, "not in the base game"),
            ({"sections": [{"index": 1, "filters": [{"index": 1, "name": "iron-plate", "quality": "rare", "count": 1}]}]}, "quality"),
            ({"sections": [section(("iron-plate", 10)), section(("iron-plate", 20), index=2)]}, "several slots"),
        ]
        for rf, message in bad:
            with self.assertRaisesRegex(BlueprintIRError, message):
                parse_blueprint_dict({"entities": [requester(request_filters=rf)]})

    def test_storage_chest_filter(self):
        chest = {"entity_number": 1, "name": "logistic-chest-storage", "position": {"x": 0.5, "y": 0.5},
                 "request_filters": [{"index": 1, "name": "iron-gear-wheel", "count": 0}]}
        ir = parse_blueprint_dict({"entities": [chest]}, version=(1, 1))
        self.assertEqual(ir.entities[0].extras_dict, {"requests": "iron-gear-wheel=0"})
        self.assertIsNotNone(lift_and_verify_locally(ir))

    def test_lifted_code_rebuilds_requests_exactly(self):
        chests = [requester(1, request_filters={"sections": [section(("iron-plate", 200), ("copper-cable", 400))],
                                                "request_from_buffers": True}),
                  requester(2, x=3.5, request_filters={"sections": [section(("coal", 50))]})]
        ir = parse_blueprint_dict({"entities": chests})
        code = lift_and_verify_locally(ir)
        self.assertIsNotNone(code)
        self.assertIn("add_entity(bp, 'requester-chest', (0, 0), request_from_buffers=True, "
                      "requests={'iron-plate': 200, 'copper-cable': 400})", code)

    def test_lifted_code_parses_back_in_the_same_slot_order(self):
        chest = requester(request_filters={"sections": [section(("stone", 1), ("coal", 2), ("iron-ore", 3))]})
        self.assertIn("requests={'stone': 1, 'coal': 2, 'iron-ore': 3}", lift_ir_to_code(parse_blueprint_dict({"entities": [chest]})))


class TestHelpers(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_export_matches_the_game_format(self):
        bp = Blueprint()
        add_entity(bp, "requester-chest", (0, 0), requests={"iron-plate": 200})
        add_entity_row(bp, "assembling-machine-3", start=(2, 0), count=2, step=(3, 0), recipe="iron-gear-wheel",
                       modules={"productivity-module-3": 4})
        entities = bp.to_dict()["blueprint"]["entities"]
        # the game numbers the first section 1 (draftsman 3.3.1 would write 2 without the fix)
        self.assertEqual(entities[0]["request_filters"]["sections"][0]["index"], 1)
        stacks = entities[1]["items"][0]["items"]["in_inventory"]
        self.assertEqual([s["stack"] for s in stacks], [0, 1, 2, 3])
        ir = parse_blueprint_string(bp.to_string())
        self.assertEqual([e.extras_dict.get("modules") for e in ir.entities if e.name.startswith("assembling")],
                         ["productivity-module-3=4"] * 2)

    def test_too_many_modules_and_overlaps_are_refused(self):
        bp = Blueprint()
        with self.assertRaisesRegex(ValueError, "module slots"):
            add_entity(bp, "beacon", (0, 0), modules={"speed-module-3": 3})
        add_entity(bp, "beacon", (0, 0), modules={"speed-module-3": 2})
        with self.assertRaisesRegex(ValueError, "already occupied"):
            add_entity(bp, "requester-chest", (1, 1), requests={"coal": 1})
        self.assertEqual(json.loads(json.dumps(bp.to_dict()["blueprint"]["entities"][0]["items"]))[0]["id"]["name"],
                         "speed-module-3")


if __name__ == "__main__":
    unittest.main()
