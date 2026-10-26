import unittest

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.dataset_lifter import lift_ir_to_code, run_trusted_script
from factoriollm.validator import validate, validate_string


def bp(*entities):
    return IRBlueprint(list(entities))


def smelter_row(n=3):
    """belt (y=0, east) -> inserter (y=1, north-facing: takes from belt, drops south) -> furnace (y=2..3),
    small pole between furnace pairs."""
    ents = [IREntity("transport-belt", x, 0, 4) for x in range(2 * n + 1)]
    for k in range(n):
        ents.append(IREntity("inserter", 2 * k, 1, 0))
        ents.append(IREntity("stone-furnace", 2 * k, 2))
    ents.append(IREntity("small-electric-pole", 1, 1))
    ents.append(IREntity("small-electric-pole", 3, 1))
    ents.append(IREntity("small-electric-pole", 5, 1))
    return ents


def categories(report):
    return {i.category for i in report.errors}, {i.category for i in report.warnings}


class TestValidator(unittest.TestCase):
    def test_correct_smelter_row_is_clean(self):
        report = validate(bp(*smelter_row()))
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])
        self.assertEqual(report.facts["entity_counts"]["stone-furnace"], 3)

    def test_offshore_pump_blocks_the_tile_in_front(self):
        # the 2.0 pump's collision box reaches ~1 tile towards the water (north when facing north)
        blocked = validate(IRBlueprint([IREntity("offshore-pump", 0, 1, 0), IREntity("pipe", 0, 0)]))
        self.assertEqual([i.category for i in blocked.errors], ["collision"])
        behind = validate(IRBlueprint([IREntity("offshore-pump", 0, 0, 0), IREntity("pipe", 0, 1)]))
        self.assertEqual(behind.errors, [])

    def test_collision_is_error(self):
        errors, _ = categories(validate(bp(IREntity("stone-furnace", 0, 0), IREntity("transport-belt", 1, 1, 4))))
        self.assertIn("collision", errors)

    def test_inserter_into_pole_or_from_wall_is_error(self):
        into_pole = bp(IREntity("wooden-chest", 0, 0), IREntity("inserter", 0, 1), IREntity("small-electric-pole", 0, 2))
        from_wall = bp(IREntity("stone-wall", 0, 0), IREntity("inserter", 0, 1), IREntity("wooden-chest", 0, 2))
        for ir in (into_pole, from_wall):
            errors, _ = categories(validate(ir))
            self.assertIn("inserter_target", errors)

    def test_inserter_on_ground_is_warning(self):
        # the empty drop tile (0, 2) is inside the blueprint because of the chest at (1, 2)
        ir = bp(IREntity("wooden-chest", 0, 0), IREntity("burner-inserter", 0, 1), IREntity("iron-chest", 1, 2))
        errors, warnings = categories(validate(ir))
        self.assertEqual(errors, set())
        self.assertIn("inserter_ground", warnings)

    def test_inserter_reaching_outside_blueprint_is_fine(self):
        # tileable modules feed from belts / wagons that are not part of the blueprint
        _, warnings = categories(validate(bp(IREntity("wooden-chest", 0, 0), IREntity("burner-inserter", 0, 1))))
        self.assertNotIn("inserter_ground", warnings)

    def test_rotated_inserter_targets(self):
        # east-facing inserter takes from the east tile and drops west
        ir = bp(IREntity("wooden-chest", 0, 0), IREntity("burner-inserter", 1, 0, 4), IREntity("iron-chest", 2, 0))
        report = validate(ir)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_belt_into_furnace_is_warning(self):
        errors, warnings = categories(validate(bp(IREntity("transport-belt", 0, 0, 4), IREntity("stone-furnace", 1, 0), IREntity("transport-belt", 0, 1, 8))))
        self.assertEqual(errors, set())
        self.assertIn("belt_end", warnings)

    def test_head_on_belts_warning(self):
        _, warnings = categories(validate(bp(IREntity("transport-belt", 0, 0, 4), IREntity("transport-belt", 1, 0, 12))))
        self.assertIn("belt_head_on", warnings)

    def test_lone_underground_is_warning(self):
        errors, warnings = categories(validate(bp(IREntity("transport-belt", 0, 0, 4), IREntity("underground-belt", 1, 0, 4))))
        self.assertEqual(errors, set())
        self.assertIn("lone_underground", warnings)

    def test_power_warnings(self):
        no_poles = bp(IREntity("assembling-machine-2", 0, 0))
        self.assertIn("no_poles", categories(validate(no_poles))[1])
        far_pole = bp(IREntity("assembling-machine-2", 0, 0), IREntity("small-electric-pole", 20, 0))
        self.assertIn("unpowered", categories(validate(far_pole))[1])
        near_pole = bp(IREntity("assembling-machine-2", 0, 0), IREntity("small-electric-pole", 4, 1))
        self.assertNotIn("unpowered", categories(validate(near_pole))[1])

    def test_validate_string_reports_collisions_instead_of_raising(self):
        clean = run_trusted_script(lift_ir_to_code(bp(*smelter_row(2))))
        self.assertTrue(validate_string(clean).ok)

    def test_center_positions_are_snapped_and_reported_as_off_grid(self):
        # The legacy model wrote 1x1 belts with integer *centers* (position={'x': 0.0, 'y': 0.0}).
        from draftsman.blueprintable import Blueprint

        legacy = Blueprint()
        for x in range(3):
            legacy.entities.append("transport-belt", position={"x": float(x), "y": 0.0}, direction=4)
        legacy.entities.append("stone-furnace", position={"x": 1.0, "y": 2.0})  # 2x2 on grid
        report = validate_string(legacy.to_string())
        self.assertIn("off_grid", {i.category for i in report.errors})
        self.assertIn("3 entities", report.errors[0].message)
        self.assertEqual(report.facts["entity_counts"], {"transport-belt": 3, "stone-furnace": 1})

    def test_feedback_text(self):
        report = validate(bp(IREntity("stone-wall", 0, 0), IREntity("inserter", 0, 1), IREntity("wooden-chest", 0, 2)))
        self.assertIn("[inserter_target]", report.feedback())


if __name__ == "__main__":
    unittest.main()
