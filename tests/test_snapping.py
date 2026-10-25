import unittest
import warnings

from draftsman.blueprintable import Blueprint

from factoriollm.blueprint_ir import BlueprintIRError, extract_facts, parse_blueprint_dict, rotated
from factoriollm.dataset_lifter import lift_and_verify_locally
from factoriollm.export_dataset import ir_from_json


def chest(n, x, y):
    return {"entity_number": n, "name": "wooden-chest", "position": {"x": x + 0.5, "y": y + 0.5}}


class TestGridSnapping(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_relative_snap_keeps_entities_in_their_grid_cells(self):
        # the game exports relative-snapped blueprints with entities left of the cell origin, e.g. x = -2
        bp = {"entities": [chest(1, -2, 0), chest(2, 3, 1)], "snap-to-grid": {"x": 7, "y": 4}}
        ir = parse_blueprint_dict(bp)
        self.assertEqual((ir.snap, ir.absolute_snap), ((7, 4), False))
        self.assertEqual(sorted((e.x, e.y) for e in ir.entities), [(5, 0), (10, 1)])  # shifted by whole cells
        self.assertEqual(ir.size, (6, 2))

    def test_absolute_snap_with_offset(self):
        bp = {"entities": [chest(1, 0, 0)], "snap-to-grid": {"x": 32, "y": 32}, "absolute-snapping": True,
              "position-relative-to-grid": {"x": 5, "y": 1}}
        ir = parse_blueprint_dict(bp)
        self.assertEqual(ir.snap_key(), ((32, 32), True, (5, 1)))
        facts = extract_facts(ir)
        self.assertEqual((facts["snap"], facts["absolute_snapping"], facts["snap_offset"]), ([32, 32], True, [5, 1]))

    def test_lifted_code_sets_the_grid_and_round_trips(self):
        relative = parse_blueprint_dict({"entities": [chest(1, -2, 0), chest(2, 3, 1)], "snap-to-grid": {"x": 7, "y": 4}})
        code = lift_and_verify_locally(relative)
        self.assertIsNotNone(code)
        self.assertIn("bp.snapping_grid_size = (7, 4)", code)
        self.assertNotIn("absolute_snapping", code)
        absolute = parse_blueprint_dict({"entities": [chest(1, 0, 0)], "snap-to-grid": {"x": 32, "y": 32},
                                         "absolute-snapping": True, "position-relative-to-grid": {"x": 5, "y": 1}})
        code = lift_and_verify_locally(absolute)
        self.assertIn("bp.absolute_snapping = True", code)
        self.assertIn("bp.position_relative_to_grid = (5, 1)", code)

    def test_draftsman_exports_snapping_like_the_game(self):
        # the game writes "absolute-snapping" only as true; a missing key means relative snapping
        bp = Blueprint()
        bp.snapping_grid_size = (3, 4)
        self.assertNotIn("absolute-snapping", bp.to_dict()["blueprint"])
        self.assertFalse(Blueprint.from_string(bp.to_string()).absolute_snapping)
        bp.absolute_snapping = True
        self.assertIs(bp.to_dict()["blueprint"]["absolute-snapping"], True)

    def test_snap_mismatch_is_not_equivalent(self):
        ir = parse_blueprint_dict({"entities": [chest(1, 0, 0)], "snap-to-grid": {"x": 2, "y": 2}})
        plain = parse_blueprint_dict({"entities": [chest(1, 0, 0)]})
        self.assertIsNotNone(lift_and_verify_locally(ir))
        self.assertNotEqual(ir.snap_key(), plain.snap_key())

    def test_rotation_swaps_the_grid_and_keeps_absolute_blueprints_fixed(self):
        ir = parse_blueprint_dict({"entities": [chest(1, 0, 0), chest(2, 4, 1)], "snap-to-grid": {"x": 7, "y": 4}})
        turned = rotated(ir, 1)
        self.assertEqual(turned.snap, (4, 7))
        self.assertIsNotNone(lift_and_verify_locally(turned))
        absolute = parse_blueprint_dict({"entities": [chest(1, 0, 0)], "snap-to-grid": {"x": 4, "y": 4},
                                         "absolute-snapping": True})
        with self.assertRaises(BlueprintIRError):
            rotated(absolute, 1)

    def test_export_restores_snapping_from_facts(self):
        ir = parse_blueprint_dict({"entities": [chest(1, -2, 0)], "snap-to-grid": {"x": 7, "y": 4}})
        data = [[e.name, e.x, e.y, e.direction, dict(e.extras)] for e in ir.entities]
        restored = ir_from_json(data, extract_facts(ir))
        self.assertEqual(restored.snap_key(), ir.snap_key())
        self.assertEqual(restored.canonical_set(), ir.canonical_set())


class TestLegacyOrigin(unittest.TestCase):
    def test_uniform_half_tile_shift_of_old_strings_is_removed(self):
        # 0.16 export: a 1x1 belt and a 3x3 drill both sit on integer positions, i.e. half a tile off
        bp = {"entities": [{"entity_number": 1, "name": "transport-belt", "position": {"x": -14, "y": -4}},
                           {"entity_number": 2, "name": "electric-mining-drill", "position": {"x": -15, "y": -10}}]}
        ir = parse_blueprint_dict(bp, version=(0, 16))
        self.assertEqual(sorted((e.name, e.x, e.y) for e in ir.entities),
                         [("electric-mining-drill", 0, 0), ("transport-belt", 2, 7)])

    def test_mixed_offsets_and_new_strings_stay_errors(self):
        mixed = {"entities": [{"entity_number": 1, "name": "transport-belt", "position": {"x": 0, "y": 0}},
                              {"entity_number": 2, "name": "transport-belt", "position": {"x": 2.5, "y": 0.5}}]}
        with self.assertRaisesRegex(BlueprintIRError, "off the tile grid"):
            parse_blueprint_dict(mixed, version=(0, 16))
        shifted = {"entities": [{"entity_number": 1, "name": "transport-belt", "position": {"x": 0, "y": 0}}]}
        with self.assertRaisesRegex(BlueprintIRError, "off the tile grid"):
            parse_blueprint_dict(shifted, version=(2, 0))


if __name__ == "__main__":
    unittest.main()
