import unittest

from factoriollm.blueprint_ir import IRBlueprint, IREntity, normalized
from factoriollm.dataset_lifter import (
    find_arithmetic_runs,
    ir_equivalent,
    lift_and_verify_locally,
    lift_ir_to_code,
    run_trusted_script,
)


def bp(*entities):
    return normalized(IRBlueprint(list(entities)))


class TestDatasetLifter(unittest.TestCase):
    def assertRoundTrips(self, ir):
        code = lift_ir_to_code(ir)
        self.assertTrue(ir_equivalent(ir, run_trusted_script(code)), code)
        return code

    def test_find_arithmetic_runs(self):
        self.assertEqual(find_arithmetic_runs([0, 2, 4, 6]), [(0, 2, 4)])
        self.assertEqual(find_arithmetic_runs([1, 2, 5, 6, 9, 10]), [(1, 4, 3), (2, 4, 3)])
        self.assertEqual(find_arithmetic_runs([0, 7]), [])

    def test_lift_belt_line_and_furnaces(self):
        ir = bp(
            *[IREntity("fast-transport-belt", x, 0, 4) for x in range(5)],
            *[IREntity("stone-furnace", x, 2) for x in (0, 2, 4)],
        )
        code = self.assertRoundTrips(ir)
        self.assertIn("add_belt_line", code)
        self.assertIn("add_entity_row(bp, 'stone-furnace', start=(0, 2), count=3, step=(2, 0))", code)

    def test_lift_underground_pair(self):
        ir = bp(
            IREntity("underground-belt", 0, 0, 4),
            IREntity("underground-belt", 5, 0, 4, (("io_type", "output"),)),
        )
        code = self.assertRoundTrips(ir)
        self.assertIn("add_underground_pair(bp, start=(0, 0), end=(5, 0), direction='east'", code)

    def test_lift_vertical_column_with_recipe(self):
        ir = bp(*[IREntity("assembling-machine-2", 0, y, 0, (("recipe", "electronic-circuit"),)) for y in (0, 4, 8)])
        code = self.assertRoundTrips(ir)
        self.assertIn("step=(0, 4)", code)
        self.assertIn("recipe='electronic-circuit'", code)

    def test_interleaved_rows_and_row_loops(self):
        xs = [1, 2, 5, 6, 9, 10]
        ir = bp(*[IREntity("inserter", x, y, 8) for x in xs for y in (0, 3)])
        code = self.assertRoundTrips(ir)
        self.assertIn("for start in [", code)
        self.assertIn("step=(4, 0), direction='south'", code)

    def test_flat_mode_has_no_macros(self):
        ir = bp(*[IREntity("transport-belt", x, 0, 4) for x in range(5)])
        code = lift_ir_to_code(ir, use_macros=False)
        self.assertNotIn("add_", code)
        self.assertEqual(code.count("bp.entities.append("), 5)
        self.assertTrue(ir_equivalent(ir, run_trusted_script(code)))

    def test_mismatch_detected(self):
        ir = bp(*[IREntity("transport-belt", x, 0, 4) for x in range(3)])
        other = bp(*[IREntity("transport-belt", x, 0, 8) for x in range(3)])
        self.assertFalse(ir_equivalent(other, run_trusted_script(lift_ir_to_code(ir))))
        self.assertIsNotNone(lift_and_verify_locally(ir))


if __name__ == "__main__":
    unittest.main()
