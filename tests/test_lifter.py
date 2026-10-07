import unittest
from factoriollm.dataset_lifter import (
    lift_blueprint_to_code,
    verify_code_equivalence,
    execute_script_to_blueprint,
)
from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt, StoneFurnace, UndergroundBelt


class TestDatasetLifter(unittest.TestCase):
    def test_lift_belt_line_and_furnaces(self):
        # Create unrolled code like the original dataset
        orig_code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt, StoneFurnace

bp = Blueprint()
bp.entities.append(TransportBelt('fast-transport-belt', position={'x': 0.0, 'y': 0.0}, direction=4))
bp.entities.append(TransportBelt('fast-transport-belt', position={'x': 1.0, 'y': 0.0}, direction=4))
bp.entities.append(TransportBelt('fast-transport-belt', position={'x': 2.0, 'y': 0.0}, direction=4))
bp.entities.append(TransportBelt('fast-transport-belt', position={'x': 3.0, 'y': 0.0}, direction=4))
bp.entities.append(TransportBelt('fast-transport-belt', position={'x': 4.0, 'y': 0.0}, direction=4))
bp.entities.append(StoneFurnace('stone-furnace', position={'x': 0.0, 'y': 3.0}))
bp.entities.append(StoneFurnace('stone-furnace', position={'x': 3.0, 'y': 3.0}))
bp.entities.append(StoneFurnace('stone-furnace', position={'x': 6.0, 'y': 3.0}))
print(bp.to_string())
"""
        bp_orig = execute_script_to_blueprint(orig_code)
        self.assertIsNotNone(bp_orig)

        lifted_code = lift_blueprint_to_code(bp_orig, "Test")
        self.assertIn("add_belt_line", lifted_code)
        self.assertIn("add_entity_row", lifted_code)

        # Verify 100% equivalence
        is_equivalent = verify_code_equivalence(orig_code, lifted_code)
        self.assertTrue(is_equivalent)

    def test_lift_underground_pair(self):
        orig_code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import UndergroundBelt

bp = Blueprint()
bp.entities.append(UndergroundBelt('underground-belt', position={'x': 0.0, 'y': 0.0}, direction=4, io_type='input'))
bp.entities.append(UndergroundBelt('underground-belt', position={'x': 5.0, 'y': 0.0}, direction=4, io_type='output'))
print(bp.to_string())
"""
        bp_orig = execute_script_to_blueprint(orig_code)
        self.assertIsNotNone(bp_orig)

        lifted_code = lift_blueprint_to_code(bp_orig, "Test")
        self.assertIn("add_underground_pair", lifted_code)

        is_equivalent = verify_code_equivalence(orig_code, lifted_code)
        self.assertTrue(is_equivalent)

    def test_lift_vertical_column_with_recipe(self):
        orig_code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import AssemblingMachine2

bp = Blueprint()
bp.entities.append(AssemblingMachine2('assembling-machine-2', position={'x': 2.0, 'y': 0.0}, recipe='electronic-circuit'))
bp.entities.append(AssemblingMachine2('assembling-machine-2', position={'x': 2.0, 'y': 4.0}, recipe='electronic-circuit'))
bp.entities.append(AssemblingMachine2('assembling-machine-2', position={'x': 2.0, 'y': 8.0}, recipe='electronic-circuit'))
print(bp.to_string())
"""
        bp_orig = execute_script_to_blueprint(orig_code)
        self.assertIsNotNone(bp_orig)

        lifted_code = lift_blueprint_to_code(bp_orig, "Assemblers")
        self.assertIn("add_entity_row", lifted_code)
        self.assertIn("step=(0.0, 4.0)", lifted_code)
        self.assertIn("electronic-circuit", lifted_code)

        is_equivalent = verify_code_equivalence(orig_code, lifted_code)
        self.assertTrue(is_equivalent)


if __name__ == "__main__":
    unittest.main()
