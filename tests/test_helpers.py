import unittest
from draftsman.blueprintable import Blueprint
from factoriollm.helpers import (
    resolve_direction,
    add_belt_line,
    add_underground_pair,
    add_entity_row,
    add_power_poles,
)


class TestDraftsmanHelpers(unittest.TestCase):
    def test_resolve_direction(self):
        self.assertEqual(resolve_direction("north"), 0)
        self.assertEqual(resolve_direction("east"), 4)
        self.assertEqual(resolve_direction("south"), 8)
        self.assertEqual(resolve_direction("west"), 12)
        self.assertEqual(resolve_direction(4), 4)

    def test_add_belt_line(self):
        bp = Blueprint()
        add_belt_line(bp, start=(0, 0), length=10, direction="east", belt_type="transport-belt")
        self.assertEqual(len(bp.entities), 10)
        # Check that blueprint serializes cleanly
        bp_string = bp.to_string()
        self.assertTrue(bp_string.startswith("0e"))

    def test_add_underground_pair(self):
        bp = Blueprint()
        add_underground_pair(bp, start=(0, 0), end=(5, 0), direction="east", belt_type="underground-belt")
        self.assertEqual(len(bp.entities), 2)
        bp_string = bp.to_string()
        self.assertTrue(bp_string.startswith("0e"))

    def test_add_entity_row(self):
        bp = Blueprint()
        add_entity_row(bp, "stone-furnace", start=(0, 2), count=12, step=(3.0, 0.0))
        self.assertEqual(len(bp.entities), 12)
        bp_string = bp.to_string()
        self.assertTrue(bp_string.startswith("0e"))

    def test_add_power_poles(self):
        bp = Blueprint()
        add_power_poles(bp, start=(0, 0), count=5, step=(7.0, 0.0), pole_type="small-electric-pole")
        self.assertEqual(len(bp.entities), 5)
        bp_string = bp.to_string()
        self.assertTrue(bp_string.startswith("0e"))


if __name__ == "__main__":
    unittest.main()
