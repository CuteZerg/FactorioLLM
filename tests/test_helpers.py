import unittest
from collections import Counter

from draftsman.blueprintable import Blueprint
from factoriollm.helpers import (
    resolve_direction,
    add_belt_line,
    add_underground_pair,
    add_entity_row,
    add_power_poles,
)


def occupied_tiles(bp):
    """Counts how many entities cover each tile."""
    tiles = Counter()
    for e in bp.entities:
        x0, y0 = e.tile_position.x, e.tile_position.y
        for dx in range(e.tile_width):
            for dy in range(e.tile_height):
                tiles[(x0 + dx, y0 + dy)] += 1
    return tiles


class TestDraftsmanHelpers(unittest.TestCase):
    def test_resolve_direction(self):
        self.assertEqual(resolve_direction("north"), 0)
        self.assertEqual(resolve_direction("east"), 4)
        self.assertEqual(resolve_direction("south"), 8)
        self.assertEqual(resolve_direction("west"), 12)
        self.assertEqual(resolve_direction(4), 4)

    def test_resolve_direction_rejects_garbage(self):
        with self.assertRaises(ValueError):
            resolve_direction("sideways")
        with self.assertRaises(ValueError):
            resolve_direction(3)

    def test_add_belt_line_occupies_consecutive_tiles(self):
        bp = Blueprint()
        add_belt_line(bp, start=(0, 0), length=10, direction="east", belt_type="transport-belt")
        self.assertEqual(len(bp.entities), 10)
        self.assertEqual([tuple(e.tile_position) for e in bp.entities], [(x, 0) for x in range(10)])
        self.assertTrue(bp.to_string().startswith("0e"))

    def test_add_belt_line_west(self):
        bp = Blueprint()
        add_belt_line(bp, start=(5, 2), length=3, direction="west")
        self.assertEqual([tuple(e.tile_position) for e in bp.entities], [(5, 2), (4, 2), (3, 2)])

    def test_add_underground_pair(self):
        bp = Blueprint()
        add_underground_pair(bp, start=(0, 0), end=(5, 0), direction="east", belt_type="underground-belt")
        self.assertEqual(len(bp.entities), 2)
        self.assertEqual([e.io_type for e in bp.entities], ["input", "output"])
        self.assertTrue(bp.to_string().startswith("0e"))

    def test_add_entity_row_touching_furnaces_do_not_overlap(self):
        bp = Blueprint()
        add_entity_row(bp, "stone-furnace", start=(0, 2), count=12, step=(2, 0))
        self.assertEqual(len(bp.entities), 12)
        self.assertEqual(max(occupied_tiles(bp).values()), 1)
        self.assertTrue(bp.to_string().startswith("0e"))

    def test_mixed_sizes_share_one_grid(self):
        # belt row above a furnace row: no tile may be covered twice
        bp = Blueprint()
        add_belt_line(bp, start=(0, 0), length=8, direction="east")
        add_entity_row(bp, "stone-furnace", start=(0, 1), count=4, step=(2, 0))
        add_entity_row(bp, "inserter", start=(0, 3), count=8, step=(1, 0), direction="south")
        self.assertEqual(max(occupied_tiles(bp).values()), 1)

    def test_add_entity_row_passes_recipe(self):
        bp = Blueprint()
        add_entity_row(bp, "assembling-machine-2", start=(0, 0), count=3, step=(3, 0), recipe="iron-gear-wheel")
        self.assertEqual({e.recipe for e in bp.entities}, {"iron-gear-wheel"})

    def test_unknown_entity_suggests_names(self):
        bp = Blueprint()
        with self.assertRaises(ValueError) as ctx:
            add_entity_row(bp, "stone-furnance", start=(0, 0), count=1)
        self.assertIn("stone-furnace", str(ctx.exception))

    def test_non_integer_tiles_rejected(self):
        bp = Blueprint()
        with self.assertRaises(ValueError):
            add_belt_line(bp, start=(0.5, 0), length=2)

    def test_add_power_poles(self):
        bp = Blueprint()
        add_power_poles(bp, start=(0, 0), count=5, step=(7, 0), pole_type="small-electric-pole")
        self.assertEqual(len(bp.entities), 5)
        self.assertTrue(bp.to_string().startswith("0e"))


if __name__ == "__main__":
    unittest.main()
