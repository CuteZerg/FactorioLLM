import unittest

from factoriollm.export_dataset import _stable_fraction, rotate_direction_words
from factoriollm.prompt_generator import allowed_numbers, is_grounded, strip_rich_text


class TestExportHelpers(unittest.TestCase):
    def test_rotate_english_directions(self):
        self.assertEqual(rotate_direction_words("Belt flows East, output to the north", 1), "Belt flows South, output to the east")
        self.assertEqual(rotate_direction_words("west side", 2), "east side")
        self.assertEqual(rotate_direction_words("no directions here", 3), "no directions here")

    def test_rotate_russian_directions_once(self):
        self.assertEqual(rotate_direction_words("Конвейер идёт на запад", 1), "Конвейер идёт на север")
        # adjective must be rotated exactly once (западн -> северн, not further to восточн)
        self.assertEqual(rotate_direction_words("западная сторона", 1), "северная сторона")
        self.assertEqual(rotate_direction_words("Южный вход", 1), "Западный вход")

    def test_stable_split_is_deterministic(self):
        self.assertEqual(_stable_fraction("abc"), _stable_fraction("abc"))
        self.assertTrue(0.0 <= _stable_fraction("xyz") <= 1.0)

    def test_grounded_numbers(self):
        facts = {"entity_counts": {"stone-furnace": 24, "inserter": 48}, "width": 9, "height": 26, "total_entities": 150}
        numbers = allowed_numbers(facts, label="Copper smelting", title="")
        self.assertTrue(is_grounded("Smelter with 24 stone furnaces and 48 inserters, 9x26", numbers))
        self.assertFalse(is_grounded("Smelter with 30 furnaces", numbers))
        self.assertTrue(is_grounded("Плавильня на 24 печи", numbers))

    def test_rich_text_numbers_not_allowed(self):
        numbers = allowed_numbers({"entity_counts": {"solar-panel": 104}}, label="[color=255,255,102]Small 6 MW[/color]", title="")
        self.assertIn(6, numbers)
        self.assertNotIn(255, numbers)
        self.assertEqual(strip_rich_text("[item=steam-engine] Early Game [/color]"), "Early Game")


if __name__ == "__main__":
    unittest.main()
