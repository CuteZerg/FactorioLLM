import unittest

from factoriollm.export_dataset import _stable_fraction, rotate_direction_words
from factoriollm.prompt_generator import (
    allowed_numbers,
    count_conflicts,
    is_grounded,
    normalize_prompt,
    prompt_language,
    strip_rich_text,
)


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



class TestPromptCleanup(unittest.TestCase):
    def test_normalize_prototype_ids(self):
        self.assertEqual(normalize_prompt("Build  10 assembling-machine-2 units"), "Build 10 assembling machine 2 units")
        self.assertEqual(normalize_prompt("Gate with 120 stone-walls"), "Gate with 120 stone walls")
        self.assertEqual(normalize_prompt("red/green circuit setup"), "red/green circuit setup")

    def test_prompt_language(self):
        self.assertEqual(prompt_language("Участок с 2 помпами"), "ru")
        self.assertEqual(prompt_language("Solar farm"), "en")

    def test_consistent_counts_pass(self):
        facts = {"entity_counts": {"stone-furnace": 48, "inserter": 96, "transport-belt": 162}}
        self.assertEqual(count_conflicts("Need an early game setup with 48 stone furnaces for iron", facts), [])
        self.assertEqual(count_conflicts("Плавильня на 48 каменных печах", facts), [])

    def test_absent_entity_with_number_conflicts(self):
        facts = {"entity_counts": {"burner-mining-drill": 14, "assembling-machine-1": 2, "inserter": 2}}
        self.assertTrue(count_conflicts("Early setup with 14 burner mining drills and 2 stone furnaces.", facts))

    def test_wrong_count_conflicts(self):
        facts = {"entity_counts": {"stone-furnace": 24, "inserter": 48}}
        self.assertTrue(count_conflicts("Smelter with 48 stone furnaces", facts))

    def test_size_and_label_numbers_are_ignored(self):
        facts = {"entity_counts": {"small-lamp": 157, "stone-wall": 96}}
        self.assertEqual(count_conflicts("Need a 13 by 18 lamp sign for letter Q", facts), [])
        self.assertEqual(count_conflicts("T1 straight wall featuring 96 stone walls.", facts), [])

    def test_ratios_tiers_and_nearest_noun(self):
        facts = {"entity_counts": {"assembling-machine-2": 15, "substation": 4, "stone-wall": 172, "gun-turret": 11,
                                   "fast-splitter": 1, "iron-chest": 6, "inserter": 12}}
        self.assertEqual(count_conflicts("Схема с 15 сборщиками и подстанциями", facts), [])
        self.assertEqual(count_conflicts("Level 2 line wall with 11 gun turrets and 172 stone walls", facts), [])
        self.assertEqual(count_conflicts("1 to 2 red belt splitter", facts), [])
        self.assertEqual(count_conflicts("Buffer layout with 6 iron chests and regular inserters", facts), [])
        self.assertEqual(count_conflicts("Setup with 15 assembling machine 2", facts), [])

    def test_balancer_lanes_and_tier_suffixes(self):
        facts = {"entity_counts": {"express-transport-belt": 16, "express-splitter": 7, "assembling-machine-3": 34,
                                   "flamethrower-turret": 14, "laser-turret": 43, "roboport": 1}}
        self.assertEqual(count_conflicts("Balanced split-off for 4 blue belts with 7 express splitters", facts), [])
        self.assertEqual(count_conflicts("Space science block with 34 assemblers-3 and beacons", facts), [])
        self.assertEqual(count_conflicts("Балансировщик 4 + 1 на синих лентах", facts), [])
        self.assertEqual(count_conflicts("Рядный балансировщик 4 к 4 для синих лент", facts), [])
        self.assertEqual(count_conflicts("Оборона с 43 лазерными турелями, 14 огнеметными и дронстанцией", facts), [])
        self.assertTrue(count_conflicts("Main bus with 67 blue belts", facts))

    def test_units_after_number_are_ignored(self):
        facts = {"entity_counts": {"nuclear-reactor": 2, "stone-wall": 240, "assembling-machine-3": 12}}
        self.assertEqual(count_conflicts("Build a 160 MW nuclear reactor", facts), [])
        self.assertEqual(count_conflicts("Curved 90 degree wall with 240 stone walls", facts), [])
        self.assertEqual(count_conflicts("Looking for a 16 lane red belt balancer", facts), [])
        self.assertEqual(count_conflicts("Поставь 12 сборщиков 3 для каменных стен", facts), [])

    def test_russian_na_before_count_is_checked(self):
        facts = {"entity_counts": {"stone-furnace": 24}}
        self.assertTrue(count_conflicts("Плавильня на 48 каменных печах", facts))
        self.assertTrue(count_conflicts("17 chemical plants for cracking", {"entity_counts": {"chemical-plant": 18}}))

    def test_assembler_recipe_count_allowed(self):
        facts = {"entity_counts": {"assembling-machine-2": 30}, "recipes": {"copper-cable": 18, "electronic-circuit": 12}}
        self.assertEqual(count_conflicts("Electronic circuits with 12 electronic circuit assemblers", facts), [])


if __name__ == "__main__":
    unittest.main()
