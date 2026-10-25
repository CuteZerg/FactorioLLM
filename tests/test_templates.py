import unittest
import warnings
from collections import Counter

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.templates import Template, extract, extract_cell, find_period, metrics, tiling_check


def furnace_row(pairs: int, with_head: bool = True, lamp_at: int = -1) -> IRBlueprint:
    """
    A belt-fed stone furnace row repeating every 4 tiles (two furnaces, two inserters, one pole):
        row 0: input belt (east)   row 1: inserters + a pole   rows 2-3: furnaces
    The input belt starts one tile late (as real belt lines do) and a splitter head sits in front.
    """
    ents = []
    if with_head:
        ents += [IREntity("splitter", 0, 0, 4), IREntity("transport-belt", 1, 0, 4)]
    x0 = 2
    for k in range(pairs):
        u = x0 + 4 * k
        ents += [IREntity("inserter", u, 1, 0), IREntity("inserter", u + 2, 1, 0),
                 IREntity("small-electric-pole", u + 1, 1, 0),
                 IREntity("stone-furnace", u, 2, 0), IREntity("stone-furnace", u + 2, 2, 0)]
        ents += [IREntity("transport-belt", u + d, 0, 4) for d in range(4)]
    if lamp_at >= 0:
        ents.append(IREntity("small-lamp", x0 + 4 * lamp_at + 3, 1, 0))
    return IRBlueprint(ents)


class TestTemplates(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_period_is_found_despite_head_and_decor(self):
        self.assertEqual(find_period(furnace_row(5, lamp_at=2)), 4)

    def test_extract_cuts_head_segment_and_tail(self):
        t = extract(furnace_row(5), "smelting_stone")
        self.assertIsNotNone(t)
        self.assertEqual((t.period, t.repeats), (4, 5))
        self.assertEqual(Counter(e.name for e in t.segment)["stone-furnace"], 2)
        self.assertEqual({e.name for e in t.head}, {"splitter", "transport-belt"})

    def test_build_scales_the_design(self):
        t = extract(furnace_row(5), "smelting_stone")
        for n in (1, 3, 12):
            ir = t.build(n)
            self.assertEqual(Counter(e.name for e in ir.entities)["stone-furnace"], 2 * n)
            self.assertEqual(len({(e.x, e.y) for e in ir.entities}), len(ir.entities))
        self.assertTrue(tiling_check(t)["ok"])

    def test_decor_inside_the_repeat_is_dropped(self):
        t = extract(furnace_row(5, lamp_at=2), "smelting_stone")
        self.assertNotIn("small-lamp", {e.name for e in t.build(5).entities})

    def test_metrics(self):
        m = metrics(extract(furnace_row(5), "smelting_stone"))
        self.assertEqual(m["machines_per_segment"], {"stone-furnace": 2})
        self.assertEqual(m["segment_size"], [4, 4])
        self.assertEqual(m["tiles_per_machine"], 8.0)

    def test_cell_tiles_by_snap_pitch_and_shares_identical_entities(self):
        # a 5-wide cell with a pole on each edge: copies overlap by one column (snap 4)
        cell = IRBlueprint([IREntity("small-electric-pole", 0, 0), IREntity("solar-panel", 1, 0),
                            IREntity("small-electric-pole", 4, 0)], snap=(4, 3))
        t = extract_cell(cell, "solar")
        self.assertEqual(t.period, 4)
        built = t.build(3)
        self.assertEqual(Counter(e.name for e in built.entities), {"small-electric-pole": 4, "solar-panel": 3})

    def test_json_round_trip(self):
        t = extract(furnace_row(4), "smelting_stone")
        again = Template.from_json(t.to_json())
        self.assertEqual(again.build(3).canonical_set(), t.build(3).canonical_set())

    def test_no_repeat_no_template(self):
        self.assertIsNone(extract(furnace_row(1, with_head=False), "smelting_stone"))


if __name__ == "__main__":
    unittest.main()
