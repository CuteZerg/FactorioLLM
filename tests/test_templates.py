import unittest
import warnings
from collections import Counter

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.templates import (
    Template,
    add_port_labels,
    extract,
    extract_cell,
    find_period,
    has_weaving,
    main_belt,
    metrics,
    prepare,
    prune_poles,
    repair_power,
    retier,
    tiling_check,
    underground_pairs,
)

OUT = (("io_type", "output"),)


def ug(name, x, output=False, y=0):
    return IREntity(name, x, y, 4, OUT if output else ())


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

    def test_lamps_are_an_optional_layer(self):
        t = extract(furnace_row(5, lamp_at=2), "smelting_stone")
        self.assertEqual(Counter(e.name for e in t.build(4).entities)["small-lamp"], 0)
        self.assertEqual(Counter(e.name for e in t.build(4, lamps=True).entities)["small-lamp"], 4)
        again = Template.from_json(t.to_json())
        self.assertEqual(again.build(2, lamps=True).canonical_set(), t.build(2, lamps=True).canonical_set())

    def test_own_grid_instead_of_the_authors(self):
        row = furnace_row(5)
        t = extract(IRBlueprint(row.entities, snap=(7, 5), absolute_snap=True, snap_offset=(3, 1)), "smelting_stone")
        self.assertEqual(t.snap, (4, 4))
        segment = t.build_segment()
        self.assertEqual((segment.snap, segment.absolute_snap), ((4, 4), False))
        self.assertEqual(min(e.x for e in segment.entities), 0)
        self.assertIsNone(t.build(3).snap)  # a finished build is not dragged


class TestSelfContained(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_prepare_cuts_surroundings_and_author_parts(self):
        row = furnace_row(4)
        tap = [IREntity("transport-belt", 10, y, 8) for y in range(-12, 0)]  # a bus tap coming from far above
        cut_pair = [IREntity("underground-belt", 20, 0, 0), IREntity("underground-belt", 20, -4, 0, OUT)]
        extra = [IREntity("radar", 40, 0), IREntity("constant-combinator", 1, 1, 0, (("signals", "item:coal=1"),)),
                 IREntity("small-lamp", 5, 1)]
        ir = prepare(IRBlueprint(row.entities + tap + cut_pair + extra))
        names = Counter(e.name for e in ir.entities)
        self.assertEqual((names["radar"], names["constant-combinator"], names["small-lamp"]), (0, 0, 1))
        self.assertEqual(names["underground-belt"], 0)  # its exit was cut, so the entrance goes too
        self.assertEqual(min(e.y for e in ir.entities if e.name == "transport-belt"), -2)

    def test_opposite_same_tier_undergrounds_block(self):
        east = [IREntity("underground-belt", 0, 0, 4), IREntity("underground-belt", 4, 0, 4, OUT)]
        west = [IREntity("underground-belt", 2, 0, 12), IREntity("underground-belt", 1, 0, 12, OUT)]
        self.assertEqual(underground_pairs(east), {0: 1})
        self.assertNotIn(0, underground_pairs(east + west))  # a yellow pair going the other way in between
        mixed = retier(east + [IREntity("fast-underground-belt", 2, 0, 12), IREntity("fast-underground-belt", 1, 0, 12, OUT)],
                       "transport-belt")
        self.assertEqual(len({e.name for e in mixed}), 2)  # retier keeps the two directions on two tiers

    def test_weave_between_copies_keeps_its_tiers(self):
        # a fast pair from one copy to the next passes over the yellow pair of the next copy
        seg = [ug("underground-belt", 0, True), ug("underground-belt", 1), ug("fast-underground-belt", 2, True),
               ug("fast-underground-belt", 3), IREntity("transport-belt", 0, 1, 4)]
        t = Template("labs", 4, [], seg, [], 1)
        t.belt = "transport-belt"
        for n in (1, 3):
            names = {e.name for e in t.build(n).entities if "underground" in e.name}
            self.assertEqual(names, {"underground-belt", "fast-underground-belt"}, n)
        built = t.build(3).entities
        self.assertIn((3, 6), {(built[i].x, built[j].x) for i, j in underground_pairs(built).items()})

    def test_mixed_belt_labels_name_the_lane(self):
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).parent / "fixtures" / "flow" / "processing_units_mixed_belt.json").read_text())
        ir = IRBlueprint([IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]])
        signals = [e.extras_dict["signals"] for e in add_port_labels(ir).entities if e.name == "constant-combinator"]
        self.assertTrue(any("electronic-circuit" in s and "signal-" in s for s in signals), signals)
        self.assertTrue(any("processing-unit" in s and "signal-" in s for s in signals), signals)

    def test_underground_across_the_line_does_not_block(self):
        north = [IREntity("underground-belt", 0, 5, 0), IREntity("underground-belt", 0, 1, 0, OUT),
                 IREntity("underground-belt", 0, 3, 4)]  # a crossing line goes underground there
        self.assertEqual(underground_pairs(north), {0: 1})

    def test_strip_unused_keeps_feeds_and_drops_a_passing_bus(self):
        from factoriollm.templates import strip_unused
        gear = IREntity("assembling-machine-1", 0, 2, 0, (("recipe", "iron-gear-wheel"),))
        feed = [IREntity("transport-belt", x, 0, 4) for x in range(4)] + [IREntity("inserter", 1, 1, 0)]
        second = [IREntity("transport-belt", x, 6, 12) for x in range(4)] + [IREntity("inserter", 1, 5, 8)]
        bus = [IREntity("transport-belt", x, -2, 4) for x in range(4)]
        kept = strip_unused(IRBlueprint([gear] + feed + second + bus)).entities
        self.assertEqual(sum(e.y == -2 for e in kept), 0)  # nobody takes from the passing bus
        self.assertEqual(sum(e.y == 6 and "belt" in e.name for e in kept), 4)  # a second feed is not spare

    def test_recipe_modules_only_where_they_fit(self):
        from factoriollm.templates import normalize_modules
        def am(name, x, r, modules):
            return IREntity(name, x, 0, 0, (("modules", modules), ("recipe", r)) if modules else (("recipe", r),))
        ents = [am("assembling-machine-3", 3 * i, "electronic-circuit", "speed-module-3=4") for i in range(3)] + \
            [am("assembling-machine-3", 9, "iron-gear-wheel", "productivity-module-3=4"),
             am("assembling-machine-2", 12, "iron-gear-wheel", "")]
        t = normalize_modules(Template("multi", 15, [], ents, [], 1))
        am2 = next(e for e in t.segment if e.name == "assembling-machine-2")
        self.assertNotIn("modules", am2.extras_dict)  # 4 modules do not fit its 2 slots

    def test_feeder_poles_are_pruned(self):
        lab = IREntity("lab", 0, 0)
        near = IREntity("medium-electric-pole", 3, 1)
        feeder = [IREntity("medium-electric-pole", 3 + 8 * k, 1) for k in range(1, 4)]
        kept = prune_poles([lab, near] + feeder)
        self.assertEqual([e for e in kept if "pole" in e.name], [near])

    def test_retier_brings_every_belt_to_one_tier(self):
        ents = furnace_row(3).entities + [IREntity("fast-transport-belt", 14, 0, 4)]
        self.assertEqual(main_belt(ents), "transport-belt")
        names = {e.name for e in retier(ents, "express-transport-belt") if "belt" in e.name or "splitter" in e.name}
        self.assertEqual(names, {"express-transport-belt", "express-splitter"})

    def test_retier_keeps_belt_weaving(self):
        # a red pair (span 6) over a yellow pair (span 2)
        weave = [ug("fast-underground-belt", 0), ug("underground-belt", 2), ug("underground-belt", 4, True),
                 ug("fast-underground-belt", 6, True)]
        for belt in ("transport-belt", "fast-transport-belt", "express-transport-belt"):
            out = retier(weave, belt)
            self.assertEqual(underground_pairs(out), underground_pairs(weave), belt)
            self.assertNotEqual(out[0].name, out[1].name, belt)  # the woven pairs differ in tier

    def test_retier_handles_long_underground_chains(self):
        # ten yellow pairs in a row on one line, plus a lone entrance: neighbours, not weaves
        chain = [ug("underground-belt", 6 * k + d, output=d == 4) for k in range(10) for d in (0, 4)]
        chain.append(ug("underground-belt", 60))
        for belt in ("fast-transport-belt", "express-transport-belt"):
            out = retier(chain, belt)
            self.assertEqual(underground_pairs(out), underground_pairs(chain), belt)
            self.assertEqual({e.name for e in out}, {belt.replace("transport", "underground")})

    def test_retier_raises_the_tier_of_a_span_too_long(self):
        long_pair = [ug("express-underground-belt", 0), ug("express-underground-belt", 8, True)]
        self.assertEqual({e.name for e in retier(long_pair, "transport-belt")}, {"express-underground-belt"})
        short = [ug("express-underground-belt", 0), ug("express-underground-belt", 3, True)]
        self.assertEqual({e.name for e in retier(short, "transport-belt")}, {"underground-belt"})
        with self.assertRaises(ValueError):
            retier(short, "underground-belt")

    def test_port_labels_from_recipes(self):
        gears = [IREntity("assembling-machine-2", 3 * i, 2, 0, (("recipe", "iron-gear-wheel"),)) for i in range(2)]
        belts = [IREntity("transport-belt", x, y, 4) for x in range(6) for y in (0, 6)]
        inserters = [IREntity("inserter", 3 * i + 1, y, 0) for i in range(2) for y in (1, 5)]
        ir = add_port_labels(IRBlueprint(gears + belts + inserters))
        labels = {(e.x, e.y): e.extras_dict["signals"] for e in ir.entities if e.name == "constant-combinator"}
        self.assertEqual(labels, {(-1, 0): "item:iron-plate=1", (6, 6): "item:iron-gear-wheel=1"})

    def test_power_is_strict_and_repaired(self):
        unpowered = IRBlueprint([e for e in furnace_row(5).entities if e.name != "small-electric-pole"])
        t = extract(unpowered, "smelting_stone")
        check = tiling_check(t)
        self.assertFalse(check["ok"])
        self.assertTrue(check["power_only"])
        fixed = repair_power(t)
        self.assertIsNotNone(fixed)
        self.assertTrue(tiling_check(fixed)["ok"])
        self.assertIn("medium-electric-pole", {e.name for e in fixed.segment})  # no pole in the design: the default

    def test_weaving_is_detected(self):
        weave = [ug("fast-underground-belt", 0), ug("underground-belt", 2), ug("underground-belt", 4, True),
                 ug("fast-underground-belt", 6, True)]
        self.assertTrue(has_weaving(weave))
        self.assertFalse(has_weaving([ug("underground-belt", 0), ug("underground-belt", 4, True)]))

    def test_one_label_when_lanes_do_not_matter(self):
        # iron plates (steel) and coal on the two lanes of the input belt; the furnaces take both from
        # either lane, so the lanes do not matter: one label with both items
        ir = add_port_labels(furnace_row(3), smelt="iron-plate")
        labels = sorted(e.extras_dict["signals"] for e in ir.entities if e.name == "constant-combinator")
        self.assertEqual(labels, ["item:coal=1;item:iron-plate=1"])
        inside = add_port_labels(furnace_row(3), inside_only=True)
        self.assertTrue(all(e.x >= 0 and e.y >= 0 for e in inside.entities))

    def test_crop_keeps_the_underground_copies_join_at(self):
        # a yellow pair weaves from one copy of the cell into the next: in one copy its exit has no
        # entrance, but it is the author's joint, not surroundings to cut
        import json
        from pathlib import Path
        from factoriollm.templates import prepare
        data = json.loads((Path(__file__).parent / "fixtures" / "flow" / "engines_weave_cell.json").read_text())
        ir = IRBlueprint([IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]])
        lone = [(e.x, e.y) for e in ir.entities if e.name == "underground-belt" and e.extras_dict.get("io_type") == "output"]
        kept = {(e.x, e.y) for e in prepare(ir, 2).entities}
        self.assertTrue(lone and all(t in kept for t in lone), lone)

    def test_filter_inserters_as_signs(self):
        from factoriollm.templates import author_signs, prepare
        # a filter inserter right next to a belt start, picking from empty ground: the author's sign
        sign = IREntity("fast-inserter", 0, 1, 8, (("filters", "iron-plate"),))  # picks from (0, 2), drops on (0, 0)
        belt = [IREntity("transport-belt", x, 0, 4) for x in range(4)]
        # one loading a chest from outside (a wagon at a station) is no sign
        loader = IREntity("fast-inserter", 6, 1, 8, (("filters", "coal"),))
        chest = IREntity("wooden-chest", 6, 0)
        ir = IRBlueprint(belt + [sign, loader, chest])
        signs = author_signs(ir)
        self.assertEqual([(e, tile, set(items)) for e, tile, items in signs], [(sign, (0, 0), {"iron-plate"})])
        kept = prepare(ir, margin=None).entities
        self.assertNotIn(sign, kept)
        self.assertIn(loader, kept)

    def test_spare_semi_finished_goods_get_no_label(self):
        # gears are made for the belt machine; the gears it leaves on the output belt are no product
        ir = IRBlueprint([IREntity("transport-belt", x, y, 4) for x in range(8) for y in (0, 6)] + [
            IREntity("inserter", 1, 1, 0), IREntity("assembling-machine-2", 0, 2, 0, (("recipe", "iron-gear-wheel"),)),
            IREntity("inserter", 1, 5, 0), IREntity("inserter", 5, 1, 0), IREntity("inserter", 4, 5, 8),
            IREntity("assembling-machine-2", 4, 2, 0, (("recipe", "transport-belt"),)), IREntity("inserter", 6, 5, 0)])
        labels = [e.extras_dict["signals"] for e in add_port_labels(ir).entities if e.name == "constant-combinator"]
        self.assertTrue(any("item:transport-belt" in s for s in labels), labels)
        self.assertFalse(any("iron-gear-wheel" in s for s in labels), labels)

    def test_pass_through_belt_is_not_mixed(self):
        # the coal belt runs past the boilers on to the next block: nothing is added to it, so it is
        # labelled once at its start, without L / R (boilers hand coal on only to the next boiler)
        import json
        from pathlib import Path
        data = json.loads((Path(__file__).parent / "fixtures" / "flow" / "steam_coal_pass_through.json").read_text())
        ir = IRBlueprint([IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]])
        labels = [e.extras_dict["signals"] for e in add_port_labels(ir).entities if e.name == "constant-combinator"]
        self.assertEqual([s for s in labels if s.startswith("item:")], ["item:coal=1"])


if __name__ == "__main__":
    unittest.main()
