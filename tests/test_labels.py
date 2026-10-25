import unittest
import warnings

from draftsman.blueprintable import Blueprint

from factoriollm.blueprint_ir import BlueprintIRError, extract_facts, parse_blueprint_dict, parse_blueprint_string
from factoriollm.dataset_lifter import lift_and_verify_locally, lift_ir_to_code
from factoriollm.helpers import add_display_panel, add_signal_label


def combinator_20(entity_number=1, x=0.5, y=0.5):
    return {"entity_number": entity_number, "name": "constant-combinator", "position": {"x": x, "y": y}, "direction": 4,
            "control_behavior": {"sections": {"sections": [{"index": 1, "filters": [
                {"index": 1, "name": "iron-ore", "count": 1, "quality": "normal", "comparator": "="},
                {"index": 2, "name": "signal-A", "type": "virtual", "count": 5, "quality": "normal", "comparator": "="},
            ]}]}}}


BELT = {"entity_number": 2, "name": "transport-belt", "position": {"x": 1.5, "y": 0.5}, "direction": 4}


class TestLabelEntities(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_constant_combinator_signals_are_kept(self):
        ir = parse_blueprint_dict({"entities": [combinator_20(), BELT]})
        label = next(e for e in ir.entities if e.name == "constant-combinator")
        self.assertEqual(label.extras_dict["signals"], "item:iron-ore=1;virtual:signal-A=5")
        self.assertEqual(extract_facts(ir)["label_signals"], ["iron-ore", "signal-A"])

    def test_legacy_filters_are_read(self):
        legacy = {"entity_number": 1, "name": "constant-combinator", "position": {"x": 0.5, "y": 0.5},
                  "control_behavior": {"filters": [{"signal": {"type": "item", "name": "coal"}, "count": 1, "index": 1}]}}
        ir = parse_blueprint_dict({"entities": [legacy]}, version=(1, 1))
        self.assertEqual(ir.entities[0].extras_dict["signals"], "item:coal=1")

    def test_wired_combinators_are_logic_and_rejected(self):
        wired = {"entities": [combinator_20(), BELT], "wires": [[1, 1, 2, 1]]}
        with self.assertRaisesRegex(BlueprintIRError, "circuit-wired"):
            parse_blueprint_dict(wired)
        legacy = combinator_20()
        legacy["connections"] = {"1": {"red": [{"entity_id": 2}]}}
        with self.assertRaisesRegex(BlueprintIRError, "circuit-wired"):
            parse_blueprint_dict({"entities": [legacy, BELT]}, version=(1, 1))

    def test_power_wires_do_not_count_as_logic(self):
        ir = parse_blueprint_dict({"entities": [combinator_20(), BELT], "wires": [[1, 5, 2, 5]]})
        self.assertEqual(len(ir.entities), 2)

    def test_space_age_signals_are_rejected(self):
        cc = combinator_20()
        cc["control_behavior"]["sections"]["sections"][0]["filters"][0]["quality"] = "legendary"
        with self.assertRaisesRegex(BlueprintIRError, "label signal"):
            parse_blueprint_dict({"entities": [cc]})

    def test_display_panel_text_and_icon(self):
        panel = {"entity_number": 1, "name": "display-panel", "position": {"x": 0.5, "y": 0.5},
                 "text": "Iron\n[item=iron-plate]", "icon": {"type": "item", "name": "iron-plate"}}
        ir = parse_blueprint_dict({"entities": [panel]})
        self.assertEqual(ir.entities[0].extras_dict, {"text": "Iron [item=iron-plate]", "icon": "item:iron-plate"})

    def test_lifted_code_rebuilds_labels_exactly(self):
        panel = {"entity_number": 3, "name": "display-panel", "position": {"x": 0.5, "y": 2.5}, "text": "it's \"ore\"",
                 "icon": {"type": "virtual", "name": "signal-A"}}
        ir = parse_blueprint_dict({"entities": [combinator_20(), BELT, panel]})
        code = lift_and_verify_locally(ir)
        self.assertIsNotNone(code)
        self.assertIn("add_signal_label(bp, (0, 0), signals=['iron-ore', ('signal-A', 5, 'virtual')], direction='east')", code)
        self.assertIn("add_display_panel(bp, (0, 2), text='it\\'s \"ore\"', icon=('signal-A', 'virtual'))", code)

    def test_labels_are_not_folded_into_rows(self):
        entities = [combinator_20(i + 1, x=i + 0.5) for i in range(5)]
        code = lift_ir_to_code(parse_blueprint_dict({"entities": entities}))
        self.assertEqual(code.count("add_signal_label(bp,"), 5)
        self.assertNotIn("add_entity_row", code)

    def test_helpers_place_labels(self):
        bp = Blueprint()
        add_signal_label(bp, (0, 0), signals=["iron-ore", ("water", 1, "fluid")])
        add_display_panel(bp, (1, 0), text="Smelting", icon="iron-plate")
        ir = parse_blueprint_string(bp.to_string())
        extras = {e.name: e.extras_dict for e in ir.entities}
        self.assertEqual(extras["constant-combinator"]["signals"], "item:iron-ore=1;fluid:water=1")
        self.assertEqual(extras["display-panel"], {"text": "Smelting", "icon": "item:iron-plate"})
        with self.assertRaisesRegex(ValueError, "already occupied"):
            add_signal_label(bp, (0, 0), signals=["coal"])


if __name__ == "__main__":
    unittest.main()
