import base64
import json
import unittest
import zlib
from collections import Counter

from factoriollm import blueprint_ir as ir
from factoriollm.dataset_lifter import ir_equivalent, run_trusted_script
from factoriollm.decompiler import decompile_and_normalize

# 4-to-4 balancer (Factorio 2.x) with 1x1 belts, 2x1 splitters and underground belts
BALANCER = "0eNqdltuOgyAQht+Fa9rIACq+ymaz6YE0JBYN4mZN47sv2k3atLDqXHlivhnGf2a4kWPd69YZ60l1I+bU2I5UHzfSmYs91NM7e7hqUhHvDrZrG+d3R117MlJi7Fn/kIqNNLK8a2vjvXZPC2H8pERbb7zRdyfzw/Bl++sxrKwYTTijpG26YNbYyUNA7cReUjKEGwZ7OU4RvLAAw8riLI5gqThKIFBFQFFyNk6f7t9FBCwR4DweY45AyTiqQKB4HFXSN2lFIH8/ksUZagNDxBEsW78lviRS9lB8H2rEXVwTrsu4bFaEH9rJsul9208F+Y4HRKzqVW0MYmiOQBeJLAhUFvLnJBibyoFEBJqQIMvX6IffGWWCUWxgJKqKlev3BIsaVFuyD1s1CBkiVrWm4wFDkBMSBEAlYZ0EgSMCTUgQxBr5wP9tEOQGSKIPwoYpwRaHdYGBJaY1lAhYYlyDQrCKVR2UZwh0YmJzhmAlmgsHBGtWazjdGa+vwfBxpqTkW7tuNpE5KClAKFbITMA4/gJph2Zz"


def encode(blueprint: dict) -> str:
    return "0" + base64.b64encode(zlib.compress(json.dumps(blueprint).encode())).decode()


def v1_1(entities, label="test"):
    return {"blueprint": {"item": "blueprint", "label": label, "entities": entities, "version": (1 << 48) | (1 << 32)}}


class TestDecompiler(unittest.TestCase):
    def test_balancer_roundtrip_keeps_exact_tile_layout(self):
        original = ir.parse_blueprint_string(BALANCER)
        rebuilt = run_trusted_script(decompile_and_normalize(BALANCER))
        self.assertTrue(ir_equivalent(original, rebuilt))

    def test_no_tile_is_covered_twice(self):
        tiles = Counter()
        for e in ir.parse_blueprint_string(BALANCER).entities:
            w, h = ir.footprint(e.name, e.direction)
            tiles.update((e.x + dx, e.y + dy) for dx in range(w) for dy in range(h))
        self.assertEqual(max(tiles.values()), 1)

    def test_normalized_to_origin_on_integer_grid(self):
        b = ir.parse_blueprint_string(BALANCER)
        self.assertEqual(min(e.x for e in b.entities), 0)
        self.assertEqual(min(e.y for e in b.entities), 0)

    def test_legacy_1_1_directions_and_renames(self):
        s = encode(v1_1([
            {"entity_number": 1, "name": "transport-belt", "position": {"x": 0.5, "y": 0.5}, "direction": 2},
            {"entity_number": 2, "name": "stack-inserter", "position": {"x": 1.5, "y": 0.5}, "direction": 4},
            {"entity_number": 3, "name": "stone-furnace", "position": {"x": 3, "y": 1}},
        ]))
        b = ir.parse_blueprint_string(s)
        self.assertIn(ir.IREntity("transport-belt", 0, 0, 4), b.entities)   # 1.1 east (2) -> 2.0 east (4)
        self.assertIn(ir.IREntity("bulk-inserter", 1, 0, 8), b.entities)    # renamed in 2.0
        self.assertIn(ir.IREntity("stone-furnace", 2, 0, 0), b.entities)

    def test_label_injection_is_not_executed(self):
        s = encode(v1_1(
            [{"entity_number": 1, "name": "wooden-chest", "position": {"x": 0.5, "y": 0.5}}],
            label="x\nimport os; os.system('echo pwned')",
        ))
        code = decompile_and_normalize(s)
        self.assertNotIn("os.system", code)

    def test_malicious_entity_name_rejected(self):
        s = encode(v1_1([{"entity_number": 1, "name": "x'),__import__('os').system('calc'),('", "position": {"x": 0.5, "y": 0.5}}]))
        with self.assertRaises(ir.BlueprintIRError):
            decompile_and_normalize(s)

    def test_malicious_recipe_rejected(self):
        s = encode(v1_1([{"entity_number": 1, "name": "assembling-machine-2", "position": {"x": 1.5, "y": 1.5}, "recipe": "a')\nimport os#"}]))
        with self.assertRaises(ir.BlueprintIRError):
            decompile_and_normalize(s)

    def test_zip_bomb_rejected(self):
        bomb = "0" + base64.b64encode(zlib.compress(b"[" + b" " * (ir.MAX_DECOMPRESSED_BYTES + 10) + b"]")).decode()
        with self.assertRaises(ir.BlueprintIRError):
            ir.decode_blueprint_string(bomb)

    def test_space_age_entity_not_allowed_by_default(self):
        s = encode({"blueprint": {"entities": [
            {"entity_number": 1, "name": "turbo-transport-belt", "position": {"x": 0.5, "y": 0.5}}
        ], "version": 2 << 48}})
        with self.assertRaises(ir.BlueprintIRError):
            ir.parse_blueprint_string(s)
        self.assertEqual(len(ir.parse_blueprint_string(s, allowed=None).entities), 1)

    def test_book_children_are_iterated(self):
        child = v1_1([{"entity_number": 1, "name": "wooden-chest", "position": {"x": 0.5, "y": 0.5}}])
        book = {"blueprint_book": {"blueprints": [dict(child, index=0), dict(child, index=3)], "version": 1 << 48}}
        self.assertEqual([path for path, _, _ in ir.iter_blueprints(book)], ["0", "3"])

    def test_rotation_four_times_is_identity_and_facts(self):
        b = ir.parse_blueprint_string(BALANCER)
        r = b
        for _ in range(4):
            r = ir.rotated(r, 1)
        self.assertEqual(r.canonical_set(), b.canonical_set())
        r1 = ir.rotated(b, 1)
        self.assertEqual(r1.size, (b.size[1], b.size[0]))
        facts = ir.extract_facts(b)
        self.assertEqual(facts["entity_counts"]["splitter"], 6)
        self.assertEqual(facts["belt_tiers"], {"yellow": facts["total_entities"]})


if __name__ == "__main__":
    unittest.main()
