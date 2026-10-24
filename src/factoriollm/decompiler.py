"""
Decompiler: blueprint string -> flat, normalized draftsman script.

Parsing goes through the tile-grid IR (blueprint_ir.py): positions are shifted by an integer
tile offset so every entity keeps its alignment, and all values are validated / repr()-escaped
before they are embedded into code.
"""

from factoriollm.blueprint_ir import parse_blueprint_string
from factoriollm.dataset_lifter import lift_ir_to_code


def decompile_and_normalize(blueprint_string: str) -> str:
    """Returns a flat script (one append per entity) that rebuilds the blueprint at tile origin (0, 0)."""
    ir = parse_blueprint_string(blueprint_string, allowed=None)
    return lift_ir_to_code(ir, use_macros=False)


# --- TESTING THE DECOMPILER ---
if __name__ == "__main__":
    # 4to4 balancer
    TEST_BLUEPRINT_STRING = "0eNqdltuOgyAQht+Fa9rIACq+ymaz6YE0JBYN4mZN47sv2k3atLDqXHlivhnGf2a4kWPd69YZ60l1I+bU2I5UHzfSmYs91NM7e7hqUhHvDrZrG+d3R117MlJi7Fn/kIqNNLK8a2vjvXZPC2H8pERbb7zRdyfzw/Bl++sxrKwYTTijpG26YNbYyUNA7cReUjKEGwZ7OU4RvLAAw8riLI5gqThKIFBFQFFyNk6f7t9FBCwR4DweY45AyTiqQKB4HFXSN2lFIH8/ksUZagNDxBEsW78lviRS9lB8H2rEXVwTrsu4bFaEH9rJsul9208F+Y4HRKzqVW0MYmiOQBeJLAhUFvLnJBibyoFEBJqQIMvX6IffGWWCUWxgJKqKlev3BIsaVFuyD1s1CBkiVrWm4wFDkBMSBEAlYZ0EgSMCTUgQxBr5wP9tEOQGSKIPwoYpwRaHdYGBJaY1lAhYYlyDQrCKVR2UZwh0YmJzhmAlmgsHBGtWazjdGa+vwfBxpqTkW7tuNpE5KClAKFbITMA4/gJph2Zz"
    print(decompile_and_normalize(TEST_BLUEPRINT_STRING))
