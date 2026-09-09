from draftsman.blueprintable import Blueprint


def format_class_name(entity_name: str) -> str:
    """Converts 'underground-belt' to 'UndergroundBelt'"""
    return "".join(word.capitalize() for word in entity_name.split("-"))

def decompile_and_normalize(blueprint_string: str) -> str:
    try:
        bp = Blueprint.from_string(blueprint_string)
    except Exception:
        bp = Blueprint(blueprint_string)

    if not bp.entities:
        return "bp = Blueprint()"

    # 1. Normalize coordinates (find minimum x and y)
    min_x = min(ent.position['x'] for ent in bp.entities if ent.position)
    min_y = min(ent.position['y'] for ent in bp.entities if ent.position)

    # 2. Collect unique classes for import
    used_classes = set(format_class_name(ent.name) for ent in bp.entities)
    imports_str = ", ".join(used_classes)

    script_lines = [
        "from draftsman.blueprintable import Blueprint",
        f"from draftsman.entity import {imports_str}",
        "",
        f"# Blueprint: {bp.label or 'Normalized Blueprint'}",
        "bp = Blueprint()",
        ""
    ]

    # 3. Generate code with coordinate shift and ALL necessary attributes
    for i, ent in enumerate(bp.entities):
        class_name = format_class_name(ent.name)
        
        # Shift coordinates to zero-origin
        norm_x = ent.position['x'] - min_x
        norm_y = ent.position['y'] - min_y
        
        # Start building the keyword arguments for the entity
        kwargs = [f"position={{'x': {norm_x}, 'y': {norm_y}}}"]
        
        # Safely extract direction
        if hasattr(ent, "direction") and ent.direction is not None:
            dir_val = ent.direction.value if hasattr(ent.direction, 'value') else int(ent.direction)
            kwargs.append(f"direction={dir_val}")
            
        # BUGFIX: Extract io_type for Underground Belts and Pipes
        if hasattr(ent, "io_type") and ent.io_type is not None:
            io_val = ent.io_type.value if hasattr(ent.io_type, 'value') else str(ent.io_type)
            kwargs.append(f"io_type='{io_val}'")
            
        # Feature: Extract priorities and filters for Splitters
        if hasattr(ent, "input_priority") and ent.input_priority is not None:
            val = ent.input_priority.value if hasattr(ent.input_priority, 'value') else str(ent.input_priority)
            kwargs.append(f"input_priority='{val}'")
            
        if hasattr(ent, "output_priority") and ent.output_priority is not None:
            val = ent.output_priority.value if hasattr(ent.output_priority, 'value') else str(ent.output_priority)
            kwargs.append(f"output_priority='{val}'")

        # Join all arguments and create the append string
        kwargs_str = ", ".join(kwargs)
        script_lines.append(f"bp.entities.append({class_name}('{ent.name}', {kwargs_str}))")
    script_lines.append("print(bp.to_string())")  
        
    return "\n".join(script_lines)

# --- TESTING THE DECOMPILER ---
if __name__ == "__main__":
    # 4to4 balancer
    TEST_BLUEPRINT_STRING = "0eNqdltuOgyAQht+Fa9rIACq+ymaz6YE0JBYN4mZN47sv2k3atLDqXHlivhnGf2a4kWPd69YZ60l1I+bU2I5UHzfSmYs91NM7e7hqUhHvDrZrG+d3R117MlJi7Fn/kIqNNLK8a2vjvXZPC2H8pERbb7zRdyfzw/Bl++sxrKwYTTijpG26YNbYyUNA7cReUjKEGwZ7OU4RvLAAw8riLI5gqThKIFBFQFFyNk6f7t9FBCwR4DweY45AyTiqQKB4HFXSN2lFIH8/ksUZagNDxBEsW78lviRS9lB8H2rEXVwTrsu4bFaEH9rJsul9208F+Y4HRKzqVW0MYmiOQBeJLAhUFvLnJBibyoFEBJqQIMvX6IffGWWCUWxgJKqKlev3BIsaVFuyD1s1CBkiVrWm4wFDkBMSBEAlYZ0EgSMCTUgQxBr5wP9tEOQGSKIPwoYpwRaHdYGBJaY1lAhYYlyDQrCKVR2UZwh0YmJzhmAlmgsHBGtWazjdGa+vwfBxpqTkW7tuNpE5KClAKFbITMA4/gJph2Zz" 

    try:
        generated_python_code = decompile_and_normalize(TEST_BLUEPRINT_STRING)
        
        # Save to decompiled_test.py for AI refactoring pipeline
        with open("./src/factoriollm/decompiled_test.py", "w", encoding="utf-8") as f:
            f.write(generated_python_code)
            
        print("[*] SUCCESS! Code saved to decompiled_test.py")
        print("[*] Run the AI refactor script now to generate loops.")
        
    except Exception as e:
        print(f"[!] Decoding error: {e}")