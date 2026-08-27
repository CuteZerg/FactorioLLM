from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity


def decompile_blueprint_to_code(blueprint_string: str) -> str:
    """
    Accepts a Factorio blueprint string and generates Python code
    (using Draftsman) that recreates the blueprint.
    """
    # Correct way to load in newer versions of Draftsman
    try:
        bp = Blueprint.from_string(blueprint_string)
    except AttributeError:
        # Fallback if version is older or this is a BlueprintBook
        print("Warning: This may be a BlueprintBook instead of a single blueprint.")
        bp = Blueprint(blueprint_string)
        
    script_lines = [
        "from draftsman.blueprintable import Blueprint",
        "from draftsman.entity import new_entity",
        "",
        f"# Original Blueprint: {bp.label or 'Unnamed'}",
        "bp = Blueprint()",
        ""
    ]
    
    # Iterate over all entities.
    for i, ent in enumerate(bp.entities):
        name = ent.name
        
        # 1. Safely extract the position as a regular dictionary.
        pos = {'x': ent.position['x'], 'y': ent.position['y']} if ent.position else {'x': 0, 'y': 0}
        
        # 2. Safely extract the direction.
        dir_str = ""
        if hasattr(ent, "direction") and ent.direction is not None:
            dir_val = ent.direction.value if hasattr(ent.direction, 'value') else ent.direction
            dir_str = f", direction={dir_val}"
            
        var_name = f"ent_{i}"
        script_lines.append(f"{var_name} = new_entity('{name}', position={pos}{dir_str})")
        
        # 3. Include the recipe when present (for assemblers).
        if hasattr(ent, 'recipe') and ent.recipe:
            script_lines.append(f"{var_name}.recipe = '{ent.recipe}'")
            
        script_lines.append(f"bp.entities.append({var_name})")
        script_lines.append("")  # Blank line
        
    return "\n".join(script_lines)

# --- TEST ---

TEST_BLUEPRINT_STRING = "0eNqdlttuwyAMQP/Fz6wKBnL7lWmaekEVUkoiQqZVVf59STqp3Qpt8FOUCA7GOdhcYNcMunPGeqgvYPat7aF+v0BvjnbbzN/s9qShBu+2tu9a5992uvEwMjD2oL+h5iMLDO+7xniv3d1AHD8YaOuNN/q6yPJy/rTDaTeNrDmLLMaga/tpWmvnFeY1UWwUgzPU1UaNcwD/UEhAcR5mCQoLwyxJYcmJxeBgnN5fB8gAWVHIKhxlTmHlYVZBYZVhVske9ApQ5JWRhRlVCqMIM3iWsCn53FR+s36Yzok7unZ6vqQtsjLw526e2Q6+G+ZD+YhHQqiLvX+M4xhiCwpbRvIgaXlQ93kwNpYGRQk1oiHPVzn0CxERSJECiRwuXiZsS70QsUr6ASpRRMwIoT6KGCp9yCnoiIeItDSs8xAFJdSIhyhXKZQ/rYeoUiCRgogpHSN/0boLAivWu7GkwCLNGysKTK6qpSKjsCP9W3AKLFJjBFJgi7LTfc94fZpm3m6ZDL6065c5KsdKSZQVL1QmcRx/AJpna08=" 

try:
    generated_python_code = decompile_blueprint_to_code(TEST_BLUEPRINT_STRING)
    print("=== GENERATED PYTHON CODE ===")
    print(generated_python_code)
    
    with open("decompiled_test.py", "w", encoding="utf-8") as f:
        f.write(generated_python_code)
        f.write("\n# Check the result\n")
        f.write("print(bp.to_string())\n")
        
    print("\nCode saved to decompiled_test.py. Run it!")
except (AttributeError, TypeError, ValueError) as e:
    print(f"Error while decoding: {e}")