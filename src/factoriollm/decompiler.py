from draftsman.blueprintable import Blueprint
from draftsman.entity import new_entity


def format_class_name(entity_name: str) -> str:
    """Превращает 'underground-belt' в 'UndergroundBelt'"""
    return "".join(word.capitalize() for word in entity_name.split("-"))

def decompile_and_normalize(blueprint_string: str) -> str:
    try:
        bp = Blueprint.from_string(blueprint_string)
    except Exception:
        print(Exception)
        bp = Blueprint()

    if not bp.entities:
        return "bp = Blueprint()"

    # 1. НОРМАЛИЗАЦИЯ КООРДИНАТ (ищем минимальные x и y)
    min_x = min(ent.position['x'] for ent in bp.entities if ent.position)
    min_y = min(ent.position['y'] for ent in bp.entities if ent.position)

    # 2. Собираем уникальные классы для импорта
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

    # 3. Генерируем код со сдвигом к нулю
    for i, ent in enumerate(bp.entities):
        class_name = format_class_name(ent.name)
        
        # Сдвигаем координаты!
        norm_x = ent.position['x'] - min_x
        norm_y = ent.position['y'] - min_y
        
        dir_str = ""
        if hasattr(ent, "direction") and ent.direction is not None:
            dir_val = ent.direction.value if hasattr(ent.direction, 'value') else int(ent.direction)
            dir_str = f", direction={dir_val}"
            
        script_lines.append(
            f"bp.entities.append({class_name}('{ent.name}', position={{'x': {norm_x}, 'y': {norm_y}}}{dir_str}))"
        )
        
    return "\n".join(script_lines)

# Запусти этот код с той же строкой балансировщика!

# --- TEST ---

TEST_BLUEPRINT_STRING = "0eNqdlttuwyAMQP/Fz6wKBnL7lWmaekEVUkoiQqZVVf59STqp3Qpt8FOUCA7GOdhcYNcMunPGeqgvYPat7aF+v0BvjnbbzN/s9qShBu+2tu9a5992uvEwMjD2oL+h5iMLDO+7xniv3d1AHD8YaOuNN/q6yPJy/rTDaTeNrDmLLMaga/tpWmvnFeY1UWwUgzPU1UaNcwD/UEhAcR5mCQoLwyxJYcmJxeBgnN5fB8gAWVHIKhxlTmHlYVZBYZVhVske9ApQ5JWRhRlVCqMIM3iWsCn53FR+s36Yzok7unZ6vqQtsjLw526e2Q6+G+ZD+YhHQqiLvX+M4xhiCwpbRvIgaXlQ93kwNpYGRQk1oiHPVzn0CxERSJECiRwuXiZsS70QsUr6ASpRRMwIoT6KGCp9yCnoiIeItDSs8xAFJdSIhyhXKZQ/rYeoUiCRgogpHSN/0boLAivWu7GkwCLNGysKTK6qpSKjsCP9W3AKLFJjBFJgi7LTfc94fZpm3m6ZDL6065c5KsdKSZQVL1QmcRx/AJpna08=" 

try:
    generated_python_code = decompile_and_normalize(TEST_BLUEPRINT_STRING)
    print("=== GENERATED PYTHON CODE ===")
    print(generated_python_code)
    
    with open("decompiled_test.py", "w", encoding="utf-8") as f:
        f.write(generated_python_code)
        f.write("\n# Check the result\n")
        f.write("print(bp.to_string())\n")
        
    print("\nCode saved to decompiled_test.py. Run it!")
except (AttributeError, TypeError, ValueError) as e:
    print(f"Error while decoding: {e}")