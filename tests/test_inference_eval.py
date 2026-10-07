import sys
from factoriollm.inference import FactorioInference, extract_python_code
from factoriollm.dataset_lifter import execute_script_to_blueprint

def test_prompts():
    infer = FactorioInference()
    infer.load_model()

    prompts = [
        "Линия выплавки меди на 24 каменные печи с красными конвейерами",
        "Compact 4x4 belt balancer with splitters and undergrounds",
        "Steam power setup with 1 offshore pump, 20 boilers and 40 steam engines",
    ]

    for p in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: {p}")
        print("=" * 60)
        raw = infer.generate([{"role": "user", "content": p}], max_new_tokens=1024)
        code = extract_python_code(raw)
        print("GENERATED CODE:")
        print(code)
        print("-" * 60)
        lines = len(code.splitlines())
        has_print = "print(bp.to_string())" in code
        bp = execute_script_to_blueprint(code)
        valid = bp is not None
        ent_count = len(bp.entities) if bp else 0
        print(f"Stats: Lines={lines}, HasPrint={has_print}, Compiles={valid}, Entities={ent_count}")

if __name__ == "__main__":
    test_prompts()
