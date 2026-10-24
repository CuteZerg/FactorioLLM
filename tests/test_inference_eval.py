"""
Manual GPU evaluation of the fine-tuned model (run only on request: loads the model on the GPU).
Generated code is untrusted and is executed exclusively inside the Docker sandbox.
"""

from factoriollm.blueprint_ir import extract_facts, parse_blueprint_string
from factoriollm.inference import FactorioInference, extract_python_code
from factoriollm.sandbox import DockerSandbox


def test_prompts():
    infer = FactorioInference()
    infer.load_model()
    sandbox = DockerSandbox()

    prompts = [
        "Линия выплавки меди на 24 каменные печи с красными конвейерами",
        "Create a compact copper smelting line for 24 stone furnaces using red belts and inserters",
        "Compact 4x4 belt balancer with splitters and undergrounds",
        "Steam power setup with 1 offshore pump, 20 boilers and 40 steam engines",
    ]

    for p in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: {p}")
        print("=" * 60)
        raw = infer.generate([{"role": "user", "content": p}], max_new_tokens=2048)
        code = extract_python_code(raw)
        print("GENERATED CODE:")
        print(code)
        print("-" * 60)
        res = sandbox.execute(code)
        facts = {}
        if res.success:
            try:
                facts = extract_facts(parse_blueprint_string(res.blueprint_string, allowed=None))
            except Exception as e:  # e.g. overlapping entities
                facts = {"error": str(e)}
        print(f"Stats: Lines={len(code.splitlines())}, Executes={res.success}, Error={res.error_message}")
        print(f"Facts: {facts}")


if __name__ == "__main__":
    test_prompts()
