"""
Manual GPU evaluation of the fine-tuned model (run only on request: loads the model on the GPU).
Thin wrapper over factoriollm.eval_suite; model code is executed only inside the Docker sandbox.

    uv run python tests/test_inference_eval.py [adapter_dir]
"""

import sys

from factoriollm.eval_suite import run_eval


def test_prompts(adapter=None):
    run_eval(adapter, sets="A")


if __name__ == "__main__":
    test_prompts(sys.argv[1] if len(sys.argv) > 1 else None)
