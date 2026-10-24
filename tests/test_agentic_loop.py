import unittest

from factoriollm.agentic_loop import AgenticBlueprintGenerator
from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.dataset_lifter import lift_ir_to_code, run_trusted_script
from factoriollm.sandbox import ExecutionResult

GOOD = IRBlueprint([IREntity("wooden-chest", 0, 0), IREntity("burner-inserter", 0, 1), IREntity("iron-chest", 0, 2)])
# inserter drops into a stone wall: compiles, but never works in game
BAD = IRBlueprint([IREntity("wooden-chest", 0, 0), IREntity("burner-inserter", 0, 1), IREntity("stone-wall", 0, 2)])


class FakeInference:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.histories = []

    def generate_stream(self, history, temperature=0.2):
        self.histories.append([dict(m) for m in history])
        yield self.scripts.pop(0)


class FakeSandbox:
    """Runs only lifter-generated (trusted) code, on the host."""
    timeout = 5.0

    def ensure_image(self):
        pass

    def execute(self, code, timeout=None):
        return ExecutionResult(success=True, blueprint_string=run_trusted_script(code))


def run(agent, prompt="test"):
    gen = agent.generate_blueprint_stream(prompt)
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return stop.value, events


class TestAgenticLoopValidation(unittest.TestCase):
    def test_validation_error_triggers_retry_with_report(self):
        inference = FakeInference([lift_ir_to_code(BAD), lift_ir_to_code(GOOD)])
        result, events = run(AgenticBlueprintGenerator(inference=inference, sandbox=FakeSandbox(), max_retries=3))
        self.assertTrue(result.success)
        self.assertEqual(result.attempts_used, 2)
        retry_prompt = inference.histories[1][-1]["content"]
        self.assertIn("cannot work in Factorio", retry_prompt)
        self.assertIn("[inserter_target]", retry_prompt)
        self.assertIn("retry", [e.event_type for e in events])

    def test_validation_can_be_disabled(self):
        inference = FakeInference([lift_ir_to_code(BAD)])
        result, _ = run(AgenticBlueprintGenerator(inference=inference, sandbox=FakeSandbox(), validate=False))
        self.assertTrue(result.success)
        self.assertIsNone(result.validation)

    def test_warnings_do_not_block_success(self):
        lone = IRBlueprint([IREntity("transport-belt", 0, 0, 4), IREntity("underground-belt", 1, 0, 4)])
        result, _ = run(AgenticBlueprintGenerator(inference=FakeInference([lift_ir_to_code(lone)]), sandbox=FakeSandbox()))
        self.assertTrue(result.success)
        self.assertIn("lone_underground", result.validation.category_counts())


if __name__ == "__main__":
    unittest.main()
