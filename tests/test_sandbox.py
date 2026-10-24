import subprocess
import unittest
from factoriollm.sandbox import DockerSandbox


class TestDockerSandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = DockerSandbox(timeout=5.0)
        cls.sandbox.ensure_image()

    def test_sandbox_success(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import Container

bp = Blueprint()
bp.entities.append(Container("wooden-chest", tile_position=(0, 0)))
print(bp.to_string())
"""
        res = self.sandbox.execute(code)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.blueprint_string)
        self.assertTrue(res.blueprint_string.startswith("0e"))

    def test_sandbox_syntax_error(self):
        code = """
def broken(:
    pass
"""
        res = self.sandbox.execute(code)
        self.assertFalse(res.success)
        self.assertTrue("SyntaxError" in res.stderr or "SyntaxError" in (res.error_message or ""))

    def test_sandbox_draftsman_runtime_error(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import Container

bp = Blueprint()
# Passing invalid tile_position format
bp.entities.append(Container("wooden-chest", tile_position="invalid_pos"))
print(bp.to_string())
"""
        res = self.sandbox.execute(code)
        self.assertFalse(res.success)
        feedback = res.get_feedback_for_model()
        self.assertIn("Traceback", feedback)

    def test_sandbox_timeout(self):
        code = """
import time
while True:
    time.sleep(0.1)
"""
        short_sb = DockerSandbox(timeout=2.0)
        res = short_sb.execute(code)
        self.assertFalse(res.success)
        self.assertIn("TimeoutExpired", res.error_message or "")

    def test_sandbox_synthetic_classes(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import StoneFurnace, SmallLamp, MediumElectricPole, LongHandedInserter

bp = Blueprint()
bp.entities.append(StoneFurnace('stone-furnace', position={'x': 1.0, 'y': 2.0}))
bp.entities.append(MediumElectricPole('medium-electric-pole', position={'x': 3.0, 'y': 2.0}))
bp.entities.append(LongHandedInserter('long-handed-inserter', position={'x': 2.0, 'y': 2.0}))
bp.entities.append(SmallLamp('small-lamp', position={'x': 4.0, 'y': 2.0}))
print(bp.to_string())
"""
        res = self.sandbox.execute(code)
        self.assertTrue(res.success)
        self.assertIsNotNone(res.blueprint_string)
        self.assertTrue(res.blueprint_string.startswith("0e"))

    def test_sandbox_network_isolated(self):
        code = """
import urllib.request
try:
    urllib.request.urlopen("http://1.1.1.1", timeout=2)
    print("NETWORK_ACCESSIBLE")
except Exception as e:
    print(f"NETWORK_BLOCKED: {e}")
"""
        res = self.sandbox.execute(code)
        self.assertIn("NETWORK_BLOCKED", res.stdout)
        self.assertNotIn("NETWORK_ACCESSIBLE", res.stdout)


    def test_sandbox_helpers_macro(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman_helpers import add_belt_line, add_entity_row, add_power_poles

bp = Blueprint()
add_belt_line(bp, start=(0, 0), length=16, direction="east", belt_type="fast-transport-belt")
add_entity_row(bp, "stone-furnace", start=(0, 2), count=8, step=(2, 0))
add_power_poles(bp, start=(0, 4), count=4, step=(7, 0))
print(bp.to_string())
"""
        res = self.sandbox.execute(code)
        self.assertTrue(res.success, f"Execution failed: {res.error_message}\n{res.stderr}")
        self.assertIsNotNone(res.blueprint_string)
        self.assertTrue(res.blueprint_string.startswith("0e"))

    def test_traceback_line_numbers_match_generated_code(self):
        code = "from draftsman.blueprintable import Blueprint\nbp = Blueprint()\nraise ValueError('boom')\n"
        res = self.sandbox.execute(code)
        self.assertFalse(res.success)
        self.assertIn('File "script.py", line 3', res.stderr)

    def test_timeout_kills_container(self):
        res = DockerSandbox(timeout=3.0).execute("while True:\n    pass\n")
        self.assertIn("TimeoutExpired", res.error_message or "")
        running = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=factoriollm-"], capture_output=True, text=True
        ).stdout.strip()
        self.assertEqual(running, "")

    def test_filesystem_is_read_only(self):
        res = self.sandbox.execute("open('/home/sandbox/x.txt', 'w').write('x')\n")
        self.assertFalse(res.success)
        self.assertIn("Read-only file system", res.stderr)

    def test_output_is_capped(self):
        sandbox = DockerSandbox(max_output_bytes=10_000)
        res = sandbox.execute("print('A' * 5_000_000)\n")
        self.assertLessEqual(len(res.stdout), 10_000)

    def test_execute_batch_isolates_failures(self):
        ok = """
from draftsman.blueprintable import Blueprint
bp = Blueprint()
bp.entities.append("wooden-chest", tile_position=(0, 0))
print(bp.to_string())
"""
        results = self.sandbox.execute_batch([ok, "raise RuntimeError('x')", "while True:\n    pass\n", ok], per_script_timeout=3)
        self.assertEqual([r.success for r in results], [True, False, False, True])
        self.assertIn("RuntimeError", results[1].stderr)


if __name__ == "__main__":
    unittest.main()

