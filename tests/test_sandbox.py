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


if __name__ == "__main__":
    unittest.main()

