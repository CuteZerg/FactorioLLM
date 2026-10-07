import unittest
from factoriollm.inference import is_code_complete, is_stuck_in_repetition


class TestCodeCompletion(unittest.TestCase):
    def test_complete_code(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt

bp = Blueprint()
for x in range(10):
    bp.entities.append(TransportBelt('transport-belt', position={'x': float(x), 'y': 0.0}))
print(bp.to_string())
"""
        complete, reason = is_code_complete(code)
        self.assertTrue(complete)
        self.assertEqual(reason, "Complete")

    def test_truncated_code_unclosed_dict(self):
        code = """
from draftsman.blueprintable import Blueprint
from draftsman.entity import TransportBelt

bp = Blueprint()
bp.entities.append(TransportBelt('transport-belt', position={'x': 46
"""
        complete, reason = is_code_complete(code)
        self.assertFalse(complete)
        self.assertIn("to_string()", reason)

    def test_truncated_code_with_to_string_unclosed(self):
        code = """
from draftsman.blueprintable import Blueprint
bp = Blueprint()
bp.entities.append(TransportBelt('transport-belt', position={'x': 46
print(bp.to_string())
"""
        complete, reason = is_code_complete(code)
        self.assertFalse(complete)
        self.assertIn("Incomplete syntax", reason)

    def test_missing_blueprint_instantiation(self):
        code = """
print(bp.to_string())
"""
        complete, reason = is_code_complete(code)
        self.assertFalse(complete)
        self.assertIn("Blueprint()", reason)

    def test_empty_code(self):
        complete, reason = is_code_complete("")
        self.assertFalse(complete)
        self.assertIn("empty", reason)

    def test_repetition_detection(self):
        repeated_text = "\n".join([
            f"bp.entities.append(TransportBelt('transport-belt', position={{'x': {i}.0, 'y': 7.0}}))"
            for i in range(110)
        ])
        self.assertTrue(is_stuck_in_repetition(repeated_text))

        short_text = "bp.entities.append(TransportBelt('transport-belt'))"
        self.assertFalse(is_stuck_in_repetition(short_text))


if __name__ == "__main__":
    unittest.main()
