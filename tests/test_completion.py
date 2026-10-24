import unittest
from factoriollm.inference import fit_history_to_budget, is_code_complete, is_stuck_in_repetition


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
            "bp.entities.append(TransportBelt('transport-belt', position={'x': 1.0, 'y': 1.0}))"
            for _ in range(12)
        ])
        self.assertTrue(is_stuck_in_repetition(repeated_text, consecutive_threshold=8))

        non_repeating_text = "\n".join([
            f"bp.entities.append(TransportBelt('transport-belt', position={{'x': {i}.0, 'y': 1.0}}))"
            for i in range(12)
        ])
        self.assertFalse(is_stuck_in_repetition(non_repeating_text, consecutive_threshold=8))


def count_words(messages):
    return sum(len(m["content"].split()) for m in messages)


class TestHistoryBudget(unittest.TestCase):
    def setUp(self):
        self.old_code = "\n".join(f"bp.entities.append('inserter', tile_position=({i}, 0))" for i in range(300))
        self.history = [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "Build 24 stone furnaces"},
            {"role": "assistant", "content": self.old_code},
            {"role": "user", "content": "Traceback: ValueError: bad thing\nPlease fix the script."},
        ]

    def test_fits_unchanged(self):
        self.assertEqual(fit_history_to_budget(self.history, count_words, 10_000), self.history)

    def test_shrinks_old_code_but_keeps_request_and_error(self):
        out = fit_history_to_budget(self.history, count_words, 400)
        self.assertLessEqual(count_words(out), 400)
        self.assertEqual(out[1]["content"], "Build 24 stone furnaces")
        self.assertEqual(out[3]["content"], self.history[3]["content"])
        self.assertIn("lines omitted", out[2]["content"])
        self.assertIn("tile_position=(0, 0)", out[2]["content"])    # head kept
        self.assertIn("tile_position=(299, 0)", out[2]["content"])  # tail kept
        self.assertEqual(self.history[2]["content"], self.old_code)  # input not mutated

    def test_raises_when_impossible(self):
        with self.assertRaises(ValueError):
            fit_history_to_budget(self.history, count_words, 5)


if __name__ == "__main__":
    unittest.main()
