import json
import tempfile
import unittest
from pathlib import Path

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.dataset_lifter import lift_ir_to_code, run_trusted_script
from factoriollm.eval_suite import EvalCase, load_set_a, load_set_b, safe_validate, score_case, summarize
from factoriollm.sandbox import ExecutionResult


def smelter(n_furnaces, belt="fast-transport-belt"):
    ents = [IREntity(belt, x, 0, 4) for x in range(2 * n_furnaces)]
    for k in range(n_furnaces):
        ents += [IREntity("burner-inserter", 2 * k, 1), IREntity("stone-furnace", 2 * k, 2)]
    return IRBlueprint(ents)


def executed(ir):
    code = lift_ir_to_code(ir)
    bp = run_trusted_script(code)
    return code, ExecutionResult(success=True, blueprint_string=bp), safe_validate(bp)


class TestEvalSuite(unittest.TestCase):
    def test_set_a_loads(self):
        cases = load_set_a()
        self.assertGreaterEqual(len(cases), 40)
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertTrue(any(c.expect.get("counts", {}).get("stone-furnace") == 24 for c in cases))

    def test_score_exact_match(self):
        case = EvalCase("x", "A", "24 stone furnaces, red belts", {"counts": {"stone-furnace": 4}, "belt_tier": "red"})
        result = score_case(case, *executed(smelter(4)))
        self.assertTrue(result["complete"] and result["exec_ok"] and result["valid"])
        self.assertTrue(result["count_checks"][0]["exact"])
        self.assertTrue(result["belt_tier_ok"])

    def test_score_wrong_count_and_tier(self):
        case = EvalCase("x", "A", "p", {"counts": {"stone-furnace": 24}, "belt_tier": "blue", "contains": ["lab"]})
        result = score_case(case, *executed(smelter(4, belt="transport-belt")))
        self.assertFalse(result["count_checks"][0]["exact"])
        self.assertFalse(result["count_checks"][0]["within10"])
        self.assertFalse(result["belt_tier_ok"])
        self.assertFalse(result["contains_ok"])

    def test_failed_execution(self):
        case = EvalCase("x", "A", "p", {"counts": {"stone-furnace": 4}})
        result = score_case(case, "print(", ExecutionResult(success=False, error_message="boom"), None)
        self.assertFalse(result["exec_ok"] or result["valid"] or result["complete"])
        self.assertEqual(result["count_checks"][0]["actual"], 0)

    def test_summarize(self):
        ok = score_case(EvalCase("a", "A", "p", {"counts": {"stone-furnace": 4}}), *executed(smelter(4)))
        bad = score_case(EvalCase("b", "B", "p", {"counts": {"stone-furnace": 9}}), *executed(smelter(4)))
        summary = summarize([ok, bad])
        self.assertEqual(summary["all"]["n"], 2)
        self.assertEqual(summary["all"]["count_exact"], 0.5)
        self.assertEqual(summary["A"]["count_exact"], 1.0)
        self.assertEqual(summary["B"]["count_exact"], 0.0)

    def test_set_b_reference_expectations(self):
        record = {"messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "smelter with 4 stone furnaces"},
            {"role": "assistant", "content": lift_ir_to_code(smelter(4))},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            cases = load_set_b(path, limit=5)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].expect["counts"], {"stone-furnace": 4})
        self.assertEqual(cases[0].expect["belt_tier"], "red")
        self.assertIn("warnings_per_100", cases[0].reference)


if __name__ == "__main__":
    unittest.main()
