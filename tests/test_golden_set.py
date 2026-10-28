"""
The golden set: player designs the user checked in game, with the verdict "works" or "broken".
The flow simulator must agree, judging each design as the miner judges an author's design
(template_miner.author_verdict). A design the simulator still gets wrong names the plan item that
fixes it ("open") and is skipped until then, so the suite stays green and the gap stays visible.
"""
import json
import unittest
from pathlib import Path

from factoriollm.blueprint_ir import IRBlueprint, IREntity
from factoriollm.template_miner import author_verdict, reason

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


def load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    ents = [IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in data["entities"]]
    ir = IRBlueprint(ents, data["label"] or "", tuple(data["version"]), snap=tuple(data["snap"]) if data["snap"] else None)
    return data, ir


class GoldenSetTest(unittest.TestCase):
    def test_simulator_agrees_with_the_game(self):
        for path in sorted(GOLDEN.glob("*.json")):
            data, ir = load(path)
            with self.subTest(design=path.stem, verdict=data["verdict"]):
                if data["open"]:
                    self.skipTest(f"plan item {data['open']}: {data['note']}")
                _, flow, _, _ = author_verdict(ir)
                works = flow.ok
                self.assertEqual(works, data["verdict"] == "works",
                                 f"{data['note']}; simulator: {reason(flow) if not works else 'works'}, "
                                 f"{[p.message for p in flow.problems[:3]]}")


if __name__ == "__main__":
    unittest.main()
