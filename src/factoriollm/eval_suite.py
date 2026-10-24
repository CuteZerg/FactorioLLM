"""
Reproducible evaluation of a fine-tuned adapter.

Set A: fixed EN/RU prompts with expected facts (eval_prompts.json), e.g. "24 stone furnaces, red belts".
Set B: deterministic sample of the held-out split (data/dataset_v4_eval.jsonl); expected facts come
       from the reference blueprint (counts of key entities, belt tier, validator warning rate).

Generation is greedy and without retries (this measures the model, not the agentic loop).
Model code runs only inside the Docker sandbox; the resulting blueprints are checked by validator.py.

    uv run python -m factoriollm.eval_suite --adapter adapters/<run> [--set A|B|all] [--limit N]
    uv run python -m factoriollm.eval_suite --compare outputs/eval/a.json outputs/eval/b.json
    uv run python -m factoriollm.eval_suite --build-reference   # warning rates of real blueprints (DB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from factoriollm.blueprint_ir import entity_type
from factoriollm.inference import is_code_complete
from factoriollm.sandbox import ExecutionResult
from factoriollm.validator import Issue, ValidationReport, validate_string

PROMPTS_PATH = Path(__file__).resolve().parent / "eval_prompts.json"
EVAL_DATASET = Path("data/dataset_v4_eval.jsonl")
OUTPUT_DIR = Path("outputs/eval")
REFERENCE_RATES = OUTPUT_DIR / "reference_rates.json"

# Entity types whose counts define "what was asked for" in set B.
KEY_TYPES = frozenset({
    "furnace", "assembling-machine", "lab", "solar-panel", "accumulator", "boiler", "generator",
    "mining-drill", "beacon", "reactor", "roboport", "ammo-turret", "electric-turret", "fluid-turret",
    "offshore-pump", "rocket-silo",
})


@dataclass
class EvalCase:
    id: str
    set: str
    prompt: str
    expect: Dict[str, Any]
    reference: Dict[str, Any] = field(default_factory=dict)


# --- Loading ------------------------------------------------------------------------------

def load_set_a(path: Path = PROMPTS_PATH) -> List[EvalCase]:
    return [EvalCase(c["id"], "A", c["prompt"], c["expect"]) for c in json.loads(path.read_text(encoding="utf-8"))]


def dominant_belt_tier(facts: Dict[str, Any]) -> Optional[str]:
    tiers = facts.get("belt_tiers") or {}
    return max(tiers, key=tiers.get) if tiers else None


def expectations_from_reference(report: ValidationReport) -> Dict[str, Any]:
    facts = report.facts
    counts = {n: c for n, c in facts.get("entity_counts", {}).items() if entity_type(n) in KEY_TYPES}
    expect: Dict[str, Any] = {"counts": counts}
    tier = dominant_belt_tier(facts)
    if tier:
        expect["belt_tier"] = tier
    if not counts:
        expect["contains"] = list(facts.get("entity_counts", {}))[:3]
    return expect


def load_set_b(path: Path = EVAL_DATASET, limit: int = 100) -> List[EvalCase]:
    """Deterministic sample of the held-out split; reference facts come from executing the reference
    code, which is lifter-generated (trusted) code from our own pipeline."""
    from factoriollm.dataset_lifter import run_trusted_script

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda r: hashlib.sha256(r["messages"][1]["content"].encode("utf-8")).hexdigest())
    cases = []
    for r in records[:limit]:
        prompt, code = r["messages"][1]["content"], r["messages"][2]["content"]
        report = validate_string(run_trusted_script(code))
        case_id = "B-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        cases.append(EvalCase(case_id, "B", prompt, expectations_from_reference(report),
                              {"warnings_per_100": warnings_per_100(report)}))
    return cases


# --- Scoring (pure) -----------------------------------------------------------------------

def warnings_per_100(report: ValidationReport) -> float:
    total = report.facts.get("total_entities") or 0
    return 100.0 * len(report.warnings) / max(total, 1)


def safe_validate(blueprint_string: Optional[str]) -> Optional[ValidationReport]:
    if not blueprint_string:
        return None
    try:
        return validate_string(blueprint_string)
    except Exception as e:
        return ValidationReport(errors=[Issue("unparseable", str(e)[:200], (0, 0))])


def score_case(case: EvalCase, code: str, exec_result: ExecutionResult, report: Optional[ValidationReport],
               gen_tokens: int = 0, gen_seconds: float = 0.0) -> Dict[str, Any]:
    expect = case.expect
    facts = report.facts if report else {}
    counts = facts.get("entity_counts", {})
    present = set(counts)
    result: Dict[str, Any] = {
        "id": case.id,
        "set": case.set,
        "prompt": case.prompt,
        "complete": is_code_complete(code)[0],
        "exec_ok": bool(exec_result.success),
        "error": None if exec_result.success else (exec_result.error_message or "")[:300],
        "valid": bool(exec_result.success and report is not None and report.ok),
        "issues": report.category_counts() if report else {},
        "total_entities": facts.get("total_entities", 0),
        "gen_tokens": gen_tokens,
        "gen_seconds": round(gen_seconds, 2),
        "count_checks": [],
    }
    for name, expected in (expect.get("counts") or {}).items():
        actual = counts.get(name, 0)
        result["count_checks"].append({
            "entity": name, "expected": expected, "actual": actual,
            "exact": actual == expected, "within10": abs(actual - expected) <= max(1, 0.1 * expected),
        })
    recipes = facts.get("recipes") or {}
    for recipe, expected in (expect.get("recipes") or {}).items():
        result["count_checks"].append({
            "entity": f"recipe:{recipe}", "expected": expected, "actual": recipes.get(recipe, 0),
            "exact": recipes.get(recipe, 0) == expected, "within10": abs(recipes.get(recipe, 0) - expected) <= max(1, 0.1 * expected),
        })
    if "belt_tier" in expect:
        result["belt_tier_ok"] = dominant_belt_tier(facts) == expect["belt_tier"]
    contains_checks = []
    if expect.get("contains"):
        contains_checks.append(set(expect["contains"]) <= present)
    if expect.get("contains_any"):
        contains_checks.append(bool(set(expect["contains_any"]) & present))
    if expect.get("recipes_any"):
        contains_checks.append(bool(set(expect["recipes_any"]) & set(recipes)))
    if contains_checks:
        result["contains_ok"] = all(contains_checks)
    if report is not None:
        result["warnings_per_100"] = round(warnings_per_100(report), 2)
    if case.reference:
        result["reference_warnings_per_100"] = round(case.reference["warnings_per_100"], 2)
    return result


def _rate(values: List[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def block(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        checks = [c for r in rows for c in r["count_checks"]]
        issue_totals: Dict[str, int] = {}
        for r in rows:
            for cat, n in r["issues"].items():
                issue_totals[cat] = issue_totals.get(cat, 0) + n
        warn = [r["warnings_per_100"] for r in rows if "warnings_per_100" in r]
        ref = [r["reference_warnings_per_100"] for r in rows if "reference_warnings_per_100" in r and "warnings_per_100" in r]
        return {
            "n": len(rows),
            "complete": _rate([r["complete"] for r in rows]),
            "exec_ok": _rate([r["exec_ok"] for r in rows]),
            "valid": _rate([r["valid"] for r in rows]),
            "count_exact": _rate([c["exact"] for c in checks]),
            "count_within10": _rate([c["within10"] for c in checks]),
            "belt_tier_ok": _rate([r["belt_tier_ok"] for r in rows if "belt_tier_ok" in r]),
            "contains_ok": _rate([r["contains_ok"] for r in rows if "contains_ok" in r]),
            "median_warnings_per_100": round(statistics.median(warn), 2) if warn else None,
            "median_reference_warnings_per_100": round(statistics.median(ref), 2) if ref else None,
            "issue_totals": dict(sorted(issue_totals.items(), key=lambda kv: -kv[1])),
            "mean_gen_tokens": round(statistics.mean(r["gen_tokens"] for r in rows), 1) if rows else None,
            "mean_gen_seconds": round(statistics.mean(r["gen_seconds"] for r in rows), 2) if rows else None,
            "mean_attempts": round(statistics.mean(r.get("attempts", 1) for r in rows), 2) if rows else None,
            "valid_first_attempt": _rate([r.get("valid_first_attempt", r["valid"]) for r in rows]),
        }

    summary = {"all": block(results)}
    for s in sorted({r["set"] for r in results}):
        summary[s] = block([r for r in results if r["set"] == s])
    return summary


# --- Running ------------------------------------------------------------------------------

def run_eval(adapter: Optional[str], sets: str = "all", limit: Optional[int] = None, max_new_tokens: int = 3500,
             retries: int = 1) -> Dict[str, Any]:
    """retries=1 measures the model alone; retries>1 runs the full agentic loop (sandbox + validator feedback)."""
    from factoriollm.inference import FactorioInference, extract_python_code

    cases: List[EvalCase] = []
    if sets in ("A", "all"):
        cases += load_set_a()
    if sets in ("B", "all"):
        cases += load_set_b(limit=limit or 100)
    if limit is not None and sets == "A":
        cases = cases[:limit]

    infer = FactorioInference(model_path=adapter)
    infer.load_model()
    print(f"[*] Adapter: {infer.model_path}; {len(cases)} cases")

    agent = None
    if retries > 1:
        from factoriollm.agentic_loop import AgenticBlueprintGenerator
        agent = AgenticBlueprintGenerator(inference=infer, max_retries=retries)

    codes, stats = [], []
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        if agent is None:
            raw = infer.generate([{"role": "user", "content": case.prompt}], temperature=0.0, max_new_tokens=max_new_tokens)
            code, attempts, first_ok = extract_python_code(raw), 1, None
        else:
            res = agent.generate_blueprint(case.prompt, max_retries=retries, temperature=0.0)
            code, attempts = res.code or "", res.attempts_used
            first_ok = bool(res.execution_history) and res.execution_history[0].success
        seconds = time.perf_counter() - start
        codes.append(code)
        tokens = len(infer.tokenizer(code, add_special_tokens=False)["input_ids"])
        stats.append((tokens, seconds, attempts, first_ok))
        print(f"    [{i}/{len(cases)}] {case.id}: {tokens} tokens, {seconds:.1f}s, attempts={attempts}")

    return _execute_score_and_save(infer.model_path, sets, cases, codes, stats, extra={"retries": retries})


def _execute_score_and_save(adapter: str, sets: str, cases: List[EvalCase], codes: List[str], stats: List[tuple],
                            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs the generated codes in the sandbox, scores them and writes the JSON report."""
    from factoriollm.sandbox import DockerSandbox

    executions = DockerSandbox().execute_batch(codes, per_script_timeout=20)
    results = []
    for case, code, exec_result, (tokens, seconds, *loop) in zip(cases, codes, executions, stats):
        report = safe_validate(exec_result.blueprint_string) if exec_result.success else None
        result = score_case(case, code, exec_result, report, tokens, seconds)
        result["code"] = code
        if loop:
            result["attempts"] = loop[0]
            if loop[1] is not None:  # agentic run: was the first attempt already valid?
                result["valid_first_attempt"] = loop[1]
        results.append(result)

    report = {
        "adapter": adapter,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sets": sets,
        **(extra or {}),
        "summary": summarize(results),
        "results": results,
    }
    if REFERENCE_RATES.exists():
        report["reference_rates"] = json.loads(REFERENCE_RATES.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{Path(adapter).name}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(report)
    print(f"[*] Report: {out}")
    return report


def rescore(path: str) -> Dict[str, Any]:
    """Re-executes and re-scores the codes of an existing report (no GPU), e.g. after validator changes."""
    old = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {c.id: c for c in load_set_a()}
    n_b = sum(r["set"] == "B" for r in old["results"])
    if n_b:
        known.update({c.id: c for c in load_set_b(limit=n_b)})
    rows = [r for r in old["results"] if r["id"] in known]
    if len(rows) != len(old["results"]):
        print(f"[!] {len(old['results']) - len(rows)} cases no longer exist and are skipped")
    cases = [known[r["id"]] for r in rows]
    stats = [(r["gen_tokens"], r["gen_seconds"], r.get("attempts", 1), r.get("valid_first_attempt")) for r in rows]
    return _execute_score_and_save(old["adapter"], old["sets"], cases, [r["code"] for r in rows], stats,
                                   extra={"rescored_from": str(path)})


async def _build_reference_rates() -> Dict[str, Any]:
    from factoriollm.blueprint_ir import IRBlueprint, IREntity
    from factoriollm.database import get_connection
    from factoriollm.validator import validate

    conn = await get_connection()
    rows = await conn.fetch("SELECT ir FROM canonical_blueprints WHERE status = 'ok'")
    await conn.close()
    rates, per_category = [], {}
    for r in rows:
        ents = [IREntity(n, x, y, d, tuple(sorted(ex.items()))) for n, x, y, d, ex in json.loads(r["ir"])]
        rep = validate(IRBlueprint(ents))
        rates.append(warnings_per_100(rep))
        for cat, n in rep.category_counts().items():
            per_category[cat] = per_category.get(cat, 0) + n
    return {"blueprints": len(rows), "median_warnings_per_100": round(statistics.median(rates), 2),
            "mean_warnings_per_100": round(statistics.mean(rates), 2), "issue_totals": per_category}


def print_summary(report: Dict[str, Any]) -> None:
    keys = ["n", "complete", "exec_ok", "valid", "valid_first_attempt", "mean_attempts", "count_exact", "count_within10", "belt_tier_ok", "contains_ok",
            "median_warnings_per_100", "median_reference_warnings_per_100", "mean_gen_tokens", "mean_gen_seconds"]
    summary = report["summary"]
    cols = list(summary)
    print(f"\n{'metric':36}" + "".join(f"{c:>10}" for c in cols))
    for k in keys:
        print(f"{k:36}" + "".join(f"{str(summary[c].get(k)):>10}" for c in cols))
    if "reference_rates" in report:
        print(f"{'real blueprints median warnings/100':36}{report['reference_rates']['median_warnings_per_100']:>10}")


def compare(path_a: str, path_b: str) -> None:
    a = json.loads(Path(path_a).read_text(encoding="utf-8"))["summary"]
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))["summary"]
    print(f"{'metric':36}{'A':>10}{'B':>10}{'delta':>10}")
    for section in sorted(set(a) & set(b)):
        print(f"[{section}]")
        for k, va in a[section].items():
            vb = b[section].get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                print(f"  {k:34}{va:>10}{vb:>10}{round(vb - va, 3):>10}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a FactorioLLM adapter")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--set", dest="sets", choices=["A", "B", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Set B sample size (or first N of set A)")
    parser.add_argument("--max-new-tokens", type=int, default=3500)
    parser.add_argument("--retries", type=int, default=1, help=">1: evaluate the full agentic loop with validator feedback")
    parser.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"))
    parser.add_argument("--build-reference", action="store_true", help="Compute warning rates of real blueprints from the DB")
    parser.add_argument("--rescore", metavar="REPORT", help="Re-execute and re-score an existing report (no GPU)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
    elif args.rescore:
        rescore(args.rescore)
    elif args.build_reference:
        import asyncio
        rates = asyncio.run(_build_reference_rates())
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REFERENCE_RATES.write_text(json.dumps(rates, indent=1), encoding="utf-8")
        print(json.dumps(rates, indent=1))
    else:
        run_eval(args.adapter, args.sets, args.limit, args.max_new_tokens, args.retries)


if __name__ == "__main__":
    main()
