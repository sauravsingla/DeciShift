from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "public_digits_xgboost" / "run.py"
OUT = ROOT / "hf-space" / "data.json"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("decishift_digits_example", EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_example_module()
    payload = module.run()
    case = payload["decision_change_case_study"]

    cohorts = []
    for row in case.get("cohorts", []):
        if row.get("dimension") != "actual_digit":
            continue
        label = str(row["cohort"])
        limit = 0.08 if label == "6" else 0.10
        observed = float(row["action_shift_rate"])
        cohorts.append(
            {
                "digit": label,
                "size": int(row["size"]),
                "coverage": float(row["coverage"]),
                "action_shift_rate": observed,
                "global_action_shift_rate": float(row["global_action_shift_rate"]),
                "excess_action_shift_rate": float(row["excess_action_shift_rate"]),
                "most_common_transition": str(row["most_common_transition"]),
                "declared_limit": limit,
                "violation": observed > limit,
            }
        )

    compact = {
        "title": "DeciShift Decision Change Explorer",
        "scenario": "Public Digits / XGBoost",
        "generated_from_commit": os.environ.get("GITHUB_SHA", "local"),
        "source_dataset": payload["dataset"],
        "records": int(case["records"]),
        "changed_actions": int(case["changed_actions"]),
        "action_shift_rate": float(case["action_shift_rate"]),
        "model_metrics": case["model_metrics"],
        "baseline_action_distribution": case["baseline_action_distribution"],
        "candidate_action_distribution": case["candidate_action_distribution"],
        "transitions": [
            {"transition": key, "count": int(value)}
            for key, value in case["transitions"].items()
        ],
        "changed_nodes": list(case["changed_nodes"]),
        "attribution": [
            {"node": key, "share": float(value)}
            for key, value in sorted(
                case["attribution_absolute_share"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "cohorts": sorted(cohorts, key=lambda row: int(row["digit"])),
        "contract": case["contract"],
        "contract_evaluation": case["contract_evaluation"],
        "limits_note": (
            "What-if controls in the Space change only the displayed gate interpretation; "
            "they do not rerun DeciShift or alter the published benchmark evidence."
        ),
    }

    OUT.write_text(json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
