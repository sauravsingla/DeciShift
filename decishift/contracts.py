from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from decishift.diff.compare import ComparisonResult

EXIT_PASS = 0
EXIT_USAGE = 2
EXIT_CONTRACT_VIOLATED = 10
EXIT_INSUFFICIENT_EVIDENCE = 11
EXIT_INTEGRITY_FAILURE = 12


@dataclass
class ContractCheck:
    name: str
    status: str
    observed: Any = None
    limit: Any = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContractEvaluation:
    checks: list[ContractCheck]
    result: str
    exit_code: int
    note: str = "A passing Decision Contract means only that the user-declared DeciShift contract passed."

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "result": self.result,
            "exit_code": self.exit_code,
            "note": self.note,
        }


def load_contract(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Decision Contract YAML root must be a mapping")
    contract = value.get("contract", value)
    if not isinstance(contract, dict):
        raise ValueError("contract must be a mapping")
    return contract


def _limit(checks: list[ContractCheck], name: str, observed: float | None, limit: Any, *, insufficient_if_none: bool = True) -> None:
    if limit is None:
        return
    if observed is None:
        checks.append(ContractCheck(name, "INSUFFICIENT", None, limit, "required evidence is unavailable"))
        return
    passed = float(observed) <= float(limit)
    checks.append(ContractCheck(name, "PASS" if passed else "FAIL", float(observed), float(limit)))


def _cohort_override(cohorts: pd.DataFrame, key: str) -> pd.DataFrame:
    if "=" not in key:
        return cohorts.iloc[0:0]
    dimension, value = key.split("=", 1)
    return cohorts[(cohorts["dimension"].astype(str) == dimension) & (cohorts["cohort"].astype(str) == value)]


def evaluate_contract(
    result: ComparisonResult,
    contract: dict[str, Any],
    *,
    integrity_passed: bool | None = None,
) -> ContractEvaluation:
    checks: list[ContractCheck] = []
    summary = result.summary()
    _limit(checks, "overall decision shift", summary.get("decision_shift_rate"), contract.get("max_decision_shift_rate"))
    _limit(checks, "0->1 shift", summary.get("rate_0_to_1"), contract.get("max_0_to_1_rate"))
    _limit(checks, "1->0 shift", summary.get("rate_1_to_0"), contract.get("max_1_to_0_rate"))

    attribution_contract = contract.get("attribution") or {}
    if attribution_contract.get("require_reproducible_identity"):
        observed = result.metadata.get("reproducibility_status")
        passed = observed == "reproducible"
        checks.append(ContractCheck("reproducibility", "PASS" if passed else "FAIL", observed, "reproducible"))
    if "max_score_efficiency_mae" in attribution_contract:
        observed = None if result.diagnostics is None else result.diagnostics.score_efficiency_mae
        _limit(checks, "attribution score efficiency MAE", observed, attribution_contract["max_score_efficiency_mae"])

    approx_contract = contract.get("approximate_attribution") or {}
    if "max_decision_ci_width" in approx_contract:
        method = result.metadata.get("attribution_method")
        if method == "approximate":
            observed = None
            if result.attribution is not None and {"decision_ci_low", "decision_ci_high"}.issubset(result.attribution.columns):
                widths = result.attribution["decision_ci_high"] - result.attribution["decision_ci_low"]
                finite = widths[pd.notna(widths)]
                if not finite.empty:
                    observed = float(finite.max())
            _limit(checks, "maximum decision attribution CI width", observed, approx_contract["max_decision_ci_width"])
        elif method == "exact":
            checks.append(ContractCheck(
                "maximum decision attribution CI width",
                "PASS",
                "not applicable (exact attribution)",
                approx_contract["max_decision_ci_width"],
                "exact attribution has no permutation-sampling interval",
            ))
        else:
            checks.append(ContractCheck(
                "maximum decision attribution CI width",
                "INSUFFICIENT",
                None,
                approx_contract["max_decision_ci_width"],
                "approximate attribution evidence unavailable",
            ))

    cohort_contract = contract.get("cohorts") or {}
    min_size = int(cohort_contract.get("min_size", 0))
    max_flip_rate = cohort_contract.get("max_flip_rate")
    cohorts = result.cohorts
    if max_flip_rate is not None:
        if cohorts is None:
            checks.append(ContractCheck("cohort flip rate", "INSUFFICIENT", None, max_flip_rate, "cohort evidence unavailable"))
        else:
            eligible = cohorts[cohorts["size"] >= min_size] if min_size else cohorts
            observed = None if eligible.empty else float(eligible["flip_rate"].max())
            _limit(checks, "cohort flip rate", observed, max_flip_rate)
    for key, limits in sorted((cohort_contract.get("overrides") or {}).items()):
        limit = (limits or {}).get("max_flip_rate")
        if limit is None:
            continue
        if cohorts is None:
            checks.append(ContractCheck(f"{key} flip rate", "INSUFFICIENT", None, limit, "cohort evidence unavailable"))
            continue
        matched = _cohort_override(cohorts, key)
        if min_size:
            matched = matched[matched["size"] >= min_size]
        observed = None if matched.empty else float(matched["flip_rate"].max())
        _limit(checks, f"{key} flip rate", observed, limit)

    if integrity_passed is not None:
        checks.append(ContractCheck("evidence integrity", "PASS" if integrity_passed else "FAIL", integrity_passed, True))

    if integrity_passed is False:
        return ContractEvaluation(checks, "BLOCK", EXIT_INTEGRITY_FAILURE)
    if any(c.status == "INSUFFICIENT" for c in checks):
        return ContractEvaluation(checks, "INSUFFICIENT EVIDENCE", EXIT_INSUFFICIENT_EVIDENCE)
    if any(c.status == "FAIL" for c in checks):
        return ContractEvaluation(checks, "BLOCK", EXIT_CONTRACT_VIOLATED)
    return ContractEvaluation(checks, "PASS", EXIT_PASS)


def render_contract_terminal(evaluation: ContractEvaluation) -> str:
    lines = ["DeciShift Decision Contract", "=" * 56]
    for check in evaluation.checks:
        if isinstance(check.observed, float) and isinstance(check.limit, (float, int)):
            detail = f"{check.observed:.4g} <= {float(check.limit):.4g}" if check.status == "PASS" else f"{check.observed:.4g} > {float(check.limit):.4g}"
        elif check.detail:
            detail = check.detail
        else:
            detail = f"{check.observed} / expected {check.limit}"
        lines.append(f"{check.status:<12}{check.name:<42}{detail}")
    lines += ["", f"RESULT: {evaluation.result}", evaluation.note]
    return "\n".join(lines)
