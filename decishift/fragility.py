from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from decishift.diff.compare import ComparisonResult


def analyze_fragility(
    result: ComparisonResult,
    *,
    boundary_bands: Iterable[float] = (0.01, 0.05),
) -> dict[str, Any]:
    """Describe proximity to decision boundaries without causal robustness claims."""
    bands = sorted({float(b) for b in boundary_bands})
    if not bands or any(b <= 0 or not np.isfinite(b) for b in bands):
        raise ValueError("boundary_bands must contain positive finite values")
    r = result.records
    base_abs = r["baseline_margin"].abs().to_numpy(dtype=float)
    cand_abs = r["candidate_margin"].abs().to_numpy(dtype=float)
    min_abs = np.minimum(base_abs, cand_abs)
    changed = r["changed"].to_numpy(dtype=bool)
    classification = np.full(len(r), "unchanged", dtype=object)

    rule_forced = np.zeros(len(r), dtype=bool)
    if result.baseline_trace is not None and result.candidate_trace is not None:
        b_rule = result.baseline_trace.pre_rule_decisions != result.baseline_trace.decisions
        c_rule = result.candidate_trace.pre_rule_decisions != result.candidate_trace.decisions
        rule_forced = changed & (b_rule | c_rule)
    classification[rule_forced] = "rule-forced change"

    boundary_crossing = changed & ~rule_forced & (
        (r["baseline_margin"].to_numpy(dtype=float) < 0) !=
        (r["candidate_margin"].to_numpy(dtype=float) < 0)
    )
    classification[boundary_crossing] = "boundary-crossing change"
    remaining = changed & ~rule_forced & ~boundary_crossing
    classification[remaining] = "far-from-boundary change"

    records = pd.DataFrame({
        "record_id": r["record_id"].to_numpy(),
        "baseline_abs_margin": base_abs,
        "candidate_abs_margin": cand_abs,
        "minimum_abs_margin": min_abs,
        "change_classification": classification,
    })
    summary: dict[str, Any] = {
        "boundary_bands": bands,
        "median_absolute_margin": float(np.median(min_abs)) if len(min_abs) else 0.0,
        "p10_absolute_margin": float(np.quantile(min_abs, 0.10)) if len(min_abs) else 0.0,
        "boundary_crossing_changes": int((classification == "boundary-crossing change").sum()),
        "far_from_boundary_changes": int((classification == "far-from-boundary change").sum()),
        "rule_forced_changes": int((classification == "rule-forced change").sum()),
        "interpretation_note": "Decision fragility measures threshold proximity; it is not a causal robustness claim.",
    }
    for band in bands:
        key = f"fraction_within_{band:g}_of_boundary"
        summary[key] = float((min_abs <= band).mean()) if len(min_abs) else 0.0
    payload = {"summary": summary, "records": records}
    result.fragility = payload
    return payload
