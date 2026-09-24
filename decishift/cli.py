from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from decishift.attribution import (
    approximate_attribution,
    calculate_attribution_diagnostics,
    exact_attribution,
    pairwise_interactions,
)
from decishift.cohorts import analyze_cohorts, select_cohort_columns
from decishift.config import build_flow, build_pipeline, configuration_sha256, load_data, load_yaml, resolve_data_path
from decishift.contracts import (
    EXIT_INTEGRITY_FAILURE,
    evaluate_contract,
    load_contract,
    render_contract_terminal,
)
from decishift.core.exceptions import DeciShiftError
from decishift.demo import run_demo
from decishift.diff.compare import compare_pipelines, compare_predictions
from decishift.evidence import sha256_bytes, sha256_file
from decishift.flow.analysis import analyze_flow_cohorts, analyze_multiclass_outcomes
from decishift.flow.attribution import approximate_flow_attribution, exact_flow_attribution, flow_pairwise_interactions
from decishift.flow.compare import compare_flows
from decishift.flow.hybrid import FlowHybridCache
from decishift.flow.reports import render_flow_html, render_flow_json, render_flow_markdown, render_flow_terminal
from decishift.fragility import analyze_fragility
from decishift.outcome import analyze_outcomes
from decishift.reports import render_html, render_json, render_markdown, render_terminal
from decishift.replay.engine import HybridReplayCache
from decishift.run_compare import compare_saved_runs
from decishift.store import RunStore

app = typer.Typer(no_args_is_help=True, help="Explain why decisions changed between ML system versions.")


def _is_flow_result(result) -> bool:
    return getattr(result, "metadata", {}).get("mode") == "flow"


def _render(result, format: str) -> str:
    if _is_flow_result(result):
        if format == "terminal":
            return render_flow_terminal(result)
        if format == "json":
            return render_flow_json(result)
        if format == "markdown":
            return render_flow_markdown(result)
        if format == "html":
            return render_flow_html(result)
    else:
        if format == "terminal":
            return render_terminal(result)
        if format == "json":
            return render_json(result)
        if format == "markdown":
            return render_markdown(result)
        if format == "html":
            return render_html(result)
    raise typer.BadParameter("format must be terminal, json, markdown, or html")


@app.command()
def demo(
    rows: Annotated[int, typer.Option(help="Synthetic equipment-maintenance records.")] = 10_000,
    save: Annotated[bool, typer.Option(help="Persist the run under .decishift/runs.")] = True,
):
    """Run the fully local, CPU-only v0.1/v0.2 linear demonstration."""
    _, result = run_demo(rows)
    typer.echo(render_terminal(result))
    if save:
        run_id = RunStore().save(result)
        typer.echo(f"\nRun ID: {run_id}")
        typer.echo(f"Verify evidence: decishift verify {run_id}")
        typer.echo(f"HTML report: decishift report {run_id} --format html")


def _attach_attribution(
    result,
    baseline,
    candidate,
    frame,
    attribution_mode: str,
    permutations: int,
    seed: int,
    confidence_level: float,
    *,
    min_permutations: int | None = None,
    max_permutations: int | None = None,
    batch_size: int = 32,
    target_ci_width: float | None = None,
):
    if not result.changed_components or attribution_mode == "none":
        return
    replay = HybridReplayCache(baseline, candidate, frame)
    if attribution_mode == "exact":
        result.attribution = exact_attribution(baseline, candidate, frame, result=result, cache=replay)
        method = "exact"
        params = {"max_components": 10}
        perms = None
        tolerance = 1e-9
    elif attribution_mode == "approximate":
        result.attribution = approximate_attribution(
            baseline,
            candidate,
            frame,
            result=result,
            permutations=permutations,
            seed=seed,
            confidence_level=confidence_level,
            cache=replay,
            min_permutations=min_permutations,
            max_permutations=max_permutations,
            batch_size=batch_size,
            target_ci_width=target_ci_width,
        )
        method = "approximate"
        params = {
            "permutations": permutations,
            "min_permutations": min_permutations,
            "max_permutations": max_permutations,
            "batch_size": batch_size,
            "target_ci_width": target_ci_width,
            "confidence_level": confidence_level,
        }
        perms = int(result.attribution.attrs.get("permutations_used", permutations))
        tolerance = 1e-8
    else:
        raise typer.BadParameter("attribution must be exact, approximate, or none")
    result.interactions = pairwise_interactions(baseline, candidate, frame, result=result, cache=replay)
    result.metadata.update({
        "hybrid_pipeline_evaluations": replay.evaluations,
        "attribution_method": method,
        "attribution_parameters": params,
        "random_seed": seed if method == "approximate" else None,
    })
    result.diagnostics = calculate_attribution_diagnostics(
        result,
        result.attribution,
        method=method,
        hybrid_evaluations=replay.evaluations,
        permutations=perms,
        tolerance=tolerance,
    )


def _attach_flow_attribution(
    result,
    baseline,
    candidate,
    frame,
    attribution_mode: str,
    permutations: int,
    seed: int,
    confidence_level: float,
    *,
    attribution_target: str,
    grouped_attribution: bool,
    min_permutations: int | None,
    max_permutations: int | None,
    batch_size: int,
    target_ci_width: float | None,
):
    if attribution_mode == "none" or not result.changed_nodes:
        result.metadata.setdefault("attribution_status", "not requested" if attribution_mode == "none" else "no changed nodes")
        return
    if not result.metadata.get("topology_compatible", False):
        result.metadata["attribution_status"] = "unsupported topology change"
        return

    replay = FlowHybridCache(baseline, candidate, frame)
    if attribution_mode == "exact":
        result.attribution, result.diagnostics = exact_flow_attribution(
            baseline,
            candidate,
            frame,
            result=result,
            target=attribution_target,
            grouped=grouped_attribution,
            cache=replay,
        )
        params = {"max_players": 10, "grouped": grouped_attribution}
    elif attribution_mode == "approximate":
        result.attribution, result.diagnostics = approximate_flow_attribution(
            baseline,
            candidate,
            frame,
            result=result,
            target=attribution_target,
            grouped=grouped_attribution,
            permutations=permutations,
            min_permutations=min_permutations,
            max_permutations=max_permutations,
            batch_size=batch_size,
            target_ci_width=target_ci_width,
            confidence_level=confidence_level,
            seed=seed,
            cache=replay,
        )
        params = {
            "permutations": permutations,
            "min_permutations": min_permutations,
            "max_permutations": max_permutations,
            "batch_size": batch_size,
            "target_ci_width": target_ci_width,
            "confidence_level": confidence_level,
            "grouped": grouped_attribution,
        }
    else:
        raise typer.BadParameter("attribution must be exact, approximate, or none")

    result.interactions = flow_pairwise_interactions(
        baseline,
        candidate,
        frame,
        result=result,
        target=attribution_target,
        grouped=grouped_attribution,
        cache=replay,
    )
    result.metadata.update({
        "attribution_status": "available",
        "attribution_target": attribution_target,
        "attribution_method": attribution_mode,
        "attribution_parameters": params,
        "random_seed": seed if attribution_mode == "approximate" else None,
        "hybrid_evaluations": replay.evaluations,
        "hybrid_execution": {
            "nodes_evaluated": replay.nodes_evaluated,
            "node_outputs_reused": replay.node_outputs_reused,
            "cache_hits": replay.cache_hits,
            "cache_misses": replay.cache_misses,
        },
    })


def _post_analysis(cfg, frame, result, *, id_column: str | None):
    if _is_flow_result(result):
        if cfg.get("auto_cohorts", True):
            cohort_columns = cfg.get("cohorts")
            requested_transitions = cfg.get("cohort_transitions") or []
            result.cohorts = analyze_flow_cohorts(
                frame,
                result,
                columns=cohort_columns,
                id_column=id_column,
                min_size=int(cfg.get("min_cohort_size", 30)),
                max_categories=int(cfg.get("max_cohort_categories", 30)),
                confidence_level=float(cfg.get("cohort_confidence_level", 0.95)),
                enable_temporal=bool(cfg.get("temporal_cohorts", False)),
                requested_transitions=requested_transitions,
            )
        outcome_cfg = cfg.get("outcome") or {}
        if outcome_cfg:
            if not isinstance(outcome_cfg, dict):
                raise typer.BadParameter("flow outcome configuration must be a mapping")
            column = outcome_cfg.get("column")
            if not column:
                raise typer.BadParameter("flow outcome.column is required when outcome analysis is configured")
            result.outcome_analysis = analyze_multiclass_outcomes(
                frame,
                result,
                outcome_column=str(column),
                actions_are_predictions=bool(outcome_cfg.get("actions_are_predictions", False)),
            )
        return

    if cfg.get("auto_cohorts", True):
        cohort_columns = cfg.get("cohorts")
        if cohort_columns is None:
            excluded = set()
            if cfg.get("mode") == "predictions-only":
                excluded.update((cfg.get("columns") or {}).values())
            cohort_columns, reasons = select_cohort_columns(
                frame,
                id_column=id_column,
                excluded_columns=excluded,
                max_categories=int(cfg.get("max_cohort_categories", 30)),
                enable_temporal=bool(cfg.get("temporal_cohorts", False)),
            )
            result.metadata["cohort_auto_exclusions"] = reasons
        result.cohorts = analyze_cohorts(
            frame,
            result,
            columns=cohort_columns,
            id_column=id_column,
            min_size=int(cfg.get("min_cohort_size", 30)),
            max_categories=int(cfg.get("max_cohort_categories", 30)),
            confidence_level=float(cfg.get("cohort_confidence_level", 0.95)),
            enable_temporal=bool(cfg.get("temporal_cohorts", False)),
        )
    if cfg.get("outcome_column"):
        analyze_outcomes(frame, result, outcome_column=str(cfg["outcome_column"]))
    if cfg.get("fragility", True):
        fragility_cfg = cfg.get("fragility") if isinstance(cfg.get("fragility"), dict) else {}
        analyze_fragility(result, boundary_bands=fragility_cfg.get("boundary_bands", (0.01, 0.05)))


def _adaptive_options(cfg: dict, *, min_permutations, max_permutations, batch_size, target_ci_width):
    spec = cfg.get("approximate_attribution") or {}
    if spec and not isinstance(spec, dict):
        raise typer.BadParameter("approximate_attribution must be a mapping")
    return (
        spec.get("min_permutations", min_permutations),
        spec.get("max_permutations", max_permutations),
        int(spec.get("batch_size", batch_size)),
        spec.get("target_ci_width", target_ci_width),
    )


def _combined_compare(
    config_path: Path,
    attribution_mode: str,
    permutations: int,
    seed: int,
    confidence_level: float,
    *,
    min_permutations: int | None,
    max_permutations: int | None,
    batch_size: int,
    target_ci_width: float | None,
    attribution_target: str,
    grouped_attribution: bool,
):
    cfg = load_yaml(config_path)
    base_dir = config_path.parent
    data_spec = cfg.get("data") or {}
    frame = load_data(data_spec, base_dir=base_dir)
    id_column = cfg.get("id_column")
    hash_input = bool(cfg.get("hash_input", True))
    min_p, max_p, batch, target_width = _adaptive_options(
        cfg,
        min_permutations=min_permutations,
        max_permutations=max_permutations,
        batch_size=batch_size,
        target_ci_width=target_ci_width,
    )

    if cfg.get("mode") == "flow":
        baseline_spec = dict(cfg.get("baseline") or {})
        candidate_spec = dict(cfg.get("candidate") or {})
        if cfg.get("strict_reproducibility"):
            baseline_spec.setdefault("strict_reproducibility", True)
            candidate_spec.setdefault("strict_reproducibility", True)
        baseline = build_flow(baseline_spec, "baseline", base_dir=base_dir)
        candidate = build_flow(candidate_spec, "candidate", base_dir=base_dir)
        result = compare_flows(
            baseline,
            candidate,
            frame,
            id_column=id_column,
            allow_duplicate_ids=bool(cfg.get("allow_duplicate_ids", False)),
            allow_null_ids=bool(cfg.get("allow_null_ids", False)),
            hash_input=hash_input,
        )
        target = str(cfg.get("attribution_target", attribution_target))
        grouped = bool(cfg.get("grouped_attribution", grouped_attribution))
        _attach_flow_attribution(
            result,
            baseline,
            candidate,
            frame,
            attribution_mode,
            permutations,
            seed,
            confidence_level,
            attribution_target=target,
            grouped_attribution=grouped,
            min_permutations=min_p,
            max_permutations=max_p,
            batch_size=batch,
            target_ci_width=target_width,
        )
    elif cfg.get("mode") == "predictions-only":
        columns = cfg.get("columns") or {}
        required = ["baseline_score", "candidate_score", "baseline_decision", "candidate_decision"]
        missing = [name for name in required if name not in columns]
        if missing:
            raise typer.BadParameter(f"predictions-only columns missing: {', '.join(missing)}")
        result = compare_predictions(
            frame,
            baseline_score=columns["baseline_score"],
            candidate_score=columns["candidate_score"],
            baseline_decision=columns["baseline_decision"],
            candidate_decision=columns["candidate_decision"],
            baseline_threshold=columns.get("baseline_threshold", cfg.get("baseline_threshold", 0.5)),
            candidate_threshold=columns.get("candidate_threshold", cfg.get("candidate_threshold", 0.5)),
            id_column=id_column,
            allow_duplicate_ids=bool(cfg.get("allow_duplicate_ids", False)),
            allow_null_ids=bool(cfg.get("allow_null_ids", False)),
            hash_input=hash_input,
        )
    else:
        baseline_spec = dict(cfg.get("baseline") or {})
        candidate_spec = dict(cfg.get("candidate") or {})
        if cfg.get("strict_reproducibility"):
            baseline_spec.setdefault("strict_reproducibility", True)
            candidate_spec.setdefault("strict_reproducibility", True)
        baseline = build_pipeline(baseline_spec, "baseline", base_dir=base_dir)
        candidate = build_pipeline(candidate_spec, "candidate", base_dir=base_dir)
        result = compare_pipelines(
            baseline,
            candidate,
            frame,
            id_column=id_column,
            allow_duplicate_ids=bool(cfg.get("allow_duplicate_ids", False)),
            allow_null_ids=bool(cfg.get("allow_null_ids", False)),
            hash_input=hash_input,
        )
        _attach_attribution(
            result,
            baseline,
            candidate,
            frame,
            attribution_mode,
            permutations,
            seed,
            confidence_level,
            min_permutations=min_p,
            max_permutations=max_p,
            batch_size=batch,
            target_ci_width=target_width,
        )

    result.metadata["configuration_sha256"] = configuration_sha256(config_path)
    result.metadata["input_content_hashing_enabled"] = hash_input
    result.metadata["input_data_fingerprint"] = sha256_file(resolve_data_path(data_spec, base_dir=base_dir)) if hash_input else None
    _post_analysis(cfg, frame, result, id_column=id_column)
    return result, cfg


@app.command("compare")
def compare_command(
    configs: Annotated[list[Path], typer.Argument(help="One combined YAML config, or baseline.yaml candidate.yaml.")],
    format: Annotated[str, typer.Option(help="terminal, json, markdown, or html")] = "terminal",
    attribution: Annotated[str, typer.Option(help="exact, approximate, or none")] = "exact",
    permutations: Annotated[int, typer.Option(help="Fixed permutation count or adaptive default maximum.")] = 256,
    seed: Annotated[int, typer.Option(help="Random seed for approximate attribution.")] = 0,
    confidence_level: Annotated[float, typer.Option(help="Monte Carlo confidence level.")] = 0.95,
    min_permutations: Annotated[int | None, typer.Option(help="Adaptive minimum permutations.")] = None,
    max_permutations: Annotated[int | None, typer.Option(help="Adaptive maximum permutations.")] = None,
    batch_size: Annotated[int, typer.Option(help="Adaptive permutation batch size.")] = 32,
    target_ci_width: Annotated[float | None, typer.Option(help="Adaptive maximum target CI width.")] = None,
    attribution_target: Annotated[str, typer.Option(help="Flow target: candidate_action_support or change_from_baseline.")] = "candidate_action_support",
    grouped_attribution: Annotated[bool, typer.Option(help="For flows, attribute declared groups instead of individual changed nodes.")] = False,
    contract: Annotated[Path | None, typer.Option(help="Optional Decision Contract YAML.")] = None,
    save: Annotated[bool, typer.Option(help="Persist machine-readable evidence.")] = True,
):
    """Compare baseline and candidate decision systems on the same local records."""
    try:
        if len(configs) == 1:
            result, _ = _combined_compare(
                configs[0],
                attribution,
                permutations,
                seed,
                confidence_level,
                min_permutations=min_permutations,
                max_permutations=max_permutations,
                batch_size=batch_size,
                target_ci_width=target_ci_width,
                attribution_target=attribution_target,
                grouped_attribution=grouped_attribution,
            )
        elif len(configs) == 2:
            base_cfg = load_yaml(configs[0])
            cand_cfg = load_yaml(configs[1])
            if base_cfg.get("mode") == "flow" or cand_cfg.get("mode") == "flow":
                raise typer.BadParameter("DecisionFlow uses one combined config with baseline and candidate sections")
            data_cfg = base_cfg.get("data") or cand_cfg.get("data")
            if not data_cfg:
                raise typer.BadParameter("One of the two configs must provide data.path")
            data_base_dir = configs[0].parent if base_cfg.get("data") else configs[1].parent
            frame = load_data(data_cfg, base_dir=data_base_dir)
            id_column = base_cfg.get("id_column") or cand_cfg.get("id_column")
            merged_cfg = {**cand_cfg, **base_cfg}
            hash_input = bool(merged_cfg.get("hash_input", True))
            baseline_spec = dict(base_cfg.get("pipeline") or base_cfg)
            candidate_spec = dict(cand_cfg.get("pipeline") or cand_cfg)
            if merged_cfg.get("strict_reproducibility"):
                baseline_spec.setdefault("strict_reproducibility", True)
                candidate_spec.setdefault("strict_reproducibility", True)
            baseline = build_pipeline(baseline_spec, "baseline", base_dir=configs[0].parent)
            candidate = build_pipeline(candidate_spec, "candidate", base_dir=configs[1].parent)
            result = compare_pipelines(
                baseline,
                candidate,
                frame,
                id_column=id_column,
                allow_duplicate_ids=bool(merged_cfg.get("allow_duplicate_ids", False)),
                allow_null_ids=bool(merged_cfg.get("allow_null_ids", False)),
                hash_input=hash_input,
            )
            min_p, max_p, batch, target_width = _adaptive_options(
                merged_cfg,
                min_permutations=min_permutations,
                max_permutations=max_permutations,
                batch_size=batch_size,
                target_ci_width=target_ci_width,
            )
            _attach_attribution(
                result,
                baseline,
                candidate,
                frame,
                attribution,
                permutations,
                seed,
                confidence_level,
                min_permutations=min_p,
                max_permutations=max_p,
                batch_size=batch,
                target_ci_width=target_width,
            )
            result.metadata["configuration_sha256"] = sha256_bytes(
                configs[0].read_bytes() + b"\0" + configs[1].read_bytes()
            )
            result.metadata["input_content_hashing_enabled"] = hash_input
            result.metadata["input_data_fingerprint"] = (
                sha256_file(resolve_data_path(data_cfg, base_dir=data_base_dir)) if hash_input else None
            )
            _post_analysis(merged_cfg, frame, result, id_column=id_column)
        else:
            raise typer.BadParameter("Pass either one combined config or two pipeline configs")
    except (DeciShiftError, ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(_render(result, format))
    run_id = None
    if save or contract is not None:
        run_id = RunStore().save(result)
        typer.echo(f"\nRun ID: {run_id}")
    if contract is not None:
        verification = RunStore().verify(run_id)
        evaluation = evaluate_contract(result, load_contract(contract), integrity_passed=verification.passed)
        typer.echo("\n" + render_contract_terminal(evaluation))
        if evaluation.exit_code:
            raise typer.Exit(code=evaluation.exit_code)


@app.command()
def graph(
    config: Annotated[Path, typer.Argument(help="Combined mode: flow YAML configuration.")],
    format: Annotated[str, typer.Option(help="text or mermaid")] = "text",
):
    """Inspect a local DecisionFlow graph without executing model components."""
    try:
        cfg = load_yaml(config)
        if cfg.get("mode") != "flow":
            raise typer.BadParameter("graph requires a config with mode: flow")
        baseline = build_flow(dict(cfg.get("baseline") or {}), "baseline", base_dir=config.parent)
        candidate = build_flow(dict(cfg.get("candidate") or {}), "candidate", base_dir=config.parent)
    except (DeciShiftError, ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if format == "text":
        typer.echo(baseline.to_text())
        typer.echo("\n" + candidate.to_text())
    elif format == "mermaid":
        typer.echo("%% baseline\n" + baseline.to_mermaid())
        typer.echo("\n%% candidate\n" + candidate.to_mermaid())
    else:
        raise typer.BadParameter("format must be text or mermaid")


@app.command()
def explain(
    run: Annotated[str, typer.Option("--run", help="Saved DeciShift run ID.")],
    id: Annotated[str, typer.Option("--id", help="Record identifier.")],
):
    """Explain one historical comparison from saved evidence."""
    try:
        data = RunStore().explain(run, id)
    except (FileNotFoundError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(data, indent=2, default=str))


@app.command()
def report(
    run: Annotated[str, typer.Argument(help="Saved DeciShift run ID.")],
    format: Annotated[str, typer.Option(help="terminal, json, markdown, or html")] = "markdown",
):
    """Render a saved run without recomputing the comparison."""
    store = RunStore()
    try:
        if format == "html":
            path = store.report_path(run, "html")
            typer.echo(str(path))
        else:
            typer.echo(store.report(run, format))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def verify(run: Annotated[str, typer.Argument(help="Saved DeciShift run ID.")]):
    """Verify tamper-evident hashes for a saved evidence bundle."""
    try:
        result = RunStore().verify(run)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("DeciShift evidence verification\n")
    typer.echo(f"Run: {run}\n")
    typer.echo(f"{result.manifest_status} manifest")
    for item in result.items:
        typer.echo(f"{item.status:<7}{item.artifact}")
        if item.status != "PASS":
            typer.echo(f"  expected: {item.expected}")
            typer.echo(f"  actual:   {item.actual}")
    typer.echo(f"\nIntegrity root:\n{result.integrity_root}\n")
    typer.echo(result.message)
    if not result.passed:
        raise typer.Exit(code=EXIT_INTEGRITY_FAILURE)


@app.command()
def gate(
    run: Annotated[str, typer.Argument(help="Saved DeciShift run ID.")],
    contract: Annotated[Path, typer.Option("--contract", help="Decision Contract YAML.")],
):
    """Evaluate deterministic user-declared Decision Contract thresholds."""
    store = RunStore()
    try:
        verification = store.verify(run)
        result = store.load(run)
        evaluation = evaluate_contract(result, load_contract(contract), integrity_passed=verification.passed)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(render_contract_terminal(evaluation))
    if evaluation.exit_code:
        raise typer.Exit(code=evaluation.exit_code)


@app.command("compare-runs")
def compare_runs(
    run_a: Annotated[str, typer.Argument()],
    run_b: Annotated[str, typer.Argument()],
    format: Annotated[str, typer.Option(help="terminal or json")] = "terminal",
):
    """Compare two saved DeciShift analyses without replaying underlying models."""
    try:
        data = compare_saved_runs(run_a, run_b)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if format == "json":
        typer.echo(json.dumps(data, indent=2, default=str))
    elif format == "terminal":
        shift = data["decision_shift_rate"]
        typer.echo("DeciShift saved-run comparison\n" + "=" * 56)
        typer.echo(f"{run_a}: {shift['run_a']:.2%}")
        typer.echo(f"{run_b}: {shift['run_b']:.2%}")
        typer.echo(f"delta: {shift['delta']:+.2%}")
        typer.echo("\n" + data["note"])
    else:
        raise typer.BadParameter("format must be terminal or json")


if __name__ == "__main__":
    app()
