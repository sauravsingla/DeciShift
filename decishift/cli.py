from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from decishift.attribution import approximate_attribution, exact_attribution, pairwise_interactions
from decishift.cohorts import analyze_cohorts
from decishift.config import build_pipeline, load_data, load_yaml
from decishift.core.exceptions import DeciShiftError
from decishift.demo import run_demo
from decishift.diff.compare import compare_pipelines, compare_predictions
from decishift.reports import render_json, render_markdown, render_terminal
from decishift.replay.engine import HybridReplayCache
from decishift.store import RunStore

app = typer.Typer(no_args_is_help=True, help="Explain why decisions changed between ML system versions.")


def _render(result, format: str) -> str:
    if format == "terminal":
        return render_terminal(result)
    if format == "json":
        return render_json(result)
    if format == "markdown":
        return render_markdown(result)
    raise typer.BadParameter("format must be terminal, json, or markdown")


@app.command()
def demo(
    rows: Annotated[int, typer.Option(help="Synthetic equipment-maintenance records.")] = 10_000,
    save: Annotated[bool, typer.Option(help="Persist the run under .decishift/runs.")] = True,
):
    """Run the fully local, CPU-only demonstration."""
    _, result = run_demo(rows)
    typer.echo(render_terminal(result))
    if save:
        run_id = RunStore().save(result)
        typer.echo(f"\nRun ID: {run_id}")
        typer.echo(f"Explain a record: decishift explain --run {run_id} --id <RECORD_ID>")


def _combined_compare(config_path: Path, attribution_mode: str, permutations: int):
    cfg = load_yaml(config_path)
    base_dir = config_path.parent
    data_spec = cfg.get("data") or {}
    frame = load_data(data_spec, base_dir=base_dir)
    id_column = cfg.get("id_column")

    if cfg.get("mode") == "predictions-only":
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
        )
    else:
        baseline = build_pipeline(cfg.get("baseline") or {}, "baseline")
        candidate = build_pipeline(cfg.get("candidate") or {}, "candidate")
        result = compare_pipelines(baseline, candidate, frame, id_column=id_column)
        if result.changed_components:
            replay = HybridReplayCache(baseline, candidate, frame)
            if attribution_mode == "exact":
                result.attribution = exact_attribution(baseline, candidate, frame, result=result, cache=replay)
            elif attribution_mode == "approximate":
                result.attribution = approximate_attribution(baseline, candidate, frame, result=result, permutations=permutations, cache=replay)
            elif attribution_mode != "none":
                raise typer.BadParameter("attribution must be exact, approximate, or none")
            result.interactions = pairwise_interactions(baseline, candidate, frame, result=result, cache=replay)
            result.metadata["hybrid_pipeline_evaluations"] = replay.evaluations

    if cfg.get("auto_cohorts", True):
        cohort_columns = cfg.get("cohorts")
        if cohort_columns is None:
            excluded = {id_column} if id_column else set()
            if cfg.get("mode") == "predictions-only":
                excluded.update((cfg.get("columns") or {}).values())
            cohort_columns = [c for c in frame.columns if c not in excluded]
        result.cohorts = analyze_cohorts(
            frame, result, columns=cohort_columns, min_size=int(cfg.get("min_cohort_size", 30))
        )
    return result


@app.command("compare")
def compare_command(
    configs: Annotated[list[Path], typer.Argument(help="One combined YAML config, or baseline.yaml candidate.yaml.")],
    format: Annotated[str, typer.Option(help="terminal, json, or markdown")] = "terminal",
    attribution: Annotated[str, typer.Option(help="exact, approximate, or none")] = "exact",
    permutations: Annotated[int, typer.Option(help="Permutation count for approximate attribution.")] = 256,
    save: Annotated[bool, typer.Option(help="Persist machine-readable evidence.")] = True,
):
    """Compare baseline and candidate decision systems on the same local records."""
    try:
        if len(configs) == 1:
            result = _combined_compare(configs[0], attribution, permutations)
        elif len(configs) == 2:
            base_cfg = load_yaml(configs[0])
            cand_cfg = load_yaml(configs[1])
            data_cfg = base_cfg.get("data") or cand_cfg.get("data")
            if not data_cfg:
                raise typer.BadParameter("One of the two configs must provide data.path")
            data_base_dir = configs[0].parent if base_cfg.get("data") else configs[1].parent
            frame = load_data(data_cfg, base_dir=data_base_dir)
            id_column = base_cfg.get("id_column") or cand_cfg.get("id_column")
            baseline = build_pipeline(base_cfg.get("pipeline") or base_cfg, "baseline")
            candidate = build_pipeline(cand_cfg.get("pipeline") or cand_cfg, "candidate")
            result = compare_pipelines(baseline, candidate, frame, id_column=id_column)
            replay = HybridReplayCache(baseline, candidate, frame)
            if attribution == "exact":
                result.attribution = exact_attribution(baseline, candidate, frame, result=result, cache=replay)
            elif attribution == "approximate":
                result.attribution = approximate_attribution(baseline, candidate, frame, result=result, permutations=permutations, cache=replay)
            elif attribution != "none":
                raise typer.BadParameter("attribution must be exact, approximate, or none")
            result.interactions = pairwise_interactions(baseline, candidate, frame, result=result, cache=replay)
            result.metadata["hybrid_pipeline_evaluations"] = replay.evaluations
            auto_cohorts = base_cfg.get("auto_cohorts", cand_cfg.get("auto_cohorts", True))
            if auto_cohorts:
                cohort_columns = base_cfg.get("cohorts") or cand_cfg.get("cohorts")
                if cohort_columns is None:
                    cohort_columns = [c for c in frame.columns if c != id_column]
                min_size = int(base_cfg.get("min_cohort_size", cand_cfg.get("min_cohort_size", 30)))
                result.cohorts = analyze_cohorts(frame, result, columns=cohort_columns, min_size=min_size)
        else:
            raise typer.BadParameter("Pass either one combined config or two pipeline configs")
    except (DeciShiftError, ValueError, KeyError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(_render(result, format))
    if save:
        run_id = RunStore().save(result)
        typer.echo(f"\nRun ID: {run_id}")


@app.command()
def explain(
    run: Annotated[str, typer.Option("--run", help="Saved DeciShift run ID.")],
    id: Annotated[str, typer.Option("--id", help="Record identifier.")],
):
    """Explain one historical decision comparison from saved evidence."""
    try:
        payload = RunStore().explain(run, id)
    except (FileNotFoundError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def report(
    run: Annotated[str, typer.Argument(help="Saved DeciShift run ID.")],
    format: Annotated[str, typer.Option(help="terminal, json, or markdown")] = "markdown",
):
    """Render a saved run without recomputing the comparison."""
    try:
        typer.echo(RunStore().report(run, format))
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
