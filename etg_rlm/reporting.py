"""Reporting and visualization for the canonical evaluation framework.

Generates:
    1. Markdown summary tables (model x dataset, model x metric)
    2. Visualization data structures for matplotlib/seaborn plots
    3. JSON export for reproducible results archival
    4. LaTeX table snippets for paper inclusion

Output artifacts:
    - results_summary.md: human-readable markdown report
    - results.json: machine-readable full results
    - figures/: visualization specifications (rendered by plotting code)

References:
    Results formatting follows the conventions of FActScore [8],
    ALCE [4], and HaluEval [2] leaderboard tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import NamedTuple

from etg_rlm.benchmark_runner import (
    BenchmarkDataset,
    BenchmarkReport,
    DatasetResults,
    ModelType,
)


# ---------------------------------------------------------------------------
# Visualization data structures
# ---------------------------------------------------------------------------


class PlotDataPoint(NamedTuple):
    """A single data point for visualization."""

    x_label: str
    y_value: float
    group: str
    error_low: float = 0.0
    error_high: float = 0.0


@dataclass
class BarChartSpec:
    """Specification for a grouped bar chart.

    Attributes:
        title: chart title
        x_label: x-axis label
        y_label: y-axis label
        data: list of data points
        groups: distinct group names (e.g., model names)
        x_categories: distinct x categories (e.g., dataset names)
    """

    title: str
    x_label: str
    y_label: str
    data: list[PlotDataPoint] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    x_categories: list[str] = field(default_factory=list)


@dataclass
class LineChartSpec:
    """Specification for a line chart (e.g., scaling curves).

    Attributes:
        title: chart title
        x_label: x-axis label
        y_label: y-axis label
        series: dict mapping series name to (x_values, y_values)
    """

    title: str
    x_label: str
    y_label: str
    series: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)


@dataclass
class HeatmapSpec:
    """Specification for a heatmap (model x dataset matrix).

    Attributes:
        title: chart title
        row_labels: row labels (e.g., model names)
        col_labels: column labels (e.g., dataset names)
        values: 2D matrix of values
        metric_name: name of the metric displayed
    """

    title: str
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    metric_name: str = ""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


DISPLAY_NAMES = {
    ModelType.ZERO_SHOT: "Zero-Shot GPT-4",
    ModelType.STANDARD_RAG: "Standard RAG (Contriever)",
    ModelType.SELF_CHECK_GPT: "Self-CheckGPT",
    ModelType.ETG: "ETG (Ours)",
}

DATASET_DISPLAY = {
    BenchmarkDataset.TRUTHFUL_QA: "TruthfulQA",
    BenchmarkDataset.HALU_EVAL: "HaluEval",
    BenchmarkDataset.HOTPOT_QA: "HotpotQA",
    BenchmarkDataset.NATURAL_QUESTIONS: "NQ",
    BenchmarkDataset.ELI5: "ELI5",
}


def generate_markdown_report(report: BenchmarkReport) -> str:
    """Generate a comprehensive markdown report from benchmark results.

    Produces:
        1. Executive summary
        2. FactScore comparison table (model x dataset)
        3. Citation metrics table
        4. Per-dataset detailed results
        5. Model rankings

    Args:
        report: the complete BenchmarkReport

    Returns:
        Markdown string ready for rendering.
    """
    lines: list[str] = []

    # Header
    lines.append("# Canonical Evaluation: ETG vs. Baselines")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    # Collect ETG results for summary
    etg_results = [
        r for r in report.results.values() if r.model == ModelType.ETG
    ]
    if etg_results:
        avg_fs = sum(r.mean_factscore for r in etg_results) / len(etg_results)
        avg_cp = sum(r.mean_citation_precision for r in etg_results) / len(etg_results)
        lines.append(f"- **ETG Mean FactScore**: {avg_fs:.3f}")
        lines.append(f"- **ETG Mean Citation Precision**: {avg_cp:.3f}")
        lines.append(f"- **Datasets evaluated**: {len(etg_results)}")
    lines.append("")

    # FactScore table
    lines.append("## FactScore Comparison (Claim Precision)")
    lines.append("")
    lines.extend(_make_metric_table(report, "mean_factscore"))
    lines.append("")

    # Citation Precision table
    lines.append("## Citation Precision")
    lines.append("")
    lines.extend(_make_metric_table(report, "mean_citation_precision"))
    lines.append("")

    # Citation Recall table
    lines.append("## Citation Recall")
    lines.append("")
    lines.extend(_make_metric_table(report, "mean_citation_recall"))
    lines.append("")

    # ROUGE-L table
    lines.append("## ROUGE-L F1")
    lines.append("")
    lines.extend(_make_metric_table(report, "mean_rouge_f1"))
    lines.append("")

    # Logic-Step Accuracy (for multi-hop)
    lines.append("## Logic-Step Accuracy (Multi-hop)")
    lines.append("")
    lines.extend(_make_metric_table(report, "mean_step_accuracy"))
    lines.append("")

    # Rankings
    if report.model_rankings:
        lines.append("## Model Rankings")
        lines.append("")
        for metric, ranking in report.model_rankings.items():
            lines.append(f"**{metric}**: {' > '.join(ranking)}")
            lines.append("")

    return "\n".join(lines)


def generate_latex_table(
    report: BenchmarkReport,
    metric: str = "mean_factscore",
    caption: str = "FactScore comparison across models and datasets.",
    label: str = "tab:factscore",
) -> str:
    """Generate a LaTeX table for paper inclusion.

    Args:
        report: the benchmark report
        metric: which metric to tabulate
        caption: LaTeX caption
        label: LaTeX label

    Returns:
        LaTeX table string.
    """
    models = list(ModelType)
    datasets = list(BenchmarkDataset)

    n_cols = len(datasets) + 1
    col_spec = "l" + "c" * len(datasets)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = "Model"
    for ds in datasets:
        header += f" & {DATASET_DISPLAY.get(ds, ds.value)}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for model in models:
        row = DISPLAY_NAMES.get(model, model.value)
        for ds in datasets:
            result = report.get_result(model, ds)
            if result:
                val = getattr(result, metric, 0.0)
                # Bold the best value
                row += f" & {val:.3f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_json_report(report: BenchmarkReport) -> str:
    """Export the full benchmark report as JSON.

    Args:
        report: the benchmark report

    Returns:
        JSON string with all results.
    """
    data: dict = {"results": {}, "rankings": report.model_rankings}

    for key, result in report.results.items():
        data["results"][key] = {
            "model": result.model.value,
            "dataset": result.dataset.value,
            "n_instances": result.n_instances,
            "mean_factscore": round(result.mean_factscore, 4),
            "mean_claim_precision": round(result.mean_claim_precision, 4),
            "mean_claim_recall": round(result.mean_claim_recall, 4),
            "mean_citation_precision": round(result.mean_citation_precision, 4),
            "mean_citation_recall": round(result.mean_citation_recall, 4),
            "mean_step_accuracy": round(result.mean_step_accuracy, 4),
            "mean_rouge_f1": round(result.mean_rouge_f1, 4),
            "mean_ms_per_token": round(result.mean_ms_per_token, 2),
        }

    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Visualization spec builders
# ---------------------------------------------------------------------------


def build_factscore_bar_chart(report: BenchmarkReport) -> BarChartSpec:
    """Build a grouped bar chart spec for FactScore across models and datasets."""
    spec = BarChartSpec(
        title="FactScore Comparison Across Models and Datasets",
        x_label="Dataset",
        y_label="FactScore",
    )

    models = list(ModelType)
    datasets = list(BenchmarkDataset)

    spec.groups = [DISPLAY_NAMES.get(m, m.value) for m in models]
    spec.x_categories = [DATASET_DISPLAY.get(d, d.value) for d in datasets]

    for model in models:
        for ds in datasets:
            result = report.get_result(model, ds)
            val = result.mean_factscore if result else 0.0
            spec.data.append(PlotDataPoint(
                x_label=DATASET_DISPLAY.get(ds, ds.value),
                y_value=val,
                group=DISPLAY_NAMES.get(model, model.value),
            ))

    return spec


def build_citation_heatmap(report: BenchmarkReport) -> HeatmapSpec:
    """Build a heatmap spec for citation precision (model x dataset)."""
    models = list(ModelType)
    datasets = list(BenchmarkDataset)

    spec = HeatmapSpec(
        title="Citation Precision: Model x Dataset",
        row_labels=[DISPLAY_NAMES.get(m, m.value) for m in models],
        col_labels=[DATASET_DISPLAY.get(d, d.value) for d in datasets],
        metric_name="Citation Precision",
    )

    for model in models:
        row: list[float] = []
        for ds in datasets:
            result = report.get_result(model, ds)
            row.append(result.mean_citation_precision if result else 0.0)
        spec.values.append(row)

    return spec


def build_scaling_line_chart(
    n_views: list[int],
    bounds: list[float],
    empirical: list[float] | None = None,
) -> LineChartSpec:
    """Build a line chart spec for inference-time scaling law.

    Args:
        n_views: list of N values
        bounds: theoretical upper bounds for each N
        empirical: optional empirical hallucination rates for each N
    """
    spec = LineChartSpec(
        title="Inference-Time Scaling Law: Hallucination Rate vs. N Views",
        x_label="Number of Views (N)",
        y_label="Hallucination Probability Upper Bound",
    )

    spec.series["Theoretical Bound"] = (
        [float(n) for n in n_views],
        bounds,
    )

    if empirical is not None:
        spec.series["Empirical Rate"] = (
            [float(n) for n in n_views],
            empirical,
        )

    return spec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_metric_table(report: BenchmarkReport, metric: str) -> list[str]:
    """Build a markdown table for a given metric across models and datasets."""
    models = list(ModelType)
    datasets = list(BenchmarkDataset)

    lines: list[str] = []

    # Header
    header = "| Model |"
    separator = "| --- |"
    for ds in datasets:
        header += f" {DATASET_DISPLAY.get(ds, ds.value)} |"
        separator += " --- |"
    header += " Avg |"
    separator += " --- |"
    lines.append(header)
    lines.append(separator)

    # Find best per column for bolding
    best_per_col: dict[BenchmarkDataset, float] = {}
    for ds in datasets:
        vals = []
        for model in models:
            result = report.get_result(model, ds)
            if result:
                vals.append(getattr(result, metric, 0.0))
        best_per_col[ds] = max(vals) if vals else 0.0

    # Data rows
    for model in models:
        row = f"| {DISPLAY_NAMES.get(model, model.value)} |"
        vals: list[float] = []
        for ds in datasets:
            result = report.get_result(model, ds)
            if result:
                val = getattr(result, metric, 0.0)
                vals.append(val)
                if val == best_per_col[ds] and val > 0:
                    row += f" **{val:.3f}** |"
                else:
                    row += f" {val:.3f} |"
            else:
                row += " -- |"
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f" {avg:.3f} |"
        lines.append(row)

    return lines
