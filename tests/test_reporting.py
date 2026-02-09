"""Tests for the reporting and visualization module."""

import json
import pytest

from etg_rlm.benchmark_runner import (
    BenchmarkDataset,
    BenchmarkReport,
    DatasetResults,
    ModelType,
)
from etg_rlm.reporting import (
    PlotDataPoint,
    BarChartSpec,
    LineChartSpec,
    HeatmapSpec,
    generate_markdown_report,
    generate_latex_table,
    generate_json_report,
    build_factscore_bar_chart,
    build_citation_heatmap,
    build_scaling_line_chart,
    DISPLAY_NAMES,
    DATASET_DISPLAY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report() -> BenchmarkReport:
    """Create a sample benchmark report for testing."""
    report = BenchmarkReport()

    # ETG results
    for ds, fs, cp, cr, rf in [
        (BenchmarkDataset.TRUTHFUL_QA, 0.95, 0.92, 0.88, 0.72),
        (BenchmarkDataset.HALU_EVAL, 0.93, 0.90, 0.85, 0.68),
        (BenchmarkDataset.HOTPOT_QA, 0.91, 0.88, 0.82, 0.65),
        (BenchmarkDataset.NATURAL_QUESTIONS, 0.94, 0.91, 0.87, 0.70),
        (BenchmarkDataset.ELI5, 0.89, 0.85, 0.80, 0.62),
    ]:
        report.add_result(DatasetResults(
            model=ModelType.ETG, dataset=ds, n_instances=100,
            mean_factscore=fs, mean_claim_precision=fs,
            mean_citation_precision=cp, mean_citation_recall=cr,
            mean_rouge_f1=rf,
        ))

    # Zero-shot results (baseline)
    for ds, fs in [
        (BenchmarkDataset.TRUTHFUL_QA, 0.55),
        (BenchmarkDataset.HALU_EVAL, 0.50),
        (BenchmarkDataset.HOTPOT_QA, 0.45),
        (BenchmarkDataset.NATURAL_QUESTIONS, 0.60),
        (BenchmarkDataset.ELI5, 0.48),
    ]:
        report.add_result(DatasetResults(
            model=ModelType.ZERO_SHOT, dataset=ds, n_instances=100,
            mean_factscore=fs, mean_claim_precision=fs,
            mean_citation_precision=0.0, mean_citation_recall=0.0,
            mean_rouge_f1=0.60,
        ))

    report.compute_rankings()
    return report


# ---------------------------------------------------------------------------
# Test markdown report
# ---------------------------------------------------------------------------


class TestMarkdownReport:
    def test_generates_report(self):
        report = _make_report()
        md = generate_markdown_report(report)
        assert "# Canonical Evaluation: ETG vs. Baselines" in md
        assert "## Executive Summary" in md
        assert "## FactScore Comparison" in md

    def test_contains_etg_summary(self):
        report = _make_report()
        md = generate_markdown_report(report)
        assert "ETG Mean FactScore" in md

    def test_contains_tables(self):
        report = _make_report()
        md = generate_markdown_report(report)
        assert "| Model |" in md
        assert "| --- |" in md

    def test_best_values_bolded(self):
        report = _make_report()
        md = generate_markdown_report(report)
        # ETG should be bolded (best values)
        assert "**0.95" in md or "**0.950**" in md

    def test_rankings_included(self):
        report = _make_report()
        md = generate_markdown_report(report)
        assert "## Model Rankings" in md
        assert "mean_factscore" in md

    def test_empty_report(self):
        report = BenchmarkReport()
        md = generate_markdown_report(report)
        assert "# Canonical Evaluation" in md

    def test_citation_and_rouge_tables(self):
        report = _make_report()
        md = generate_markdown_report(report)
        assert "## Citation Precision" in md
        assert "## Citation Recall" in md
        assert "## ROUGE-L F1" in md


# ---------------------------------------------------------------------------
# Test LaTeX table
# ---------------------------------------------------------------------------


class TestLatexTable:
    def test_generates_table(self):
        report = _make_report()
        latex = generate_latex_table(report)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex

    def test_contains_model_names(self):
        report = _make_report()
        latex = generate_latex_table(report)
        assert "ETG (Ours)" in latex
        assert "Zero-Shot GPT-4" in latex

    def test_contains_dataset_names(self):
        report = _make_report()
        latex = generate_latex_table(report)
        assert "TruthfulQA" in latex
        assert "HotpotQA" in latex

    def test_custom_caption(self):
        report = _make_report()
        latex = generate_latex_table(
            report, caption="Custom caption", label="tab:custom"
        )
        assert "Custom caption" in latex
        assert "tab:custom" in latex


# ---------------------------------------------------------------------------
# Test JSON report
# ---------------------------------------------------------------------------


class TestJsonReport:
    def test_valid_json(self):
        report = _make_report()
        json_str = generate_json_report(report)
        data = json.loads(json_str)
        assert "results" in data
        assert "rankings" in data

    def test_contains_all_results(self):
        report = _make_report()
        json_str = generate_json_report(report)
        data = json.loads(json_str)
        # 5 datasets x 2 models = 10 results
        assert len(data["results"]) == 10

    def test_numeric_values_rounded(self):
        report = _make_report()
        json_str = generate_json_report(report)
        data = json.loads(json_str)
        for key, result in data["results"].items():
            assert isinstance(result["mean_factscore"], float)
            # Check that values are rounded to 4 decimal places
            assert len(str(result["mean_factscore"]).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# Test visualization specs
# ---------------------------------------------------------------------------


class TestVisualizationSpecs:
    def test_bar_chart_spec(self):
        report = _make_report()
        spec = build_factscore_bar_chart(report)
        assert spec.title  # Non-empty
        assert len(spec.groups) == len(ModelType)
        assert len(spec.x_categories) == len(BenchmarkDataset)
        assert len(spec.data) == len(ModelType) * len(BenchmarkDataset)

    def test_citation_heatmap(self):
        report = _make_report()
        spec = build_citation_heatmap(report)
        assert len(spec.row_labels) == len(ModelType)
        assert len(spec.col_labels) == len(BenchmarkDataset)
        assert len(spec.values) == len(ModelType)
        assert len(spec.values[0]) == len(BenchmarkDataset)

    def test_scaling_line_chart(self):
        n_views = [1, 5, 10, 20]
        bounds = [0.5, 0.01, 0.0001, 1e-8]
        spec = build_scaling_line_chart(n_views, bounds)
        assert "Theoretical Bound" in spec.series
        assert len(spec.series["Theoretical Bound"][0]) == 4

    def test_scaling_with_empirical(self):
        n_views = [1, 5, 10]
        bounds = [0.5, 0.01, 0.0001]
        empirical = [0.4, 0.008, 0.00005]
        spec = build_scaling_line_chart(n_views, bounds, empirical)
        assert "Empirical Rate" in spec.series
        assert len(spec.series) == 2

    def test_plot_data_point(self):
        p = PlotDataPoint(x_label="TruthfulQA", y_value=0.95, group="ETG")
        assert p.x_label == "TruthfulQA"
        assert p.y_value == 0.95
        assert p.error_low == 0.0  # Default


# ---------------------------------------------------------------------------
# Test display name mappings
# ---------------------------------------------------------------------------


class TestDisplayNames:
    def test_all_models_have_names(self):
        for model in ModelType:
            assert model in DISPLAY_NAMES

    def test_all_datasets_have_names(self):
        for ds in BenchmarkDataset:
            assert ds in DATASET_DISPLAY
