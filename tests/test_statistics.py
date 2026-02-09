"""Tests for the statistical analysis module."""

import math

import pytest

from etg_rlm.statistics import (
    BootstrapCIResult,
    EffectSizeResult,
    PairedTTestResult,
    StatisticalAnalysis,
    bootstrap_ci,
    bootstrap_paired_ci,
    cohens_d,
    full_analysis,
    paired_t_test,
)


class TestPairedTTest:
    def test_identical_inputs(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_t_test(x, x)
        assert result.t_statistic == 0.0
        assert result.p_value == 1.0
        assert result.significant is False

    def test_clearly_different(self):
        x = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        result = paired_t_test(x, y)
        assert result.significant is True
        assert result.t_statistic > 0
        assert result.p_value < 0.05
        assert result.mean_diff == pytest.approx(10.0)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            paired_t_test([1.0, 2.0], [1.0])

    def test_too_few_pairs(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_t_test([1.0], [2.0])

    def test_constant_nonzero_diff(self):
        x = [5.0, 5.0, 5.0]
        y = [3.0, 3.0, 3.0]
        result = paired_t_test(x, y)
        assert result.significant is True
        assert result.std_diff == 0.0

    def test_n_pairs(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.1, 2.1, 3.1, 4.1]
        result = paired_t_test(x, y)
        assert result.n_pairs == 4

    def test_custom_alpha(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.5, 2.5, 3.5, 4.5, 5.5]
        result_strict = paired_t_test(x, y, alpha=0.001)
        # With only 5 samples and diff=0.5, may not be significant at 0.001
        # but should be at 0.05
        result_lenient = paired_t_test(x, y, alpha=0.05)
        assert result_lenient.significant is True


class TestCohensD:
    def test_identical(self):
        x = [1.0, 2.0, 3.0]
        result = cohens_d(x, x)
        assert result.cohens_d == 0.0
        assert result.interpretation == "negligible"

    def test_large_effect(self):
        x = [10.0, 11.0, 12.0, 13.0, 14.0]
        y = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = cohens_d(x, y)
        assert abs(result.cohens_d) >= 0.8
        assert result.interpretation == "large"

    def test_small_effect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [0.7, 1.7, 2.7, 3.7, 4.7]
        result = cohens_d(x, y)
        assert abs(result.cohens_d) < 0.5
        assert result.interpretation in ("negligible", "small")

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_d([1.0], [1.0, 2.0])

    def test_too_few(self):
        with pytest.raises(ValueError, match="at least 2"):
            cohens_d([1.0], [2.0])

    def test_mean_diff(self):
        x = [10.0, 20.0, 30.0]
        y = [5.0, 15.0, 25.0]
        result = cohens_d(x, y)
        assert result.mean_diff == pytest.approx(5.0)


class TestBootstrapCI:
    def test_basic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bootstrap_ci(data, seed=42, n_bootstrap=1000)
        assert result.point_estimate == pytest.approx(3.0)
        assert result.ci_lower <= result.point_estimate
        assert result.ci_upper >= result.point_estimate
        assert result.confidence_level == 0.95

    def test_narrow_ci_for_constant(self):
        data = [5.0] * 20
        result = bootstrap_ci(data, seed=42, n_bootstrap=1000)
        assert result.ci_lower == pytest.approx(5.0)
        assert result.ci_upper == pytest.approx(5.0)

    def test_median(self):
        data = [1.0, 2.0, 3.0, 100.0, 200.0]
        result = bootstrap_ci(data, statistic="median", seed=42, n_bootstrap=1000)
        assert result.point_estimate == 3.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            bootstrap_ci([])

    def test_bad_confidence(self):
        with pytest.raises(ValueError, match="Confidence"):
            bootstrap_ci([1.0], confidence=1.5)

    def test_reproducible(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = bootstrap_ci(data, seed=42, n_bootstrap=500)
        r2 = bootstrap_ci(data, seed=42, n_bootstrap=500)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


class TestBootstrapPairedCI:
    def test_basic(self):
        x = [0.01, 0.02, 0.005, 0.015, 0.01]
        y = [0.15, 0.20, 0.18, 0.12, 0.16]
        result = bootstrap_paired_ci(x, y, seed=42, n_bootstrap=1000)
        # ETG (x) has lower hallucination than RAG (y), so diff should be negative
        assert result.point_estimate < 0

    def test_mismatched(self):
        with pytest.raises(ValueError, match="same length"):
            bootstrap_paired_ci([1.0], [1.0, 2.0])


class TestFullAnalysis:
    def test_complete_analysis(self):
        # Simulate ETG (low hallucination) vs RAG (high hallucination)
        etg = [0.01, 0.02, 0.005, 0.015, 0.01, 0.008, 0.012, 0.009, 0.011, 0.007]
        rag = [0.15, 0.20, 0.18, 0.12, 0.16, 0.14, 0.19, 0.17, 0.13, 0.21]
        result = full_analysis(
            etg, rag,
            metric_name="hallucination_rate",
            n_bootstrap=500,
            seed=42,
        )
        assert result.metric_name == "hallucination_rate"
        assert result.t_test.significant is True
        assert result.effect_size.interpretation == "large"
        assert result.etg_ci.point_estimate < result.baseline_ci.point_estimate
        assert result.diff_ci.point_estimate < 0

    def test_metric_name(self):
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        result = full_analysis(x, y, metric_name="rouge_l", n_bootstrap=100, seed=0)
        assert result.metric_name == "rouge_l"
