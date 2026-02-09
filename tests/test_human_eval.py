"""Tests for the human evaluation protocol module."""

import pytest

from etg_rlm.human_eval import (
    FaithfulnessAnnotation,
    FaithfulnessAggregation,
    FaithfulnessRating,
    HumanEvalSummary,
    PairwiseAnnotation,
    PreferenceAggregation,
    PreferenceChoice,
    PreferenceDimension,
    aggregate_faithfulness,
    aggregate_preferences,
    check_annotator_agreement,
    fleiss_kappa,
)


class TestFaithfulnessRating:
    def test_values(self):
        assert FaithfulnessRating.COMPLETELY_UNFAITHFUL == 1
        assert FaithfulnessRating.FULLY_FAITHFUL == 5

    def test_ordering(self):
        assert FaithfulnessRating.MOSTLY_UNFAITHFUL < FaithfulnessRating.PARTIALLY_FAITHFUL
        assert FaithfulnessRating.MOSTLY_FAITHFUL < FaithfulnessRating.FULLY_FAITHFUL


class TestAggregateFaithfulness:
    def test_basic(self):
        annotations = [
            FaithfulnessAnnotation("q1", "ETG", "a1", FaithfulnessRating.FULLY_FAITHFUL),
            FaithfulnessAnnotation("q1", "ETG", "a2", FaithfulnessRating.MOSTLY_FAITHFUL),
            FaithfulnessAnnotation("q1", "ETG", "a3", FaithfulnessRating.FULLY_FAITHFUL),
        ]
        result = aggregate_faithfulness(annotations, "ETG")
        assert result.n_annotations == 3
        assert result.mean_rating == pytest.approx((5 + 4 + 5) / 3.0)
        assert result.median_rating == 5.0

    def test_filters_by_system(self):
        annotations = [
            FaithfulnessAnnotation("q1", "ETG", "a1", FaithfulnessRating.FULLY_FAITHFUL),
            FaithfulnessAnnotation("q1", "RAG", "a1", FaithfulnessRating.MOSTLY_UNFAITHFUL),
        ]
        result = aggregate_faithfulness(annotations, "ETG")
        assert result.n_annotations == 1
        assert result.mean_rating == 5.0

    def test_empty(self):
        result = aggregate_faithfulness([], "ETG")
        assert result.n_annotations == 0
        assert result.mean_rating == 0.0

    def test_distribution(self):
        annotations = [
            FaithfulnessAnnotation("q1", "S", "a1", FaithfulnessRating.FULLY_FAITHFUL),
            FaithfulnessAnnotation("q2", "S", "a1", FaithfulnessRating.FULLY_FAITHFUL),
            FaithfulnessAnnotation("q3", "S", "a1", FaithfulnessRating.MOSTLY_FAITHFUL),
        ]
        result = aggregate_faithfulness(annotations, "S")
        assert result.rating_distribution[5] == 2
        assert result.rating_distribution[4] == 1

    def test_even_median(self):
        annotations = [
            FaithfulnessAnnotation("q1", "S", "a1", FaithfulnessRating.MOSTLY_FAITHFUL),
            FaithfulnessAnnotation("q2", "S", "a2", FaithfulnessRating.FULLY_FAITHFUL),
        ]
        result = aggregate_faithfulness(annotations, "S")
        assert result.median_rating == pytest.approx(4.5)


class TestAggregatePreferences:
    def test_basic(self):
        annotations = [
            PairwiseAnnotation(
                "q1", "ETG", "RAG", "a1",
                {PreferenceDimension.OVERALL_BETTER: PreferenceChoice.SYSTEM_A},
            ),
            PairwiseAnnotation(
                "q1", "ETG", "RAG", "a2",
                {PreferenceDimension.OVERALL_BETTER: PreferenceChoice.SYSTEM_A},
            ),
            PairwiseAnnotation(
                "q1", "ETG", "RAG", "a3",
                {PreferenceDimension.OVERALL_BETTER: PreferenceChoice.SYSTEM_B},
            ),
        ]
        result = aggregate_preferences(annotations, PreferenceDimension.OVERALL_BETTER)
        assert result.a_wins == 2
        assert result.b_wins == 1
        assert result.ties == 0
        assert result.a_win_rate == pytest.approx(2 / 3)

    def test_with_ties(self):
        annotations = [
            PairwiseAnnotation(
                "q1", "A", "B", "a1",
                {PreferenceDimension.MORE_TRUSTWORTHY: PreferenceChoice.TIE},
            ),
        ]
        result = aggregate_preferences(annotations, PreferenceDimension.MORE_TRUSTWORTHY)
        assert result.ties == 1
        assert result.a_win_rate == 0.0

    def test_empty(self):
        result = aggregate_preferences([], PreferenceDimension.MORE_HELPFUL)
        assert result.n_comparisons == 0
        assert result.a_win_rate == 0.0


class TestFleissKappa:
    def test_perfect_agreement(self):
        # All 3 raters agree on all items
        matrix = [
            [3, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0],
        ]
        kappa = fleiss_kappa(matrix)
        assert kappa == pytest.approx(1.0)

    def test_no_agreement_beyond_chance(self):
        # Uniform distribution of ratings => negative kappa (less than chance)
        matrix = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        kappa = fleiss_kappa(matrix)
        # With perfectly uniform ratings, kappa is negative (-0.5)
        assert kappa < 0.0

    def test_moderate_agreement(self):
        # Some agreement but not perfect
        matrix = [
            [3, 0],  # all agree on cat 1
            [2, 1],  # mostly agree on cat 1
            [0, 3],  # all agree on cat 2
            [1, 2],  # mostly agree on cat 2
        ]
        kappa = fleiss_kappa(matrix)
        assert 0.3 < kappa < 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fleiss_kappa([])

    def test_single_category_raises(self):
        with pytest.raises(ValueError, match="at least 2 categories"):
            fleiss_kappa([[3]])

    def test_single_rater_raises(self):
        with pytest.raises(ValueError, match="at least 2 raters"):
            fleiss_kappa([[1, 0], [0, 1]])

    def test_inconsistent_categories_raises(self):
        with pytest.raises(ValueError, match="Inconsistent number of categories"):
            fleiss_kappa([[2, 1], [3]])

    def test_inconsistent_raters_raises(self):
        with pytest.raises(ValueError, match="Inconsistent number of raters"):
            fleiss_kappa([[2, 1], [2, 2]])


class TestCheckAnnotatorAgreement:
    def test_acceptable(self):
        assert check_annotator_agreement(0.7) is True

    def test_unacceptable(self):
        assert check_annotator_agreement(0.5) is False

    def test_at_threshold(self):
        assert check_annotator_agreement(0.6) is True

    def test_custom_threshold(self):
        assert check_annotator_agreement(0.4, threshold=0.3) is True


class TestHumanEvalSummary:
    def test_creation(self):
        summary = HumanEvalSummary(
            n_instances=200,
            n_annotators=3,
            fleiss_kappa=0.72,
            agreement_acceptable=True,
        )
        assert summary.n_instances == 200
        assert summary.n_annotators == 3
        assert summary.agreement_acceptable is True
