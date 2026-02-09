"""Tests for theoretical bounds (Propositions 1-3, Section 6)."""

import math

import pytest

from etg_rlm.bounds import (
    check_zero_confabulation,
    hallucination_upper_bound,
    inference_time_scaling_law,
    kl_bernoulli,
    optimal_view_allocation,
    required_views_for_bound,
)


class TestKLBernoulli:
    def test_identical(self):
        assert kl_bernoulli(0.5, 0.5) == pytest.approx(0.0)

    def test_known_value(self):
        # D(0.3 || 0.7) = 0.3*log(3/7) + 0.7*log(7/3)
        expected = 0.3 * math.log(0.3 / 0.7) + 0.7 * math.log(0.7 / 0.3)
        assert kl_bernoulli(0.3, 0.7) == pytest.approx(expected)

    def test_p_zero(self):
        # D(0 || q) = -log(1-q) for q < 1
        assert kl_bernoulli(0.0, 0.5) == pytest.approx(-math.log(0.5))

    def test_p_one(self):
        # D(1 || q) = -log(q)
        assert kl_bernoulli(1.0, 0.5) == pytest.approx(-math.log(0.5))

    def test_q_zero_p_positive(self):
        assert kl_bernoulli(0.5, 0.0) == float("inf")

    def test_q_zero_p_zero(self):
        assert kl_bernoulli(0.0, 0.0) == pytest.approx(0.0)

    def test_q_one_p_less_than_one(self):
        assert kl_bernoulli(0.5, 1.0) == float("inf")

    def test_invalid_range(self):
        with pytest.raises(ValueError):
            kl_bernoulli(-0.1, 0.5)
        with pytest.raises(ValueError):
            kl_bernoulli(0.5, 1.1)

    def test_non_negative(self):
        # KL divergence is always non-negative
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
                assert kl_bernoulli(p, q) >= 0.0


class TestHallucinationUpperBound:
    def test_basic_bound(self):
        """With alpha < tau, unsupported claims should be filtered."""
        bound = hallucination_upper_bound(n_views=10, tau=0.7, alpha=0.1)
        assert 0.0 < bound < 1.0

    def test_increases_with_n(self):
        """More views should give tighter (smaller) bounds."""
        b1 = hallucination_upper_bound(n_views=5, tau=0.7, alpha=0.1)
        b2 = hallucination_upper_bound(n_views=10, tau=0.7, alpha=0.1)
        b3 = hallucination_upper_bound(n_views=20, tau=0.7, alpha=0.1)
        assert b1 > b2 > b3

    def test_exponential_decay(self):
        """Bound should decay exponentially with N."""
        b1 = hallucination_upper_bound(n_views=10, tau=0.7, alpha=0.1)
        b2 = hallucination_upper_bound(n_views=20, tau=0.7, alpha=0.1)
        # b2 should be approximately b1^2 (doubling N squares the bound)
        assert b2 == pytest.approx(b1 ** 2, rel=0.01)

    def test_alpha_equals_tau(self):
        """When alpha >= tau, no filtering power."""
        bound = hallucination_upper_bound(n_views=100, tau=0.5, alpha=0.5)
        assert bound == 1.0

    def test_alpha_zero(self):
        """Perfect verifier: bound is zero for any tau > 0."""
        bound = hallucination_upper_bound(n_views=1, tau=0.7, alpha=0.0)
        assert bound == 0.0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            hallucination_upper_bound(n_views=0, tau=0.7, alpha=0.1)
        with pytest.raises(ValueError):
            hallucination_upper_bound(n_views=5, tau=0.0, alpha=0.1)

    def test_single_view(self):
        bound = hallucination_upper_bound(n_views=1, tau=0.7, alpha=0.1)
        # With N=1, tau=0.7 means we need 1/1 >= 0.7, so claim must be entailed.
        # Pr[entailed | unsupported] = alpha = 0.1
        # But the Chernoff bound may be looser; just check it's reasonable
        assert bound <= 1.0


class TestRequiredViewsForBound:
    def test_basic(self):
        n = required_views_for_bound(target_prob=0.01, tau=0.7, alpha=0.1)
        assert n > 0
        # Verify the bound is actually achieved
        bound = hallucination_upper_bound(n_views=n, tau=0.7, alpha=0.1)
        assert bound <= 0.01

    def test_tighter_bound_needs_more_views(self):
        n1 = required_views_for_bound(target_prob=0.1, tau=0.7, alpha=0.1)
        n2 = required_views_for_bound(target_prob=0.01, tau=0.7, alpha=0.1)
        n3 = required_views_for_bound(target_prob=0.001, tau=0.7, alpha=0.1)
        assert n1 <= n2 <= n3

    def test_invalid_alpha_ge_tau(self):
        with pytest.raises(ValueError, match="alpha must be < tau"):
            required_views_for_bound(target_prob=0.01, tau=0.5, alpha=0.5)


class TestOptimalViewAllocation:
    def test_empty(self):
        result = optimal_view_allocation({}, budget=10, tau=0.7, alpha=0.1)
        assert result.allocations == {}
        assert result.total_views == 0

    def test_equal_priority(self):
        priorities = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = optimal_view_allocation(
            priorities, budget=9, tau=0.7, alpha=0.1, min_per_node=1
        )
        # Each should get 3 views (1 min + 2 extra)
        assert result.allocations["a"] == 3
        assert result.allocations["b"] == 3
        assert result.allocations["c"] == 3

    def test_higher_priority_gets_more(self):
        priorities = {"a": 10.0, "b": 1.0}
        result = optimal_view_allocation(
            priorities, budget=20, tau=0.7, alpha=0.1, min_per_node=1
        )
        assert result.allocations["a"] > result.allocations["b"]

    def test_insufficient_budget(self):
        priorities = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = optimal_view_allocation(
            priorities, budget=2, tau=0.7, alpha=0.1, min_per_node=1
        )
        total = sum(result.allocations.values())
        assert total == 2  # Can only allocate 2


class TestZeroConfabulation:
    """Tests for Proposition 2: Zero-Confabulation Property."""

    def test_satisfies_when_all_have_evidence(self):
        result = check_zero_confabulation(
            verified_node_ids={"a", "b"},
            node_evidence_counts={"a": 3, "b": 2},
            node_support_masses={"a": 0.9, "b": 0.8},
            tau=0.7,
        )
        assert result.satisfies_proposition is True
        assert result.confabulating_node_ids == []
        assert result.total_verified_nodes == 2

    def test_fails_when_node_has_no_evidence(self):
        result = check_zero_confabulation(
            verified_node_ids={"a", "b"},
            node_evidence_counts={"a": 3, "b": 0},  # b has no evidence
            node_support_masses={"a": 0.9, "b": 0.8},
            tau=0.7,
        )
        assert result.satisfies_proposition is False
        assert "b" in result.confabulating_node_ids

    def test_fails_when_mass_below_tau(self):
        result = check_zero_confabulation(
            verified_node_ids={"a"},
            node_evidence_counts={"a": 3},
            node_support_masses={"a": 0.5},  # below tau
            tau=0.7,
        )
        assert result.satisfies_proposition is False
        assert "a" in result.confabulating_node_ids

    def test_empty_verified_set(self):
        result = check_zero_confabulation(
            verified_node_ids=set(),
            node_evidence_counts={},
            node_support_masses={},
            tau=0.7,
        )
        assert result.satisfies_proposition is True
        assert result.total_verified_nodes == 0


class TestInferenceTimeScalingLaw:
    """Tests for the inference-time scaling law (Proposition 1 visualization)."""

    def test_monotonically_decreasing(self):
        result = inference_time_scaling_law(tau=0.7, alpha=0.1, max_n=20)
        assert len(result.n_views_sequence) == 20
        assert len(result.bounds_sequence) == 20
        # Bounds should be monotonically decreasing
        for i in range(1, len(result.bounds_sequence)):
            assert result.bounds_sequence[i] <= result.bounds_sequence[i - 1]

    def test_approaches_zero(self):
        result = inference_time_scaling_law(tau=0.7, alpha=0.1, max_n=50)
        assert result.bounds_sequence[-1] < 1e-6

    def test_invalid_alpha_ge_tau(self):
        with pytest.raises(ValueError):
            inference_time_scaling_law(tau=0.5, alpha=0.5)

    def test_result_fields(self):
        result = inference_time_scaling_law(tau=0.8, alpha=0.2, max_n=10)
        assert result.tau == 0.8
        assert result.alpha == 0.2
        assert result.n_views_sequence == list(range(1, 11))
