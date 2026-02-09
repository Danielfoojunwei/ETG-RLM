"""Tests for RLM recursion policies."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.policy import (
    ActionType,
    GreedyBudgetPolicy,
    UtilityWeightedPolicy,
)


def _make_graph(*nodes: tuple[str, float, int]) -> EvidenceScopedBeliefGraph:
    """Helper: create ESBG with nodes specified as (id, support_mass, n_views)."""
    g = EvidenceScopedBeliefGraph()
    for nid, mass, n_views in nodes:
        node = ESBGNode(
            node_id=nid,
            claim=AtomicClaim(claim_id=nid, text=f"Claim {nid}"),
            support_mass=mass,
            view_verdicts=[True] * int(mass * n_views) + [False] * (n_views - int(mass * n_views)),
        )
        if mass >= 0.7:
            node.status = ClaimStatus.ENTAILED
            node.claim_type = ClaimType.VERIFIED
        elif mass <= 0.3:
            node.claim_type = ClaimType.UNSUPPORTED
        else:
            node.claim_type = ClaimType.UNCERTAIN
        g.add_node(node)
    return g


class TestGreedyBudgetPolicy:
    def test_allocates_to_under_viewed(self):
        policy = GreedyBudgetPolicy(tau=0.7, min_views=3)
        g = _make_graph(
            ("a", 0.8, 3),  # Has 3 views (at min)
            ("b", 0.5, 1),  # Has 1 view (under min)
        )
        action = policy.select_action("q", "c", g, budget_remaining=5)
        assert action.action_type == ActionType.RUN_VIEW
        assert action.target_node_id == "b"  # Under min_views

    def test_stops_on_zero_budget(self):
        policy = GreedyBudgetPolicy(tau=0.7, min_views=3)
        g = _make_graph(("a", 0.5, 1))
        action = policy.select_action("q", "c", g, budget_remaining=0)
        assert action.action_type == ActionType.STOP

    def test_stops_on_empty_graph(self):
        policy = GreedyBudgetPolicy(tau=0.7, min_views=3)
        g = EvidenceScopedBeliefGraph()
        action = policy.select_action("q", "c", g, budget_remaining=10)
        assert action.action_type == ActionType.STOP

    def test_phase2_targets_threshold_boundary(self):
        policy = GreedyBudgetPolicy(tau=0.7, min_views=2)
        g = _make_graph(
            ("a", 0.65, 3),  # Close to threshold
            ("b", 0.2, 3),   # Far from threshold
        )
        action = policy.select_action("q", "c", g, budget_remaining=5)
        assert action.action_type == ActionType.RUN_VIEW
        assert action.target_node_id == "a"  # Closer to tau=0.7


class TestUtilityWeightedPolicy:
    def test_basic_allocation(self):
        policy = UtilityWeightedPolicy(tau=0.7)
        g = _make_graph(("a", 0.5, 1), ("b", 0.3, 1))
        action = policy.select_action("q", "c", g, budget_remaining=5)
        assert action.action_type == ActionType.RUN_VIEW
        assert action.target_node_id is not None

    def test_stops_when_exhausted(self):
        policy = UtilityWeightedPolicy(tau=0.7)
        g = EvidenceScopedBeliefGraph()
        action = policy.select_action("q", "c", g, budget_remaining=5)
        assert action.action_type == ActionType.STOP

    def test_risk_labels_affect_priority(self):
        policy = UtilityWeightedPolicy(
            tau=0.7,
            risk_weight=10.0,
            uncertainty_weight=0.0,
            utility_weight=0.0,
            risk_labels={"a": 0.1, "b": 1.0},
        )
        g = _make_graph(("a", 0.5, 1), ("b", 0.5, 1))
        action = policy.select_action("q", "c", g, budget_remaining=5)
        assert action.target_node_id == "b"  # Higher risk
