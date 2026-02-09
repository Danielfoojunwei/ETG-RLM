"""RLM Recursion Policy for ESBG construction.

The RLM recursion is a policy rho that chooses which claim/node to expand
next, and which view to run, given the current partial graph G_t:

    a_t ~ rho(a | q, E, G_t)

Actions include:
    - PROPOSE_CLAIM: propose a new claim node
    - ADD_DEPENDENCY: add a dependency edge
    - RUN_VIEW: run V_i for an existing claim
    - SEARCH_CONTRADICTIONS: search for contradicting evidence
    - SPLIT_CLAIM: split a claim into subclaims
    - STOP: terminate the recursion

The recursion ends at G_T.

Proposition 2 (Compute-allocation optimality):
Given a budget B for verifier calls, the best policy rho allocates more
views to claims with:
    - high utility contribution
    - high uncertainty (support mass near threshold)
    - high risk (safety-critical claims)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from etg_rlm.core import (
    AtomicClaim,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)


class ActionType(Enum):
    """Types of actions available to the recursion policy."""

    PROPOSE_CLAIM = "propose_claim"
    ADD_DEPENDENCY = "add_dependency"
    RUN_VIEW = "run_view"
    SEARCH_CONTRADICTIONS = "search_contradictions"
    SPLIT_CLAIM = "split_claim"
    STOP = "stop"


@dataclass
class PolicyAction:
    """An action chosen by the recursion policy.

    Attributes:
        action_type: the type of action
        target_node_id: the node to act on (for RUN_VIEW, SPLIT_CLAIM, etc.)
        claim: new claim to propose (for PROPOSE_CLAIM)
        from_node_id: source of a dependency edge (for ADD_DEPENDENCY)
        to_node_id: target of a dependency edge (for ADD_DEPENDENCY)
        view_index: which view to run (for RUN_VIEW)
    """

    action_type: ActionType
    target_node_id: str | None = None
    claim: AtomicClaim | None = None
    from_node_id: str | None = None
    to_node_id: str | None = None
    view_index: int | None = None


class RecursionPolicy(ABC):
    """Abstract base for RLM recursion policies.

    A policy rho selects the next action given the current state:
        a_t ~ rho(a | q, E, G_t)
    """

    @abstractmethod
    def select_action(
        self,
        query: str,
        corpus_id: str,
        esbg: EvidenceScopedBeliefGraph,
        budget_remaining: int,
    ) -> PolicyAction:
        """Select the next action given the current ESBG state.

        Args:
            query: the original query q
            corpus_id: identifier for the evidence corpus E
            esbg: the current partial graph G_t
            budget_remaining: remaining verifier call budget

        Returns:
            The next action to execute.
        """
        ...


class UtilityWeightedPolicy(RecursionPolicy):
    """A policy that allocates views based on utility-weighted priorities.

    Implements the compute-allocation strategy from Proposition 2:
    given a budget B, allocate more views to claims with:
        - high utility (how important is this claim to the answer)
        - high uncertainty (support mass near the threshold tau)
        - high risk (safety-critical claims)

    This is formalized as a knapsack / bandit objective.
    """

    def __init__(
        self,
        tau: float = 0.7,
        uncertainty_weight: float = 1.0,
        utility_weight: float = 1.0,
        risk_weight: float = 1.0,
        min_views_per_claim: int = 1,
        max_views_per_claim: int = 10,
        risk_labels: dict[str, float] | None = None,
    ) -> None:
        self.tau = tau
        self.uncertainty_weight = uncertainty_weight
        self.utility_weight = utility_weight
        self.risk_weight = risk_weight
        self.min_views_per_claim = min_views_per_claim
        self.max_views_per_claim = max_views_per_claim
        self.risk_labels = risk_labels or {}

    def select_action(
        self,
        query: str,
        corpus_id: str,
        esbg: EvidenceScopedBeliefGraph,
        budget_remaining: int,
    ) -> PolicyAction:
        """Select action by scoring unresolved nodes."""
        if budget_remaining <= 0:
            return PolicyAction(action_type=ActionType.STOP)

        nodes = esbg.nodes
        if not nodes:
            return PolicyAction(action_type=ActionType.STOP)

        # Score each node for view allocation priority
        best_score = -1.0
        best_node_id: str | None = None

        for nid, node in nodes.items():
            n_views = len(node.view_verdicts)

            # Skip nodes that have reached max views
            if n_views >= self.max_views_per_claim:
                continue

            # Skip fully resolved nodes (all views agree)
            if n_views >= self.min_views_per_claim and (
                node.claim_type == ClaimType.VERIFIED
                or node.claim_type == ClaimType.UNSUPPORTED
            ):
                # Only skip if support mass is far from thresholds
                if abs(node.support_mass - self.tau) > 0.2:
                    continue

            score = self._score_node(node)
            if score > best_score:
                best_score = score
                best_node_id = nid

        if best_node_id is None:
            return PolicyAction(action_type=ActionType.STOP)

        return PolicyAction(
            action_type=ActionType.RUN_VIEW,
            target_node_id=best_node_id,
        )

    def _score_node(self, node: ESBGNode) -> float:
        """Score a node for view allocation priority.

        Priority = w_u * uncertainty(m) + w_y * utility + w_r * risk

        Uncertainty is highest when support mass is near the threshold tau.
        """
        # Uncertainty: peaked near the threshold
        uncertainty = 1.0 - abs(node.support_mass - self.tau) / max(self.tau, 1e-9)
        uncertainty = max(0.0, uncertainty)

        # Utility: inverse of number of views already run (diminishing returns)
        n_views = max(len(node.view_verdicts), 1)
        utility = 1.0 / n_views

        # Risk: from external labels, default 0.5
        risk = self.risk_labels.get(node.node_id, 0.5)

        return (
            self.uncertainty_weight * uncertainty
            + self.utility_weight * utility
            + self.risk_weight * risk
        )


class GreedyBudgetPolicy(RecursionPolicy):
    """A simpler policy that distributes views evenly then refines uncertain claims.

    Phase 1: Ensure every claim has at least min_views verification views.
    Phase 2: Allocate remaining budget to claims nearest the threshold.
    """

    def __init__(
        self,
        tau: float = 0.7,
        min_views: int = 3,
    ) -> None:
        self.tau = tau
        self.min_views = min_views

    def select_action(
        self,
        query: str,
        corpus_id: str,
        esbg: EvidenceScopedBeliefGraph,
        budget_remaining: int,
    ) -> PolicyAction:
        if budget_remaining <= 0:
            return PolicyAction(action_type=ActionType.STOP)

        nodes = esbg.nodes

        # Phase 1: find a node that hasn't reached min_views
        for nid, node in nodes.items():
            if len(node.view_verdicts) < self.min_views:
                return PolicyAction(
                    action_type=ActionType.RUN_VIEW,
                    target_node_id=nid,
                )

        # Phase 2: find the node closest to the threshold
        closest_id: str | None = None
        closest_dist = float("inf")

        for nid, node in nodes.items():
            dist = abs(node.support_mass - self.tau)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = nid

        if closest_id is not None:
            return PolicyAction(
                action_type=ActionType.RUN_VIEW,
                target_node_id=closest_id,
            )

        return PolicyAction(action_type=ActionType.STOP)
