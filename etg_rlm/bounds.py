"""Theoretical bounds for Evidence-Typed Generation.

Proposition 1 (Hallucination upper bound):
    If the verifier has false-positive rate alpha per view, then any
    unsupported claim has:
        Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))
    where D is the KL divergence for Bernoulli distributions.

    This gives an inference-time scaling law for faithfulness:
    by increasing N, unsupported claims become exponentially unlikely
    to pass the support-mass gate.

Proposition 2 (Compute-allocation optimality):
    Given a budget B for verifier calls, the optimal allocation is
    formalized as a knapsack / bandit objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def kl_bernoulli(p: float, q: float) -> float:
    """Compute KL divergence D(p || q) for Bernoulli distributions.

    D(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))

    With conventions: 0*log(0) = 0, and D = +inf if q=0 and p>0.
    """
    if p < 0 or p > 1 or q < 0 or q > 1:
        raise ValueError(f"p and q must be in [0,1], got p={p}, q={q}")

    # Edge cases
    if q == 0.0:
        return float("inf") if p > 0.0 else 0.0
    if q == 1.0:
        return float("inf") if p < 1.0 else 0.0
    if p == 0.0:
        return -math.log(1.0 - q) if q < 1.0 else float("inf")
    if p == 1.0:
        return -math.log(q)

    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def hallucination_upper_bound(
    n_views: int,
    tau: float,
    alpha: float,
) -> float:
    """Compute the hallucination upper bound from Proposition 1.

    For an unsupported claim with verifier false-positive rate alpha:
        Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))

    This bound uses Sanov's theorem / Chernoff bound for binomial tails.

    Args:
        n_views: N, number of independent verification views
        tau: support mass threshold for the Verified type
        alpha: per-view false-positive rate of the verifier

    Returns:
        Upper bound on the probability that an unsupported claim
        passes the support-mass gate.

    Raises:
        ValueError: if parameters are out of valid ranges
    """
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")

    # If alpha >= tau, the bound is trivially 1 (no filtering power)
    if alpha >= tau:
        return 1.0

    d = kl_bernoulli(tau, alpha)
    if math.isinf(d):
        return 0.0

    return math.exp(-n_views * d)


def required_views_for_bound(
    target_prob: float,
    tau: float,
    alpha: float,
) -> int:
    """Compute the minimum N to achieve a target hallucination probability.

    Solves: exp(-N * D(tau || alpha)) <= target_prob
    => N >= -log(target_prob) / D(tau || alpha)

    Args:
        target_prob: desired upper bound on hallucination probability
        tau: support mass threshold
        alpha: per-view false-positive rate

    Returns:
        Minimum number of views N needed.
    """
    if not (0.0 < target_prob < 1.0):
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob}")
    if alpha >= tau:
        raise ValueError(
            f"alpha must be < tau for filtering to work, "
            f"got alpha={alpha}, tau={tau}"
        )

    d = kl_bernoulli(tau, alpha)
    if d == 0.0:
        raise ValueError("KL divergence is 0; cannot achieve bound")

    n = -math.log(target_prob) / d
    return math.ceil(n)


@dataclass
class ViewAllocationResult:
    """Result of optimal view allocation across claims."""

    allocations: dict[str, int]  # node_id -> number of views to allocate
    total_views: int
    expected_false_pass_rate: float


def optimal_view_allocation(
    node_priorities: dict[str, float],
    budget: int,
    tau: float,
    alpha: float,
    min_per_node: int = 1,
) -> ViewAllocationResult:
    """Allocate a verification budget across claims optimally.

    Implements the knapsack / bandit objective from Proposition 2:
    allocate more views to claims with higher priority scores.

    The priority score combines:
        - utility contribution
        - uncertainty (support mass near threshold)
        - risk (safety-critical claims)

    Uses a greedy proportional allocation:
        n_i = min_per_node + floor((B - k*min_per_node) * p_i / sum(p_j))
    where k is the number of nodes and p_i is the priority of node i.

    Args:
        node_priorities: mapping from node_id to priority score
        budget: total verification budget B
        tau: support mass threshold
        alpha: per-view false-positive rate
        min_per_node: minimum views allocated to each node

    Returns:
        ViewAllocationResult with per-node allocations.
    """
    k = len(node_priorities)
    if k == 0:
        return ViewAllocationResult(
            allocations={}, total_views=0, expected_false_pass_rate=0.0
        )

    min_budget = k * min_per_node
    if budget < min_budget:
        # Not enough budget for minimums; allocate evenly
        per_node = budget // k
        remainder = budget % k
        allocs = {}
        for i, nid in enumerate(node_priorities):
            allocs[nid] = per_node + (1 if i < remainder else 0)
    else:
        remaining = budget - min_budget
        total_priority = sum(node_priorities.values())

        allocs = {}
        if total_priority > 0:
            for nid, pri in node_priorities.items():
                extra = int(remaining * pri / total_priority)
                allocs[nid] = min_per_node + extra
        else:
            # Equal allocation if all priorities are zero
            extra_each = remaining // k
            for nid in node_priorities:
                allocs[nid] = min_per_node + extra_each

    # Compute expected false pass rate for each allocation
    total_views = sum(allocs.values())
    expected_fps: list[float] = []
    for nid, n_v in allocs.items():
        if n_v > 0 and alpha < tau:
            expected_fps.append(hallucination_upper_bound(n_v, tau, alpha))
        else:
            expected_fps.append(1.0)

    avg_fp = sum(expected_fps) / len(expected_fps) if expected_fps else 0.0

    return ViewAllocationResult(
        allocations=allocs,
        total_views=total_views,
        expected_false_pass_rate=avg_fp,
    )
