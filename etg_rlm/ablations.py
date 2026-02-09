"""Ablation study configurations for ETG (Section 3 of experimental design).

Defines four ablation experiments to understand the contribution of
each component of the ETG framework:

1. ETG-NoMultiView: N=1 single verification view
   -> Tests importance of multi-view stability

2. ETG-NoConstraint: ESBG built but generation not constrained
   -> Tests importance of constrained decoding vs. just scoring

3. ETG-Threshold-Sweep: varying tau (0.5, 0.6, 0.7, 0.8, 0.9)
   -> Characterizes the precision-recall trade-off

4. ETG-PolicyAblation: random claim selection instead of heuristic policy
   -> Tests importance of the recursive policy
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from etg_rlm.core import (
    AtomicClaim,
    ClaimType,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.pipeline import ETGConfig
from etg_rlm.policy import (
    ActionType,
    PolicyAction,
    RecursionPolicy,
)


class AblationType(Enum):
    """The four ablation experiments from the experimental design."""

    NO_MULTI_VIEW = "no_multi_view"
    NO_CONSTRAINT = "no_constraint"
    THRESHOLD_SWEEP = "threshold_sweep"
    POLICY_ABLATION = "policy_ablation"


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        ablation_type: which ablation
        name: human-readable name for reporting
        description: what this ablation tests
        etg_config: the modified ETGConfig for this ablation
    """

    ablation_type: AblationType
    name: str
    description: str
    etg_config: ETGConfig


# ---------------------------------------------------------------------------
# Ablation 4: Random Policy (replaces heuristic policy)
# ---------------------------------------------------------------------------


class RandomPolicy(RecursionPolicy):
    """Random claim selection policy for ablation study.

    Instead of the utility-weighted heuristic, selects claims randomly.
    Used to test the importance of the recursive policy (Ablation 4).
    """

    def __init__(
        self,
        max_views_per_claim: int = 5,
        seed: int | None = None,
    ) -> None:
        self.max_views_per_claim = max_views_per_claim
        self._rng = random.Random(seed)

    def select_action(
        self,
        query: str,
        corpus_id: str,
        esbg: EvidenceScopedBeliefGraph,
        budget_remaining: int,
    ) -> PolicyAction:
        """Select a random unresolved claim for verification."""
        if budget_remaining <= 0:
            return PolicyAction(action_type=ActionType.STOP)

        nodes = esbg.nodes
        if not nodes:
            return PolicyAction(action_type=ActionType.STOP)

        # Filter to claims that haven't reached max views
        eligible = [
            nid
            for nid, node in nodes.items()
            if len(node.view_verdicts) < self.max_views_per_claim
        ]

        if not eligible:
            return PolicyAction(action_type=ActionType.STOP)

        chosen = self._rng.choice(eligible)
        return PolicyAction(
            action_type=ActionType.RUN_VIEW,
            target_node_id=chosen,
        )


# ---------------------------------------------------------------------------
# Pre-defined ablation configurations
# ---------------------------------------------------------------------------


def make_no_multi_view_config(base: ETGConfig | None = None) -> AblationConfig:
    """Ablation 1: ETG with N=1 (single verification view).

    Tests the importance of multi-view stability (Definition 3).
    With N=1, the support mass is binary (0 or 1) and there is
    no multi-view consensus.
    """
    base = base or ETGConfig()
    return AblationConfig(
        ablation_type=AblationType.NO_MULTI_VIEW,
        name="ETG-NoMultiView",
        description=(
            "ETG with N=1 (single verification view). "
            "Tests the importance of multi-view stability."
        ),
        etg_config=ETGConfig(
            tau=base.tau,
            tau_prime=base.tau_prime,
            verification_budget=base.verification_budget,
            min_views_per_claim=1,
            allow_uncertain=base.allow_uncertain,
            corpus_id=base.corpus_id,
        ),
    )


def make_no_constraint_config(base: ETGConfig | None = None) -> AblationConfig:
    """Ablation 2: ESBG built but generation not constrained.

    The graph is still built and claims are scored, but
    unsupported claims are NOT blocked from the output.
    Tests constrained decoding vs. just scoring.
    """
    base = base or ETGConfig()
    return AblationConfig(
        ablation_type=AblationType.NO_CONSTRAINT,
        name="ETG-NoConstraint",
        description=(
            "ESBG is built and claims are scored, but generation is "
            "not constrained (all claims pass). Tests the importance "
            "of constrained decoding vs. just scoring."
        ),
        etg_config=ETGConfig(
            tau=0.0,  # Accept all claims
            tau_prime=0.0,
            verification_budget=base.verification_budget,
            min_views_per_claim=base.min_views_per_claim,
            allow_uncertain=True,
            corpus_id=base.corpus_id,
        ),
    )


def make_threshold_sweep_configs(
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    base: ETGConfig | None = None,
) -> list[AblationConfig]:
    """Ablation 3: ETG with varying tau thresholds.

    Characterizes the precision-recall trade-off:
    - Low tau: more claims pass -> higher recall, lower precision
    - High tau: fewer claims pass -> lower recall, higher precision
    """
    base = base or ETGConfig()
    configs = []
    for tau in thresholds:
        configs.append(AblationConfig(
            ablation_type=AblationType.THRESHOLD_SWEEP,
            name=f"ETG-tau={tau:.1f}",
            description=f"ETG with tau={tau}, characterizing precision-recall trade-off.",
            etg_config=ETGConfig(
                tau=tau,
                tau_prime=base.tau_prime,
                verification_budget=base.verification_budget,
                min_views_per_claim=base.min_views_per_claim,
                allow_uncertain=base.allow_uncertain,
                corpus_id=base.corpus_id,
            ),
        ))
    return configs


def make_policy_ablation_config(
    base: ETGConfig | None = None,
    seed: int | None = None,
) -> AblationConfig:
    """Ablation 4: Random claim selection instead of heuristic policy.

    Tests the importance of the utility-weighted recursive policy
    (Proposition 3). Random selection should lead to worse compute
    allocation and lower verification quality.
    """
    base = base or ETGConfig()
    return AblationConfig(
        ablation_type=AblationType.POLICY_ABLATION,
        name="ETG-RandomPolicy",
        description=(
            "ETG with random claim selection instead of the "
            "utility-weighted heuristic policy. Tests the "
            "importance of the recursive policy."
        ),
        etg_config=ETGConfig(
            tau=base.tau,
            tau_prime=base.tau_prime,
            verification_budget=base.verification_budget,
            min_views_per_claim=base.min_views_per_claim,
            allow_uncertain=base.allow_uncertain,
            corpus_id=base.corpus_id,
        ),
    )


def all_ablation_configs(
    base: ETGConfig | None = None,
) -> list[AblationConfig]:
    """Generate all ablation configurations.

    Returns a list of all ablation configs: NoMultiView, NoConstraint,
    5 threshold sweep configs, and PolicyAblation.
    """
    configs: list[AblationConfig] = [
        make_no_multi_view_config(base),
        make_no_constraint_config(base),
    ]
    configs.extend(make_threshold_sweep_configs(base=base))
    configs.append(make_policy_ablation_config(base))
    return configs
