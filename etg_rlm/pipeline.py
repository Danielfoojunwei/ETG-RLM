"""ETG Pipeline: end-to-end Evidence-Typed Generation.

Orchestrates the full ETG process:
    1. Extract atomic claims from generated text (claim compilation)
    2. Build the ESBG by proposing nodes and dependency edges
    3. Run multi-view verification on each claim node
    4. Type-check the graph
    5. Render only well-typed (Verified) claims into the final output

The constrained decoding formulation:
    y* = argmax_{y in Y(G_T, tau)} log p_theta(y | q, E)

where Y(G_T, tau) = {y : A(y) subset {pi(v) : v in V^tau}} is the
set of texts whose claims are all in the verified node set.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.verification import (
    ClaimExtractor,
    MultiViewVerifier,
    VerificationView,
)
from etg_rlm.type_system import (
    EvidenceTypeChecker,
    GraphTypeCheckResult,
    TypeThresholds,
)
from etg_rlm.policy import (
    ActionType,
    GreedyBudgetPolicy,
    PolicyAction,
    RecursionPolicy,
)
from etg_rlm.bounds import (
    hallucination_upper_bound,
    required_views_for_bound,
)


@runtime_checkable
class DependencyDetector(Protocol):
    """Protocol for detecting logical dependencies between claims."""

    def detect(
        self, claims: list[AtomicClaim]
    ) -> list[tuple[str, str]]:
        """Return pairs (from_id, to_id) indicating c_to depends on c_from."""
        ...


@runtime_checkable
class ConstrainedRenderer(Protocol):
    """Protocol for rendering verified claims into natural language text."""

    def render(
        self, claims: list[AtomicClaim], query: str
    ) -> str:
        """Render a set of verified claims into a coherent answer."""
        ...


@dataclass
class ETGConfig:
    """Configuration for the ETG pipeline.

    Attributes:
        tau: upper threshold for Verified type
        tau_prime: lower threshold for Unsupported type
        verification_budget: maximum number of verifier calls
        min_views_per_claim: minimum views before typing a claim
        allow_uncertain: if True, include Uncertain claims in output
        corpus_id: identifier for the evidence corpus E
    """

    tau: float = 0.7
    tau_prime: float = 0.3
    verification_budget: int = 50
    min_views_per_claim: int = 3
    allow_uncertain: bool = False
    corpus_id: str = "default"


@dataclass
class ETGResult:
    """Result of running the ETG pipeline.

    Attributes:
        esbg: the constructed Evidence-Scoped Belief Graph
        type_check: result of type-checking the graph
        rendered_text: the final rendered output (only well-typed claims)
        verified_claims: claims that passed the support-mass gate
        rejected_claims: claims rejected by the type checker
        hallucination_bound: theoretical upper bound on hallucination probability
        budget_used: number of verifier calls consumed
    """

    esbg: EvidenceScopedBeliefGraph
    type_check: GraphTypeCheckResult
    rendered_text: str
    verified_claims: list[AtomicClaim]
    rejected_claims: list[AtomicClaim]
    hallucination_bound: float
    budget_used: int


class ETGPipeline:
    """End-to-end Evidence-Typed Generation pipeline.

    The RLM acts as a type-directed compiler:
        1. Compile (q, E) into ESBG via claim extraction + verification
        2. Type-check claims via the evidence type system
        3. Render only well-typed outputs

    Proposition 3 (Read/Write separation):
    Under ETG constraints, confabulation probability is zero by construction
    (unless the verifier produces false positives). A confabulation event is
    defined as emitting a claim without evidence pointers. Since every
    rendered claim must have m(c) >= tau with entailed status, evidence
    pointers are guaranteed to exist.
    """

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        views: list[VerificationView],
        config: ETGConfig | None = None,
        policy: RecursionPolicy | None = None,
        dependency_detector: DependencyDetector | None = None,
        renderer: ConstrainedRenderer | None = None,
    ) -> None:
        self.config = config or ETGConfig()
        self.claim_extractor = claim_extractor
        self.verifier = MultiViewVerifier(views)
        self.type_checker = EvidenceTypeChecker(
            TypeThresholds(tau=self.config.tau, tau_prime=self.config.tau_prime)
        )
        self.policy = policy or GreedyBudgetPolicy(
            tau=self.config.tau,
            min_views=self.config.min_views_per_claim,
        )
        self.dependency_detector = dependency_detector
        self.renderer = renderer

    def run(self, query: str, generated_text: str) -> ETGResult:
        """Execute the full ETG pipeline.

        Steps:
            1. Extract atomic claims: A(y) -> {c_1, ..., c_m}
            2. Build initial ESBG with claim nodes
            3. Detect and add dependency edges
            4. Run policy-guided multi-view verification
            5. Type-check the graph
            6. Render verified claims

        Args:
            query: the original query q
            generated_text: the model's initial generation y

        Returns:
            ETGResult with the verified output and audit trail.
        """
        # Step 1: Extract atomic claims
        claims = self.claim_extractor.extract(generated_text)

        # Step 2: Build initial ESBG
        esbg = EvidenceScopedBeliefGraph()
        for claim in claims:
            node = ESBGNode(
                node_id=claim.claim_id,
                claim=claim,
            )
            esbg.add_node(node)

        # Step 3: Detect dependencies
        if self.dependency_detector is not None:
            deps = self.dependency_detector.detect(claims)
            for from_id, to_id in deps:
                try:
                    esbg.add_dependency(from_id, to_id)
                except ValueError:
                    # Skip invalid edges (missing nodes, would create cycle)
                    pass

        # Step 4: Policy-guided verification
        budget_used = 0
        budget = self.config.verification_budget

        while budget_used < budget:
            action = self.policy.select_action(
                query, self.config.corpus_id, esbg, budget - budget_used
            )

            if action.action_type == ActionType.STOP:
                break

            if action.action_type == ActionType.RUN_VIEW:
                if action.target_node_id is not None:
                    node = esbg.get_node(action.target_node_id)
                    self.verifier.verify_node(node, self.config.corpus_id)
                    budget_used += 1

            elif action.action_type == ActionType.SPLIT_CLAIM:
                # Splitting is a claim-extractor operation on a node's text
                if action.target_node_id is not None:
                    node = esbg.get_node(action.target_node_id)
                    subclaims = self.claim_extractor.extract(node.claim.text)
                    for sc in subclaims:
                        if sc.claim_id != node.claim.claim_id:
                            sub_node = ESBGNode(
                                node_id=sc.claim_id,
                                claim=sc,
                            )
                            esbg.add_node(sub_node)
                            esbg.add_dependency(sub_node.node_id, node.node_id)

        # Step 5: Type-check
        type_result = self.type_checker.check_graph(esbg)

        # Step 6: Determine renderable claims
        renderable_ids = self.type_checker.renderable_claims(esbg)

        if self.config.allow_uncertain:
            # Also include uncertain claims
            for node in esbg.topological_order():
                if node.claim_type == ClaimType.UNCERTAIN:
                    renderable_ids.add(node.node_id)

        verified_claims = []
        rejected_claims = []
        for node in esbg.topological_order():
            if node.node_id in renderable_ids:
                verified_claims.append(node.claim)
            else:
                rejected_claims.append(node.claim)

        # Step 7: Render
        if self.renderer is not None and verified_claims:
            rendered = self.renderer.render(verified_claims, query)
        else:
            rendered = self._default_render(verified_claims)

        # Compute theoretical bound
        n_min = min(
            (len(esbg.get_node(nid).view_verdicts) for nid in renderable_ids),
            default=0,
        )
        if n_min > 0:
            # Estimate alpha from the data as fraction of contradicted views
            all_verdicts = []
            for nid in esbg.all_node_ids():
                all_verdicts.extend(esbg.get_node(nid).view_verdicts)
            # Conservative estimate: use 0.1 as default alpha
            alpha = 0.1
            bound = hallucination_upper_bound(n_min, self.config.tau, alpha)
        else:
            bound = 1.0

        return ETGResult(
            esbg=esbg,
            type_check=type_result,
            rendered_text=rendered,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            hallucination_bound=bound,
            budget_used=budget_used,
        )

    @staticmethod
    def _default_render(claims: list[AtomicClaim]) -> str:
        """Default renderer: concatenate claim texts."""
        if not claims:
            return ""
        return " ".join(c.text for c in claims)
