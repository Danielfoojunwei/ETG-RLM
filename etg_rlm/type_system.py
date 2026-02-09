"""Evidence-Typed Generation (ETG) Type System.

Treats each claim c as having a "type" given by its support mass:

    type(c) =
        Verified      if m(c) >= tau
        Uncertain     if tau' < m(c) < tau
        Unsupported   if m(c) <= tau'

The type-checker rejects an answer if it contains Unsupported claims.
The RLM acts as a type-directed compiler:
    1. Compile (q, E) into ESBG
    2. Type-check claims
    3. Render only well-typed outputs

Hallucination control becomes static checking rather than prompting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from etg_rlm.core import (
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)


@dataclass
class TypeThresholds:
    """Threshold parameters for the ETG type system.

    tau: upper threshold — claims with m(c) >= tau are Verified
    tau_prime: lower threshold — claims with m(c) <= tau' are Unsupported
    Claims with tau' < m(c) < tau are Uncertain.
    """

    tau: float = 0.7
    tau_prime: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_prime < self.tau <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= tau' < tau <= 1, "
                f"got tau'={self.tau_prime}, tau={self.tau}"
            )


class TypeCheckResult(NamedTuple):
    """Result of type-checking a single claim node."""

    node_id: str
    claim_type: ClaimType
    support_mass: float
    well_typed: bool


class GraphTypeCheckResult(NamedTuple):
    """Result of type-checking the entire ESBG."""

    well_typed: bool
    node_results: list[TypeCheckResult]
    verified_count: int
    uncertain_count: int
    unsupported_count: int


class EvidenceTypeChecker:
    """Type checker for the Evidence-Typed Generation framework.

    Assigns evidence types to claims and validates that an answer
    contains only well-typed (Verified) claims.
    """

    def __init__(self, thresholds: TypeThresholds | None = None) -> None:
        self.thresholds = thresholds or TypeThresholds()

    def type_claim(self, node: ESBGNode) -> ClaimType:
        """Assign an evidence type to a claim based on its support mass.

        type(c) =
            Verified      if m(c) >= tau
            Uncertain     if tau' < m(c) < tau
            Unsupported   if m(c) <= tau'
        """
        m = node.support_mass
        if m >= self.thresholds.tau:
            return ClaimType.VERIFIED
        elif m > self.thresholds.tau_prime:
            return ClaimType.UNCERTAIN
        else:
            return ClaimType.UNSUPPORTED

    def check_node(self, node: ESBGNode) -> TypeCheckResult:
        """Type-check a single node and update its claim_type field."""
        claim_type = self.type_claim(node)
        node.claim_type = claim_type
        well_typed = claim_type == ClaimType.VERIFIED
        return TypeCheckResult(
            node_id=node.node_id,
            claim_type=claim_type,
            support_mass=node.support_mass,
            well_typed=well_typed,
        )

    def check_graph(self, esbg: EvidenceScopedBeliefGraph) -> GraphTypeCheckResult:
        """Type-check all nodes in an ESBG.

        The graph is well-typed iff every node is Verified.
        An answer is renderable only from well-typed graphs.
        """
        results: list[TypeCheckResult] = []
        verified = 0
        uncertain = 0
        unsupported = 0

        for node in esbg.topological_order():
            result = self.check_node(node)
            results.append(result)
            if result.claim_type == ClaimType.VERIFIED:
                verified += 1
            elif result.claim_type == ClaimType.UNCERTAIN:
                uncertain += 1
            else:
                unsupported += 1

        all_well_typed = unsupported == 0 and uncertain == 0
        return GraphTypeCheckResult(
            well_typed=all_well_typed,
            node_results=results,
            verified_count=verified,
            uncertain_count=uncertain,
            unsupported_count=unsupported,
        )

    def renderable_claims(
        self, esbg: EvidenceScopedBeliefGraph
    ) -> set[str]:
        """Return the set of node IDs whose claims can be rendered.

        V^tau = {v in V : m(pi(v)) >= tau AND z(v) = entailed}

        This defines the allowed output space Y(G_T, tau): the set of
        texts whose claims are all in V^tau.
        """
        renderable: set[str] = set()
        for node in esbg.topological_order():
            ct = self.type_claim(node)
            if ct == ClaimType.VERIFIED and node.status == ClaimStatus.ENTAILED:
                # Also check that all dependencies are renderable
                deps = esbg.get_dependencies(node.node_id)
                if all(d.node_id in renderable for d in deps):
                    renderable.add(node.node_id)
        return renderable
