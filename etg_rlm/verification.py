"""Multi-view evidence verification (Section 4.3).

Implements Definitions 2 and 3 from the paper.

Definition 2 (Verification View):
    A verification view V_i is a function:
        V_i : (E, c) -> (z_i, S_i)
    where z_i in {entailed, contradicted, not-found} and S_i subset S(E).
    Views differ by query rewriting, chunk boundaries, retriever randomness,
    verifier prompting, and negative sampling windows.

Definition 3 (Support Mass):
    Given N independent views:
        m(c) = (1/N) * sum_{i=1}^{N} 1[z_i = entailed]
    Evidence pointers aggregate as:
        sigma(c) = union_{i : z_i = entailed} S_i

Motivation: Single-view verification is brittle. Hallucinated claims often
pass under one retrieval or chunking but fail under others. Support mass
is a stability invariant across multiple independent evidence views.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ESBGNode,
    EvidenceSpan,
)


@dataclass
class ViewResult:
    """Result from a single verification view V_i(E, c).

    Attributes:
        verdict: entailment verdict z_i
        spans: supporting span set S_i found under this view
        confidence: optional confidence score from the verifier
        view_id: identifier of the view that produced this result
    """

    verdict: ClaimStatus
    spans: set[EvidenceSpan] = field(default_factory=set)
    confidence: float = 0.0
    view_id: str = ""


@runtime_checkable
class ClaimExtractor(Protocol):
    """Protocol for the atomic-claim extractor A(y) -> {c_1, ..., c_m}."""

    def extract(self, text: str) -> list[AtomicClaim]:
        """Extract atomic claims from generated text."""
        ...


@runtime_checkable
class EvidenceRetriever(Protocol):
    """Protocol for retrieving candidate evidence spans from corpus E."""

    def retrieve(self, query: str, corpus_id: str) -> list[EvidenceSpan]:
        """Retrieve candidate evidence spans for a query."""
        ...


@runtime_checkable
class EntailmentVerifier(Protocol):
    """Protocol for checking if evidence spans entail a claim."""

    def verify(
        self, claim: AtomicClaim, spans: list[EvidenceSpan]
    ) -> ViewResult:
        """Check entailment of claim given candidate spans."""
        ...


class VerificationView(ABC):
    """A single verification view V_i (Definition 2).

    A verification view is a function:
        V_i : (E, c) -> (z_i, S_i)
    where z_i in {entailed, contradicted, not-found} and S_i subset S(E).

    Views differ by:
        - query rewriting
        - chunk boundaries
        - retriever randomness
        - verifier prompting
        - negative sampling windows
    """

    def __init__(self, view_id: str) -> None:
        self.view_id = view_id

    @abstractmethod
    def verify(
        self,
        claim: AtomicClaim,
        corpus_id: str,
    ) -> ViewResult:
        """Run this view's retrieval + verification pipeline on a claim.

        Returns a ViewResult with the entailment verdict and supporting spans.
        """
        ...


class ComposableView(VerificationView):
    """A view composed from a retriever and a verifier.

    This is the standard construction: retrieve candidate spans,
    then verify entailment.
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.query_rewriter = query_rewriter

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        query = claim.text
        if self.query_rewriter is not None:
            query = self.query_rewriter.rewrite(query)

        spans = self.retriever.retrieve(query, corpus_id)
        result = self.verifier.verify(claim, spans)
        result.view_id = self.view_id
        return result


@runtime_checkable
class QueryRewriter(Protocol):
    """Protocol for query rewriting strategies."""

    def rewrite(self, query: str) -> str:
        ...


class MultiViewVerifier:
    """Runs N independent verification views and computes support mass (Definition 3).

    Given N views {V_1, ..., V_N}, for each claim c:
        1. Run each view: V_i(E, c) -> (z_i, S_i)
        2. Compute support mass: m(c) = (1/N) * sum_{i=1}^{N} 1[z_i = entailed]
        3. Aggregate evidence: sigma(c) = union_{i: z_i=entailed} S_i
        4. Determine overall status based on majority verdict

    Hallucinations are typically unstable under multi-view scrutiny;
    supported claims persist. This becomes a formal, measurable quantity
    treated as an "evidence invariant" (Section 4.3).
    """

    def __init__(self, views: list[VerificationView]) -> None:
        if not views:
            raise ValueError("At least one verification view is required")
        self.views = views
        self.n_views = len(views)

    def verify_claim(
        self, claim: AtomicClaim, corpus_id: str
    ) -> tuple[float, set[EvidenceSpan], ClaimStatus, list[bool]]:
        """Verify a claim across all views.

        Returns:
            support_mass: m(c) in [0, 1]
            evidence_spans: aggregated sigma(c)
            status: overall entailment status
            verdicts: per-view boolean verdicts (True = entailed)
        """
        results: list[ViewResult] = []
        for view in self.views:
            result = view.verify(claim, corpus_id)
            results.append(result)

        return self._aggregate(results)

    def verify_node(self, node: ESBGNode, corpus_id: str) -> ESBGNode:
        """Verify a node's claim and update its fields in place."""
        mass, spans, status, verdicts = self.verify_claim(node.claim, corpus_id)
        node.support_mass = mass
        node.evidence_spans = spans
        node.status = status
        node.view_verdicts = verdicts
        return node

    def _aggregate(
        self, results: list[ViewResult]
    ) -> tuple[float, set[EvidenceSpan], ClaimStatus, list[bool]]:
        """Aggregate results from N views.

        support mass:  m(c) = (1/N) * sum 1[z_i = entailed]
        evidence set:  sigma(c) = union_{i: z_i=entailed} S_i
        """
        entailed_count = 0
        contradicted_count = 0
        all_spans: set[EvidenceSpan] = set()
        verdicts: list[bool] = []

        for r in results:
            is_entailed = r.verdict == ClaimStatus.ENTAILED
            verdicts.append(is_entailed)
            if is_entailed:
                entailed_count += 1
                all_spans |= r.spans
            elif r.verdict == ClaimStatus.CONTRADICTED:
                contradicted_count += 1

        support_mass = entailed_count / self.n_views

        # Determine overall status by majority
        if contradicted_count > entailed_count:
            status = ClaimStatus.CONTRADICTED
        elif entailed_count > 0 and entailed_count >= contradicted_count:
            status = ClaimStatus.ENTAILED
        else:
            status = ClaimStatus.UNKNOWN

        return support_mass, all_spans, status, verdicts
