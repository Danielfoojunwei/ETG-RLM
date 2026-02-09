"""Citation Precision and Recall metrics for evaluating attribution quality.

Measures whether generated citations correctly support claims (precision)
and whether all supported claims have citations (recall). This goes beyond
simple faithfulness by evaluating the *attribution chain* -- the connection
between claims and their cited evidence.

Definitions:
    Citation Precision: Of all citations provided, what fraction actually
        supports the associated claim?
        CP = |{(c, s) : s entails c}| / |{(c, s) : c cites s}|

    Citation Recall: Of all claims that have supporting evidence in E,
        what fraction have at least one correct citation?
        CR = |{c : exists s in cited(c) s.t. s entails c}| / |{c : c is entailed by E}|

These metrics capture the quality of the provenance trail that ETG maintains
by construction through the ESBG evidence pointers sigma(v).

References:
    [4] Gao et al., "ALCE: Attributed Language Model Evaluation," ACL 2023.
    [5] Rashkin et al., "Measuring Attribution in Natural Language Generation
        Models," ACL 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, EvidenceSpan


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """A citation linking a claim to an evidence span.

    Represents the assertion that evidence span `span` supports claim `claim`.
    In ETG, these are the sigma(v) evidence pointers in the ESBG.
    """

    claim: AtomicClaim
    span: EvidenceSpan


@runtime_checkable
class CitationVerifier(Protocol):
    """Verifies whether a cited evidence span actually supports a claim.

    Returns True if the span entails the claim, False otherwise.
    """

    def verify_citation(self, claim: AtomicClaim, span: EvidenceSpan) -> bool: ...


class CitationPairResult(NamedTuple):
    """Verification result for a single (claim, span) citation pair."""

    claim_id: str
    span_doc_id: str
    span_start: int
    span_end: int
    is_valid: bool


@dataclass
class CitationMetricsResult:
    """Citation Precision and Recall for one generation.

    Attributes:
        citation_precision: fraction of citations that are valid
        citation_recall: fraction of entailed claims with >= 1 valid citation
        n_total_citations: total (claim, span) citation pairs
        n_valid_citations: citations where span entails claim
        n_entailed_claims: claims that have evidence support
        n_cited_entailed: entailed claims with at least one valid citation
        per_citation: detailed per-citation verification results
    """

    citation_precision: float
    citation_recall: float
    n_total_citations: int
    n_valid_citations: int
    n_entailed_claims: int
    n_cited_entailed: int
    per_citation: list[CitationPairResult] = field(default_factory=list)


@dataclass
class BatchCitationResult:
    """Citation metrics aggregated across multiple instances."""

    mean_citation_precision: float
    mean_citation_recall: float
    n_instances: int
    total_citations: int
    total_valid: int
    per_instance: list[CitationMetricsResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_citation_metrics(
    citations: list[Citation],
    verifier: CitationVerifier,
    entailed_claim_ids: set[str] | None = None,
) -> CitationMetricsResult:
    """Compute citation precision and recall.

    Citation Precision:
        CP = |valid citations| / |total citations|
        Measures: of all citations made, how many are correct?

    Citation Recall:
        CR = |entailed claims with valid citation| / |entailed claims|
        Measures: of all claims that have evidence, how many cite it?

    Args:
        citations: list of (claim, span) citation pairs from the generation
        verifier: checks whether a span truly entails a claim
        entailed_claim_ids: set of claim IDs known to be entailed by E
            (if None, recall is computed from claims that have valid citations)

    Returns:
        CitationMetricsResult with precision and recall.
    """
    if not citations:
        return CitationMetricsResult(
            citation_precision=1.0,
            citation_recall=0.0 if entailed_claim_ids else 1.0,
            n_total_citations=0,
            n_valid_citations=0,
            n_entailed_claims=len(entailed_claim_ids) if entailed_claim_ids else 0,
            n_cited_entailed=0,
        )

    per_citation: list[CitationPairResult] = []
    n_valid = 0
    claims_with_valid_citation: set[str] = set()

    for cit in citations:
        is_valid = verifier.verify_citation(cit.claim, cit.span)
        if is_valid:
            n_valid += 1
            claims_with_valid_citation.add(cit.claim.claim_id)

        per_citation.append(CitationPairResult(
            claim_id=cit.claim.claim_id,
            span_doc_id=cit.span.doc_id,
            span_start=cit.span.start,
            span_end=cit.span.end,
            is_valid=is_valid,
        ))

    # Precision: valid / total citations
    citation_precision = n_valid / len(citations)

    # Recall: entailed claims with valid citation / entailed claims
    if entailed_claim_ids is not None:
        n_entailed = len(entailed_claim_ids)
        n_cited_entailed = len(claims_with_valid_citation & entailed_claim_ids)
        citation_recall = n_cited_entailed / n_entailed if n_entailed > 0 else 1.0
    else:
        # Without ground truth, use claims that have at least one citation
        all_cited = {cit.claim.claim_id for cit in citations}
        n_entailed = len(all_cited)
        n_cited_entailed = len(claims_with_valid_citation)
        citation_recall = n_cited_entailed / n_entailed if n_entailed > 0 else 1.0

    return CitationMetricsResult(
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        n_total_citations=len(citations),
        n_valid_citations=n_valid,
        n_entailed_claims=n_entailed,
        n_cited_entailed=n_cited_entailed,
        per_citation=per_citation,
    )


def aggregate_citation_metrics(
    results: list[CitationMetricsResult],
) -> BatchCitationResult:
    """Aggregate citation metrics across multiple instances.

    Args:
        results: per-instance citation metrics

    Returns:
        BatchCitationResult with means and totals.
    """
    if not results:
        return BatchCitationResult(
            mean_citation_precision=0.0,
            mean_citation_recall=0.0,
            n_instances=0,
            total_citations=0,
            total_valid=0,
        )

    precisions = [r.citation_precision for r in results]
    recalls = [r.citation_recall for r in results]

    return BatchCitationResult(
        mean_citation_precision=sum(precisions) / len(precisions),
        mean_citation_recall=sum(recalls) / len(recalls),
        n_instances=len(results),
        total_citations=sum(r.n_total_citations for r in results),
        total_valid=sum(r.n_valid_citations for r in results),
        per_instance=results,
    )


# ---------------------------------------------------------------------------
# ESBG-native citation extraction
# ---------------------------------------------------------------------------


def extract_citations_from_esbg(
    nodes: dict,
) -> list[Citation]:
    """Extract citation pairs from an ESBG's evidence pointers.

    In ETG, sigma(v) gives the evidence spans for each node v.
    Each (pi(v), s) for s in sigma(v) is a citation.

    Args:
        nodes: dict of node_id -> ESBGNode (from esbg.nodes)

    Returns:
        List of Citation objects representing the ESBG's provenance trail.
    """
    citations: list[Citation] = []
    for node in nodes.values():
        for span in node.evidence_spans:
            citations.append(Citation(claim=node.claim, span=span))
    return citations
