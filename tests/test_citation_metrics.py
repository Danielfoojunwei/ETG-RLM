"""Tests for citation precision and recall metrics."""

import pytest

from etg_rlm.core import AtomicClaim, ClaimStatus, ClaimType, ESBGNode, EvidenceSpan
from etg_rlm.citation_metrics import (
    Citation,
    CitationMetricsResult,
    CitationPairResult,
    compute_citation_metrics,
    aggregate_citation_metrics,
    extract_citations_from_esbg,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class AlwaysValidVerifier:
    """Stub verifier: all citations are valid."""

    def verify_citation(self, claim: AtomicClaim, span: EvidenceSpan) -> bool:
        return True


class AlwaysInvalidVerifier:
    """Stub verifier: no citations are valid."""

    def verify_citation(self, claim: AtomicClaim, span: EvidenceSpan) -> bool:
        return False


class KeywordVerifier:
    """Stub verifier: valid if claim text appears in span text."""

    def verify_citation(self, claim: AtomicClaim, span: EvidenceSpan) -> bool:
        return claim.text.lower() in span.text.lower()


# ---------------------------------------------------------------------------
# Test compute_citation_metrics
# ---------------------------------------------------------------------------


class TestCitationMetrics:
    def test_empty_citations(self):
        result = compute_citation_metrics([], AlwaysValidVerifier())
        assert result.citation_precision == 1.0
        assert result.n_total_citations == 0

    def test_all_valid_citations(self):
        claim = AtomicClaim(claim_id="c1", text="A fact")
        span = EvidenceSpan(doc_id="d1", start=0, end=50, text="evidence")
        citations = [Citation(claim=claim, span=span)]
        result = compute_citation_metrics(citations, AlwaysValidVerifier())
        assert result.citation_precision == 1.0
        assert result.n_valid_citations == 1

    def test_no_valid_citations(self):
        claim = AtomicClaim(claim_id="c1", text="A fact")
        span = EvidenceSpan(doc_id="d1", start=0, end=50, text="evidence")
        citations = [Citation(claim=claim, span=span)]
        result = compute_citation_metrics(citations, AlwaysInvalidVerifier())
        assert result.citation_precision == 0.0
        assert result.n_valid_citations == 0

    def test_mixed_citations(self):
        c1 = AtomicClaim(claim_id="c1", text="Paris is in France")
        c2 = AtomicClaim(claim_id="c2", text="Tokyo is in Brazil")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50, text="Paris is in France")
        s2 = EvidenceSpan(doc_id="d2", start=0, end=50, text="Tokyo is in Japan")
        citations = [Citation(claim=c1, span=s1), Citation(claim=c2, span=s2)]
        result = compute_citation_metrics(citations, KeywordVerifier())
        assert result.citation_precision == 0.5
        assert result.n_valid_citations == 1

    def test_citation_recall_with_entailed_ids(self):
        c1 = AtomicClaim(claim_id="c1", text="true")
        c2 = AtomicClaim(claim_id="c2", text="true")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        citations = [Citation(claim=c1, span=s1)]
        # c1 and c2 are entailed, but only c1 has a citation
        result = compute_citation_metrics(
            citations, AlwaysValidVerifier(), entailed_claim_ids={"c1", "c2"}
        )
        assert result.citation_recall == 0.5  # 1 out of 2 entailed claims cited
        assert result.n_entailed_claims == 2
        assert result.n_cited_entailed == 1

    def test_citation_recall_all_cited(self):
        c1 = AtomicClaim(claim_id="c1", text="fact")
        c2 = AtomicClaim(claim_id="c2", text="fact")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        s2 = EvidenceSpan(doc_id="d2", start=0, end=50)
        citations = [
            Citation(claim=c1, span=s1),
            Citation(claim=c2, span=s2),
        ]
        result = compute_citation_metrics(
            citations, AlwaysValidVerifier(), entailed_claim_ids={"c1", "c2"}
        )
        assert result.citation_recall == 1.0

    def test_per_citation_details(self):
        c1 = AtomicClaim(claim_id="c1", text="fact")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        s2 = EvidenceSpan(doc_id="d2", start=0, end=50)
        citations = [Citation(claim=c1, span=s1), Citation(claim=c1, span=s2)]
        result = compute_citation_metrics(citations, AlwaysValidVerifier())
        assert len(result.per_citation) == 2
        assert all(c.is_valid for c in result.per_citation)

    def test_recall_without_entailed_ids(self):
        c1 = AtomicClaim(claim_id="c1", text="fact")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        citations = [Citation(claim=c1, span=s1)]
        result = compute_citation_metrics(citations, AlwaysValidVerifier())
        # Without entailed_claim_ids, recall is computed from cited claims
        assert result.citation_recall == 1.0

    def test_empty_entailed_claims(self):
        c1 = AtomicClaim(claim_id="c1", text="fact")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        citations = [Citation(claim=c1, span=s1)]
        result = compute_citation_metrics(
            citations, AlwaysValidVerifier(), entailed_claim_ids=set()
        )
        assert result.citation_recall == 1.0  # No entailed claims to miss


# ---------------------------------------------------------------------------
# Test aggregate_citation_metrics
# ---------------------------------------------------------------------------


class TestAggregateCitationMetrics:
    def test_empty(self):
        result = aggregate_citation_metrics([])
        assert result.n_instances == 0

    def test_multiple_instances(self):
        results = [
            CitationMetricsResult(
                citation_precision=1.0, citation_recall=0.8,
                n_total_citations=5, n_valid_citations=5,
                n_entailed_claims=5, n_cited_entailed=4,
            ),
            CitationMetricsResult(
                citation_precision=0.6, citation_recall=1.0,
                n_total_citations=5, n_valid_citations=3,
                n_entailed_claims=3, n_cited_entailed=3,
            ),
        ]
        agg = aggregate_citation_metrics(results)
        assert agg.mean_citation_precision == pytest.approx(0.8)
        assert agg.mean_citation_recall == pytest.approx(0.9)
        assert agg.total_citations == 10
        assert agg.total_valid == 8


# ---------------------------------------------------------------------------
# Test extract_citations_from_esbg
# ---------------------------------------------------------------------------


class TestExtractCitationsFromESBG:
    def test_extracts_citation_pairs(self):
        claim = AtomicClaim(claim_id="c1", text="A fact")
        s1 = EvidenceSpan(doc_id="d1", start=0, end=50)
        s2 = EvidenceSpan(doc_id="d2", start=10, end=60)
        node = ESBGNode(
            node_id="n1", claim=claim,
            evidence_spans={s1, s2},
            support_mass=0.8,
            status=ClaimStatus.ENTAILED,
        )
        citations = extract_citations_from_esbg({"n1": node})
        assert len(citations) == 2
        assert all(c.claim.claim_id == "c1" for c in citations)

    def test_empty_nodes(self):
        citations = extract_citations_from_esbg({})
        assert len(citations) == 0

    def test_node_without_evidence(self):
        claim = AtomicClaim(claim_id="c1", text="No evidence")
        node = ESBGNode(node_id="n1", claim=claim)
        citations = extract_citations_from_esbg({"n1": node})
        assert len(citations) == 0
