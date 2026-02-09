"""Tests for the FactScore evaluation metrics."""

import math
import pytest

from etg_rlm.core import AtomicClaim, EvidenceSpan
from etg_rlm.factscore import (
    ClaimScoreResult,
    FactScoreResult,
    BatchFactScoreResult,
    compute_factscore,
    aggregate_factscores,
    compute_factscore_with_retrieval,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubNLIScorer:
    """Stub scorer: returns 1.0 for claims containing 'true', 0.0 otherwise."""

    def score(self, claim: AtomicClaim, evidence: list[EvidenceSpan]) -> float:
        if "true" in claim.text.lower():
            return 0.95
        return 0.1


class PerfectNLIScorer:
    """Always returns 1.0."""

    def score(self, claim: AtomicClaim, evidence: list[EvidenceSpan]) -> float:
        return 1.0


class FailingNLIScorer:
    """Always returns 0.0."""

    def score(self, claim: AtomicClaim, evidence: list[EvidenceSpan]) -> float:
        return 0.0


class StubPerClaimRetriever:
    """Returns a fixed evidence span for any claim."""

    def retrieve_for_claim(
        self, claim: AtomicClaim, corpus_id: str, top_k: int = 5
    ) -> list[EvidenceSpan]:
        return [EvidenceSpan(doc_id="doc1", start=0, end=100, text="Evidence text")]


# ---------------------------------------------------------------------------
# Test compute_factscore
# ---------------------------------------------------------------------------


class TestComputeFactScore:
    def test_empty_claims(self):
        result = compute_factscore([], [], StubNLIScorer())
        assert result.factscore == 1.0
        assert result.n_claims == 0

    def test_all_supported(self):
        claims = [
            AtomicClaim(claim_id="c1", text="This is true fact one"),
            AtomicClaim(claim_id="c2", text="This is true fact two"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert result.factscore == 1.0
        assert result.n_supported == 2
        assert result.n_unsupported == 0

    def test_none_supported(self):
        claims = [
            AtomicClaim(claim_id="c1", text="A hallucinated claim"),
            AtomicClaim(claim_id="c2", text="Another hallucination"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert result.factscore == 0.0
        assert result.n_supported == 0
        assert result.n_unsupported == 2

    def test_mixed_support(self):
        claims = [
            AtomicClaim(claim_id="c1", text="This is true"),
            AtomicClaim(claim_id="c2", text="This is hallucinated"),
            AtomicClaim(claim_id="c3", text="Also true claim"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert result.factscore == pytest.approx(2 / 3)
        assert result.n_supported == 2
        assert result.n_unsupported == 1

    def test_precision_equals_factscore(self):
        claims = [
            AtomicClaim(claim_id="c1", text="True fact"),
            AtomicClaim(claim_id="c2", text="False claim"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert result.claim_precision == result.factscore

    def test_recall_with_reference_claims(self):
        claims = [
            AtomicClaim(claim_id="c1", text="This is true"),
            AtomicClaim(claim_id="c2", text="Also true"),
        ]
        ref_claims = [
            AtomicClaim(claim_id="r1", text="This is true"),
            AtomicClaim(claim_id="r2", text="Missing fact"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(
            claims, evidence, StubNLIScorer(), reference_claims=ref_claims
        )
        # "this is true" matches ref "this is true", "also true" doesn't match "missing fact"
        assert result.claim_recall == 0.5

    def test_recall_without_reference(self):
        claims = [AtomicClaim(claim_id="c1", text="True fact")]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert math.isnan(result.claim_recall)

    def test_per_claim_details(self):
        claims = [
            AtomicClaim(claim_id="c1", text="True claim"),
            AtomicClaim(claim_id="c2", text="Bad claim"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        assert len(result.per_claim) == 2
        assert result.per_claim[0].supported is True
        assert result.per_claim[1].supported is False

    def test_custom_threshold(self):
        claims = [AtomicClaim(claim_id="c1", text="True claim")]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        # StubNLIScorer returns 0.95 for "true" claims
        result_high = compute_factscore(claims, evidence, StubNLIScorer(), threshold=0.99)
        assert result_high.n_supported == 0  # 0.95 < 0.99
        result_low = compute_factscore(claims, evidence, StubNLIScorer(), threshold=0.5)
        assert result_low.n_supported == 1  # 0.95 >= 0.5

    def test_factual_density(self):
        claims = [
            AtomicClaim(claim_id="c1", text="True fact"),
            AtomicClaim(claim_id="c2", text="False claim"),
        ]
        evidence = [EvidenceSpan(doc_id="d1", start=0, end=50)]
        result = compute_factscore(claims, evidence, StubNLIScorer())
        # True returns 0.95, False returns 0.1 → density = (0.95 + 0.1) / 2
        assert result.factual_density == pytest.approx(0.525)


# ---------------------------------------------------------------------------
# Test aggregate_factscores
# ---------------------------------------------------------------------------


class TestAggregateFactScores:
    def test_empty(self):
        result = aggregate_factscores([])
        assert result.n_instances == 0
        assert result.mean_factscore == 0.0

    def test_single_instance(self):
        fs = FactScoreResult(
            factscore=0.8, claim_precision=0.8, claim_recall=0.6,
            n_claims=5, n_supported=4, n_unsupported=1,
        )
        result = aggregate_factscores([fs])
        assert result.mean_factscore == 0.8
        assert result.mean_claim_recall == 0.6
        assert result.n_instances == 1

    def test_multiple_instances(self):
        results = [
            FactScoreResult(factscore=1.0, claim_precision=1.0, claim_recall=0.8,
                          n_claims=3, n_supported=3, n_unsupported=0),
            FactScoreResult(factscore=0.5, claim_precision=0.5, claim_recall=0.4,
                          n_claims=4, n_supported=2, n_unsupported=2),
        ]
        agg = aggregate_factscores(results)
        assert agg.mean_factscore == pytest.approx(0.75)
        assert agg.total_claims == 7
        assert agg.total_supported == 5

    def test_nan_recall_excluded(self):
        results = [
            FactScoreResult(factscore=0.9, claim_precision=0.9, claim_recall=float("nan"),
                          n_claims=2, n_supported=2, n_unsupported=0),
            FactScoreResult(factscore=0.7, claim_precision=0.7, claim_recall=0.5,
                          n_claims=3, n_supported=2, n_unsupported=1),
        ]
        agg = aggregate_factscores(results)
        assert agg.mean_claim_recall == 0.5  # Only the non-NaN value


# ---------------------------------------------------------------------------
# Test compute_factscore_with_retrieval
# ---------------------------------------------------------------------------


class TestFactScoreWithRetrieval:
    def test_per_claim_retrieval(self):
        claims = [
            AtomicClaim(claim_id="c1", text="True fact one"),
            AtomicClaim(claim_id="c2", text="True fact two"),
        ]
        result = compute_factscore_with_retrieval(
            claims, StubPerClaimRetriever(), PerfectNLIScorer(), "corpus1"
        )
        assert result.factscore == 1.0
        assert result.n_supported == 2

    def test_empty_claims(self):
        result = compute_factscore_with_retrieval(
            [], StubPerClaimRetriever(), PerfectNLIScorer(), "corpus1"
        )
        assert result.factscore == 1.0
        assert result.n_claims == 0

    def test_evidence_doc_ids_recorded(self):
        claims = [AtomicClaim(claim_id="c1", text="A fact")]
        result = compute_factscore_with_retrieval(
            claims, StubPerClaimRetriever(), PerfectNLIScorer(), "corpus1"
        )
        assert "doc1" in result.per_claim[0].evidence_used
