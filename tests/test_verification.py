"""Tests for multi-view verification and support mass computation."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ESBGNode,
    EvidenceSpan,
)
from etg_rlm.verification import (
    MultiViewVerifier,
    VerificationView,
    ViewResult,
)


class StubView(VerificationView):
    """A stub view that returns a predetermined result."""

    def __init__(self, view_id: str, verdict: ClaimStatus, spans: set[EvidenceSpan] | None = None):
        super().__init__(view_id)
        self._verdict = verdict
        self._spans = spans or set()

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        return ViewResult(
            verdict=self._verdict,
            spans=self._spans,
            confidence=1.0 if self._verdict == ClaimStatus.ENTAILED else 0.0,
            view_id=self.view_id,
        )


class TestMultiViewVerifier:
    def _make_claim(self, cid: str = "c1") -> AtomicClaim:
        return AtomicClaim(claim_id=cid, text="Test claim")

    def test_all_entailed(self):
        span = EvidenceSpan(doc_id="d1", start=0, end=10, text="evidence")
        views = [
            StubView("v1", ClaimStatus.ENTAILED, {span}),
            StubView("v2", ClaimStatus.ENTAILED, {span}),
            StubView("v3", ClaimStatus.ENTAILED, {span}),
        ]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "corpus")

        assert mass == pytest.approx(1.0)
        assert status == ClaimStatus.ENTAILED
        assert len(spans) == 1
        assert all(verdicts)

    def test_none_entailed(self):
        views = [
            StubView("v1", ClaimStatus.UNKNOWN),
            StubView("v2", ClaimStatus.UNKNOWN),
            StubView("v3", ClaimStatus.UNKNOWN),
        ]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "corpus")

        assert mass == pytest.approx(0.0)
        assert status == ClaimStatus.UNKNOWN
        assert len(spans) == 0
        assert not any(verdicts)

    def test_mixed_verdicts(self):
        s1 = EvidenceSpan(doc_id="d1", start=0, end=5)
        s2 = EvidenceSpan(doc_id="d2", start=0, end=5)
        views = [
            StubView("v1", ClaimStatus.ENTAILED, {s1}),
            StubView("v2", ClaimStatus.UNKNOWN),
            StubView("v3", ClaimStatus.ENTAILED, {s2}),
            StubView("v4", ClaimStatus.CONTRADICTED),
            StubView("v5", ClaimStatus.UNKNOWN),
        ]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "corpus")

        assert mass == pytest.approx(2.0 / 5.0)
        assert len(spans) == 2
        assert s1 in spans and s2 in spans
        assert status == ClaimStatus.ENTAILED  # 2 entailed > 1 contradicted
        assert verdicts == [True, False, True, False, False]

    def test_majority_contradicted(self):
        views = [
            StubView("v1", ClaimStatus.CONTRADICTED),
            StubView("v2", ClaimStatus.CONTRADICTED),
            StubView("v3", ClaimStatus.ENTAILED, {EvidenceSpan("d", 0, 1)}),
        ]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "corpus")

        assert mass == pytest.approx(1.0 / 3.0)
        assert status == ClaimStatus.CONTRADICTED  # 2 contradicted > 1 entailed

    def test_verify_node_updates_fields(self):
        span = EvidenceSpan(doc_id="d1", start=0, end=10)
        views = [
            StubView("v1", ClaimStatus.ENTAILED, {span}),
            StubView("v2", ClaimStatus.ENTAILED, {span}),
        ]
        mv = MultiViewVerifier(views)
        claim = self._make_claim()
        node = ESBGNode(node_id="n1", claim=claim)
        mv.verify_node(node, "corpus")

        assert node.support_mass == pytest.approx(1.0)
        assert node.status == ClaimStatus.ENTAILED
        assert span in node.evidence_spans
        assert node.view_verdicts == [True, True]

    def test_empty_views_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiViewVerifier([])

    def test_single_view(self):
        views = [StubView("v1", ClaimStatus.ENTAILED, {EvidenceSpan("d", 0, 5)})]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "c")
        assert mass == pytest.approx(1.0)
        assert len(verdicts) == 1

    def test_evidence_aggregation_deduplicates(self):
        shared_span = EvidenceSpan(doc_id="d1", start=0, end=10)
        views = [
            StubView("v1", ClaimStatus.ENTAILED, {shared_span}),
            StubView("v2", ClaimStatus.ENTAILED, {shared_span}),
        ]
        mv = MultiViewVerifier(views)
        mass, spans, status, verdicts = mv.verify_claim(self._make_claim(), "c")
        # Same span from both views should be deduplicated
        assert len(spans) == 1
