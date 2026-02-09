"""Tests for the end-to-end ETG pipeline."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceSpan,
)
from etg_rlm.pipeline import ETGConfig, ETGPipeline, ETGResult
from etg_rlm.verification import VerificationView, ViewResult


# -- Stub implementations for testing --


class StubClaimExtractor:
    """Returns predetermined claims for testing."""

    def __init__(self, claims: list[AtomicClaim]):
        self._claims = claims

    def extract(self, text: str) -> list[AtomicClaim]:
        return self._claims


class StubDependencyDetector:
    """Returns predetermined dependencies."""

    def __init__(self, deps: list[tuple[str, str]]):
        self._deps = deps

    def detect(self, claims: list[AtomicClaim]) -> list[tuple[str, str]]:
        return self._deps


class StubRenderer:
    """Simple join renderer."""

    def render(self, claims: list[AtomicClaim], query: str) -> str:
        return " | ".join(c.text for c in claims)


class AlwaysEntailedView(VerificationView):
    """A view that always returns ENTAILED with a fixed span."""

    def __init__(self, view_id: str):
        super().__init__(view_id)

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        return ViewResult(
            verdict=ClaimStatus.ENTAILED,
            spans={EvidenceSpan(doc_id="d1", start=0, end=10, text="evidence")},
            confidence=0.95,
            view_id=self.view_id,
        )


class AlwaysUnknownView(VerificationView):
    """A view that always returns UNKNOWN."""

    def __init__(self, view_id: str):
        super().__init__(view_id)

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        return ViewResult(
            verdict=ClaimStatus.UNKNOWN,
            spans=set(),
            confidence=0.0,
            view_id=self.view_id,
        )


class MixedView(VerificationView):
    """A view that returns ENTAILED for specific claim IDs, UNKNOWN otherwise."""

    def __init__(self, view_id: str, entailed_ids: set[str]):
        super().__init__(view_id)
        self._entailed_ids = entailed_ids

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        if claim.claim_id in self._entailed_ids:
            return ViewResult(
                verdict=ClaimStatus.ENTAILED,
                spans={EvidenceSpan(doc_id="d1", start=0, end=5)},
                view_id=self.view_id,
            )
        return ViewResult(
            verdict=ClaimStatus.UNKNOWN,
            spans=set(),
            view_id=self.view_id,
        )


class TestETGPipeline:
    def test_all_claims_verified(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Fact A is true."),
            AtomicClaim(claim_id="c2", text="Fact B is true."),
        ]
        views = [AlwaysEntailedView(f"v{i}") for i in range(5)]
        extractor = StubClaimExtractor(claims)

        pipeline = ETGPipeline(
            claim_extractor=extractor,
            views=views,
            config=ETGConfig(tau=0.7, tau_prime=0.3, verification_budget=20),
        )
        result = pipeline.run("What are the facts?", "Fact A and B are true.")

        assert len(result.verified_claims) == 2
        assert len(result.rejected_claims) == 0
        assert result.type_check.well_typed is True
        assert "Fact A" in result.rendered_text
        assert "Fact B" in result.rendered_text

    def test_all_claims_rejected(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Hallucinated fact."),
        ]
        views = [AlwaysUnknownView(f"v{i}") for i in range(5)]
        extractor = StubClaimExtractor(claims)

        pipeline = ETGPipeline(
            claim_extractor=extractor,
            views=views,
            config=ETGConfig(tau=0.7, tau_prime=0.3, verification_budget=20),
        )
        result = pipeline.run("Tell me something", "Hallucinated fact.")

        assert len(result.verified_claims) == 0
        assert len(result.rejected_claims) == 1
        assert result.rendered_text == ""
        assert result.type_check.well_typed is False

    def test_mixed_claims(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Supported fact."),
            AtomicClaim(claim_id="c2", text="Unsupported claim."),
        ]
        # Views that only entail c1
        views = [MixedView(f"v{i}", {"c1"}) for i in range(5)]
        extractor = StubClaimExtractor(claims)

        pipeline = ETGPipeline(
            claim_extractor=extractor,
            views=views,
            config=ETGConfig(tau=0.7, tau_prime=0.3, verification_budget=20),
        )
        result = pipeline.run("query", "text")

        assert len(result.verified_claims) == 1
        assert result.verified_claims[0].claim_id == "c1"
        assert len(result.rejected_claims) == 1
        assert result.rejected_claims[0].claim_id == "c2"

    def test_with_dependencies(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Base fact."),
            AtomicClaim(claim_id="c2", text="Derived fact."),
        ]
        views = [AlwaysEntailedView(f"v{i}") for i in range(3)]
        extractor = StubClaimExtractor(claims)
        dep_detector = StubDependencyDetector([("c1", "c2")])

        pipeline = ETGPipeline(
            claim_extractor=extractor,
            views=views,
            config=ETGConfig(tau=0.7, verification_budget=20),
            dependency_detector=dep_detector,
        )
        result = pipeline.run("query", "text")

        assert result.esbg.num_edges() == 1
        deps = result.esbg.get_dependencies("c2")
        assert len(deps) == 1
        assert deps[0].node_id == "c1"

    def test_with_custom_renderer(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact")]
        views = [AlwaysEntailedView(f"v{i}") for i in range(3)]
        pipeline = ETGPipeline(
            claim_extractor=StubClaimExtractor(claims),
            views=views,
            config=ETGConfig(tau=0.7, verification_budget=10),
            renderer=StubRenderer(),
        )
        result = pipeline.run("q", "text")
        assert result.rendered_text == "Fact"

    def test_budget_respected(self):
        claims = [
            AtomicClaim(claim_id=f"c{i}", text=f"Claim {i}")
            for i in range(10)
        ]
        views = [AlwaysEntailedView("v1")]
        pipeline = ETGPipeline(
            claim_extractor=StubClaimExtractor(claims),
            views=views,
            config=ETGConfig(verification_budget=5),
        )
        result = pipeline.run("q", "text")
        assert result.budget_used <= 5

    def test_hallucination_bound_computed(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact")]
        views = [AlwaysEntailedView(f"v{i}") for i in range(5)]
        pipeline = ETGPipeline(
            claim_extractor=StubClaimExtractor(claims),
            views=views,
            config=ETGConfig(tau=0.7, verification_budget=20),
        )
        result = pipeline.run("q", "text")
        assert 0.0 <= result.hallucination_bound <= 1.0
