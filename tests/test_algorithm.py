"""Tests for the EBRG algorithm (Section 5) and constrained decoding (Section 4.6)."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceSpan,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.type_system import EvidenceTypeChecker, TypeThresholds
from etg_rlm.verification import VerificationView, ViewResult
from etg_rlm.algorithm import (
    constrained_decode,
    ebrg,
    ConstrainedDecodingResult,
    EBRGResult,
)


class AlwaysEntailedView(VerificationView):
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
    def __init__(self, view_id: str):
        super().__init__(view_id)

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        return ViewResult(verdict=ClaimStatus.UNKNOWN, spans=set(), view_id=self.view_id)


class SelectiveView(VerificationView):
    """Entails only specific claim IDs."""

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
        return ViewResult(verdict=ClaimStatus.UNKNOWN, spans=set(), view_id=self.view_id)


class TestConstrainedDecode:
    def test_all_verified(self):
        g = EvidenceScopedBeliefGraph()
        for i in range(3):
            n = ESBGNode(
                node_id=f"c{i}",
                claim=AtomicClaim(claim_id=f"c{i}", text=f"Fact {i}"),
                support_mass=0.9,
                status=ClaimStatus.ENTAILED,
            )
            g.add_node(n)

        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        result = constrained_decode(g, checker)

        assert len(result.verified_claims) == 3
        assert len(result.rejected_claims) == 0
        assert "Fact 0" in result.rendered_text

    def test_mixed_verified_and_rejected(self):
        g = EvidenceScopedBeliefGraph()
        n1 = ESBGNode(
            node_id="c1",
            claim=AtomicClaim(claim_id="c1", text="Supported"),
            support_mass=0.9,
            status=ClaimStatus.ENTAILED,
        )
        n2 = ESBGNode(
            node_id="c2",
            claim=AtomicClaim(claim_id="c2", text="Hallucinated"),
            support_mass=0.1,
            status=ClaimStatus.UNKNOWN,
        )
        g.add_node(n1)
        g.add_node(n2)

        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        result = constrained_decode(g, checker)

        assert len(result.verified_claims) == 1
        assert result.verified_claims[0].text == "Supported"
        assert len(result.rejected_claims) == 1
        assert "Hallucinated" not in result.rendered_text

    def test_empty_graph(self):
        g = EvidenceScopedBeliefGraph()
        checker = EvidenceTypeChecker()
        result = constrained_decode(g, checker)
        assert result.verified_claims == []
        assert result.rendered_text == ""

    def test_dependency_chain_respected(self):
        """If a dependency is not verified, dependents are also excluded."""
        g = EvidenceScopedBeliefGraph()
        # a is unsupported, b depends on a
        na = ESBGNode(
            node_id="a",
            claim=AtomicClaim(claim_id="a", text="Base"),
            support_mass=0.1,
            status=ClaimStatus.UNKNOWN,
        )
        nb = ESBGNode(
            node_id="b",
            claim=AtomicClaim(claim_id="b", text="Derived"),
            support_mass=0.9,
            status=ClaimStatus.ENTAILED,
        )
        g.add_node(na)
        g.add_node(nb)
        g.add_dependency("a", "b")

        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        result = constrained_decode(g, checker)

        # b is rejected because its dependency a is not verified
        assert "a" not in result.verified_node_ids
        assert "b" not in result.verified_node_ids


class TestEBRG:
    def test_all_verified_claims(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Earth orbits the Sun."),
            AtomicClaim(claim_id="c2", text="Water is H2O."),
        ]
        views = [AlwaysEntailedView(f"v{i}") for i in range(5)]

        result = ebrg(
            query="Science facts",
            claims=claims,
            views=views,
            tau=0.7,
            budget=20,
        )

        assert isinstance(result, EBRGResult)
        assert len(result.decoding.verified_claims) == 2
        assert len(result.decoding.rejected_claims) == 0
        assert result.type_check.well_typed is True
        assert result.zero_confabulation_holds is True
        assert result.hallucination_bound < 1.0

    def test_all_rejected_claims(self):
        claims = [AtomicClaim(claim_id="c1", text="Hallucinated.")]
        views = [AlwaysUnknownView(f"v{i}") for i in range(5)]

        result = ebrg(query="q", claims=claims, views=views, tau=0.7, budget=20)

        assert len(result.decoding.verified_claims) == 0
        assert len(result.decoding.rejected_claims) == 1
        assert result.decoding.rendered_text == ""

    def test_mixed_claims(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Supported fact."),
            AtomicClaim(claim_id="c2", text="Unsupported claim."),
        ]
        views = [SelectiveView(f"v{i}", {"c1"}) for i in range(5)]

        result = ebrg(query="q", claims=claims, views=views, tau=0.7, budget=20)

        assert len(result.decoding.verified_claims) == 1
        assert result.decoding.verified_claims[0].claim_id == "c1"
        assert len(result.decoding.rejected_claims) == 1

    def test_budget_respected(self):
        claims = [
            AtomicClaim(claim_id=f"c{i}", text=f"Claim {i}")
            for i in range(10)
        ]
        views = [AlwaysEntailedView("v1")]

        result = ebrg(query="q", claims=claims, views=views, budget=5)

        assert result.budget_used <= 5

    def test_step_log_populated(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact.")]
        views = [AlwaysEntailedView(f"v{i}") for i in range(3)]

        result = ebrg(query="q", claims=claims, views=views, tau=0.7, budget=10)

        assert len(result.step_log) == 3  # 3 views for 1 claim
        for log in result.step_log:
            assert log.node_id == "c1"
            assert log.verdict == ClaimStatus.ENTAILED

    def test_with_dependencies(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Base fact."),
            AtomicClaim(claim_id="c2", text="Derived from c1."),
        ]
        views = [AlwaysEntailedView(f"v{i}") for i in range(3)]

        result = ebrg(
            query="q",
            claims=claims,
            views=views,
            dependencies=[("c1", "c2")],
            budget=20,
        )

        assert result.esbg.num_edges() == 1
        assert len(result.decoding.verified_claims) == 2

    def test_n_views_per_claim(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact.")]
        views = [AlwaysEntailedView(f"v{i}") for i in range(10)]

        result = ebrg(
            query="q", claims=claims, views=views,
            n_views_per_claim=3, budget=100,
        )

        # Only 3 views should have been run
        assert result.budget_used == 3
        assert len(result.step_log) == 3

    def test_proposition2_zero_confabulation(self):
        """Verify Proposition 2 holds: no claim without evidence is rendered."""
        claims = [
            AtomicClaim(claim_id="c1", text="Supported."),
            AtomicClaim(claim_id="c2", text="No evidence."),
        ]
        views = [SelectiveView(f"v{i}", {"c1"}) for i in range(5)]

        result = ebrg(query="q", claims=claims, views=views, tau=0.7, budget=20)

        # c2 has no evidence -> should not be in verified set
        # Proposition 2 should hold because only c1 (with evidence) is verified
        assert result.zero_confabulation_holds is True
        for vc in result.decoding.verified_claims:
            node = result.esbg.get_node(vc.claim_id)
            assert len(node.evidence_spans) > 0
