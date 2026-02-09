"""Tests for the Evidence-Typed Generation type system."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
    EvidenceSpan,
)
from etg_rlm.type_system import (
    EvidenceTypeChecker,
    TypeThresholds,
)


class TestTypeThresholds:
    def test_defaults(self):
        t = TypeThresholds()
        assert t.tau == 0.7
        assert t.tau_prime == 0.3

    def test_custom_valid(self):
        t = TypeThresholds(tau=0.8, tau_prime=0.2)
        assert t.tau == 0.8

    def test_invalid_tau_less_than_tau_prime(self):
        with pytest.raises(ValueError):
            TypeThresholds(tau=0.3, tau_prime=0.7)

    def test_invalid_equal(self):
        with pytest.raises(ValueError):
            TypeThresholds(tau=0.5, tau_prime=0.5)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            TypeThresholds(tau=0.5, tau_prime=-0.1)


class TestEvidenceTypeChecker:
    def _make_node(self, nid: str, mass: float, status: ClaimStatus = ClaimStatus.ENTAILED) -> ESBGNode:
        node = ESBGNode(
            node_id=nid,
            claim=AtomicClaim(claim_id=nid, text=f"Claim {nid}"),
            support_mass=mass,
            status=status,
        )
        return node

    def test_verified_type(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        node = self._make_node("a", 0.8)
        ct = checker.type_claim(node)
        assert ct == ClaimType.VERIFIED

    def test_uncertain_type(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        node = self._make_node("a", 0.5)
        ct = checker.type_claim(node)
        assert ct == ClaimType.UNCERTAIN

    def test_unsupported_type(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        node = self._make_node("a", 0.2)
        ct = checker.type_claim(node)
        assert ct == ClaimType.UNSUPPORTED

    def test_boundary_verified(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        node = self._make_node("a", 0.7)
        ct = checker.type_claim(node)
        assert ct == ClaimType.VERIFIED

    def test_boundary_unsupported(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        node = self._make_node("a", 0.3)
        ct = checker.type_claim(node)
        assert ct == ClaimType.UNSUPPORTED

    def test_check_node_updates_type(self):
        checker = EvidenceTypeChecker()
        node = self._make_node("a", 0.9)
        result = checker.check_node(node)
        assert node.claim_type == ClaimType.VERIFIED
        assert result.well_typed is True

    def test_check_graph_all_verified(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a", 0.8))
        g.add_node(self._make_node("b", 0.9))
        result = checker.check_graph(g)
        assert result.well_typed is True
        assert result.verified_count == 2
        assert result.unsupported_count == 0

    def test_check_graph_mixed(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a", 0.8))
        g.add_node(self._make_node("b", 0.1))
        result = checker.check_graph(g)
        assert result.well_typed is False
        assert result.verified_count == 1
        assert result.unsupported_count == 1

    def test_renderable_claims_with_deps(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        g = EvidenceScopedBeliefGraph()

        # a is verified, b depends on a and is verified, c depends on b and is not
        na = self._make_node("a", 0.9)
        nb = self._make_node("b", 0.8)
        nc = self._make_node("c", 0.1)

        g.add_node(na)
        g.add_node(nb)
        g.add_node(nc)
        g.add_dependency("a", "b")
        g.add_dependency("b", "c")

        renderable = checker.renderable_claims(g)
        assert "a" in renderable
        assert "b" in renderable
        assert "c" not in renderable  # Unsupported

    def test_renderable_excludes_broken_dependency_chain(self):
        checker = EvidenceTypeChecker(TypeThresholds(tau=0.7, tau_prime=0.3))
        g = EvidenceScopedBeliefGraph()

        # a is unsupported, b depends on a and is verified
        na = self._make_node("a", 0.1)
        nb = self._make_node("b", 0.9)

        g.add_node(na)
        g.add_node(nb)
        g.add_dependency("a", "b")

        renderable = checker.renderable_claims(g)
        assert "a" not in renderable
        assert "b" not in renderable  # Dependency not renderable
