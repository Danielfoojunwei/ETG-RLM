"""Tests for core data structures: EvidenceSpan, AtomicClaim, ESBGNode, ESBG."""

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceSpan,
    EvidenceScopedBeliefGraph,
)


class TestEvidenceSpan:
    def test_creation(self):
        span = EvidenceSpan(doc_id="doc1", start=0, end=10, text="hello")
        assert span.doc_id == "doc1"
        assert span.start == 0
        assert span.end == 10
        assert span.text == "hello"

    def test_frozen(self):
        span = EvidenceSpan(doc_id="doc1", start=0, end=10)
        with pytest.raises(AttributeError):
            span.start = 5  # type: ignore[misc]

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="Invalid span bounds"):
            EvidenceSpan(doc_id="d", start=-1, end=5)
        with pytest.raises(ValueError, match="Invalid span bounds"):
            EvidenceSpan(doc_id="d", start=10, end=5)

    def test_hashable(self):
        s1 = EvidenceSpan(doc_id="d1", start=0, end=5)
        s2 = EvidenceSpan(doc_id="d1", start=0, end=5)
        assert s1 == s2
        assert hash(s1) == hash(s2)
        assert len({s1, s2}) == 1

    def test_zero_length_span(self):
        span = EvidenceSpan(doc_id="d", start=5, end=5)
        assert span.start == span.end


class TestAtomicClaim:
    def test_creation(self):
        c = AtomicClaim(claim_id="c1", text="The sky is blue.")
        assert c.claim_id == "c1"
        assert c.text == "The sky is blue."
        assert c.subclaims == []

    def test_equality_by_id(self):
        c1 = AtomicClaim(claim_id="c1", text="A")
        c2 = AtomicClaim(claim_id="c1", text="B")
        assert c1 == c2  # Same ID

    def test_hashable(self):
        c1 = AtomicClaim(claim_id="c1", text="A")
        c2 = AtomicClaim(claim_id="c1", text="B")
        assert hash(c1) == hash(c2)


class TestESBGNode:
    def test_creation_defaults(self):
        claim = AtomicClaim(claim_id="c1", text="test")
        node = ESBGNode(node_id="n1", claim=claim)
        assert node.support_mass == 0.0
        assert node.status == ClaimStatus.UNKNOWN
        assert node.claim_type == ClaimType.UNSUPPORTED
        assert node.evidence_spans == set()
        assert node.view_verdicts == []

    def test_equality_by_id(self):
        c1 = AtomicClaim(claim_id="c1", text="test")
        n1 = ESBGNode(node_id="n1", claim=c1)
        n2 = ESBGNode(node_id="n1", claim=c1, support_mass=0.5)
        assert n1 == n2


class TestEvidenceScopedBeliefGraph:
    def _make_node(self, nid: str) -> ESBGNode:
        return ESBGNode(
            node_id=nid,
            claim=AtomicClaim(claim_id=nid, text=f"Claim {nid}"),
        )

    def test_add_node(self):
        g = EvidenceScopedBeliefGraph()
        n = self._make_node("a")
        g.add_node(n)
        assert g.num_nodes() == 1
        assert g.get_node("a") is n

    def test_duplicate_node_raises(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        with pytest.raises(ValueError, match="already exists"):
            g.add_node(self._make_node("a"))

    def test_add_dependency(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        g.add_node(self._make_node("b"))
        g.add_dependency("a", "b")  # b depends on a
        assert g.num_edges() == 1
        deps = g.get_dependencies("b")
        assert len(deps) == 1
        assert deps[0].node_id == "a"

    def test_cycle_detection(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        g.add_node(self._make_node("b"))
        g.add_dependency("a", "b")
        with pytest.raises(ValueError, match="cycle"):
            g.add_dependency("b", "a")
        # Edge should not have been added
        assert g.num_edges() == 1

    def test_missing_node_dependency(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        with pytest.raises(ValueError, match="not in graph"):
            g.add_dependency("a", "z")

    def test_topological_order(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        g.add_node(self._make_node("b"))
        g.add_node(self._make_node("c"))
        g.add_dependency("a", "b")
        g.add_dependency("b", "c")
        order = g.topological_order()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_verified_subgraph(self):
        g = EvidenceScopedBeliefGraph()
        n1 = self._make_node("a")
        n1.support_mass = 0.8
        n1.status = ClaimStatus.ENTAILED
        n2 = self._make_node("b")
        n2.support_mass = 0.3
        n2.status = ClaimStatus.UNKNOWN
        g.add_node(n1)
        g.add_node(n2)
        verified = g.verified_subgraph(tau=0.7)
        assert verified == {"a"}

    def test_get_dependents(self):
        g = EvidenceScopedBeliefGraph()
        g.add_node(self._make_node("a"))
        g.add_node(self._make_node("b"))
        g.add_dependency("a", "b")
        dependents = g.get_dependents("a")
        assert len(dependents) == 1
        assert dependents[0].node_id == "b"

    def test_summary(self):
        g = EvidenceScopedBeliefGraph()
        n1 = self._make_node("a")
        n1.status = ClaimStatus.ENTAILED
        n1.claim_type = ClaimType.VERIFIED
        n1.support_mass = 0.9
        g.add_node(n1)
        n2 = self._make_node("b")
        n2.status = ClaimStatus.UNKNOWN
        n2.claim_type = ClaimType.UNSUPPORTED
        n2.support_mass = 0.1
        g.add_node(n2)

        s = g.summary()
        assert s["num_nodes"] == 2
        assert s["status_counts"]["entailed"] == 1
        assert s["type_counts"]["verified"] == 1
        assert s["mean_support_mass"] == pytest.approx(0.5)

    def test_get_node_not_found(self):
        g = EvidenceScopedBeliefGraph()
        with pytest.raises(KeyError):
            g.get_node("nonexistent")
