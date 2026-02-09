"""Core data structures for Evidence-Typed Generation.

Defines the Evidence-Scoped Belief Graph (ESBG) and its components:
- EvidenceSpan: a pointer into the source corpus E
- AtomicClaim: an extracted factual assertion
- ESBGNode: a node in the belief graph carrying a claim, evidence, and type
- EvidenceScopedBeliefGraph: the DAG G = (V, ->, pi, sigma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx


class ClaimStatus(Enum):
    """Entailment status z(v) of a claim node."""

    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ClaimType(Enum):
    """Evidence type assigned by the ETG type checker.

    type(c) =
        Verified      if m(c) >= tau
        Uncertain     if tau' < m(c) < tau
        Unsupported   if m(c) <= tau'
    """

    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvidenceSpan:
    """A span s in the source corpus E.

    Represents (doc_id, start, end) — a contiguous region of text
    in a specific document that supports (or contradicts) a claim.
    """

    doc_id: str
    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(
                f"Invalid span bounds: start={self.start}, end={self.end}"
            )


@dataclass
class AtomicClaim:
    """An atomic factual assertion extracted from a generated answer.

    Produced by the claim extractor A(y) -> {c_1, ..., c_m}.
    """

    claim_id: str
    text: str
    subclaims: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.claim_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicClaim):
            return NotImplemented
        return self.claim_id == other.claim_id


@dataclass
class ESBGNode:
    """A node v in the Evidence-Scoped Belief Graph.

    Carries:
      - pi(v) = claim: the atomic claim
      - sigma(v) = evidence_spans: set of supporting spans in E
      - m(v) = support_mass: multi-view support mass in [0, 1]
      - z(v) = status: entailment status
      - type(v) = claim_type: evidence type (assigned by type checker)
    """

    node_id: str
    claim: AtomicClaim
    evidence_spans: set[EvidenceSpan] = field(default_factory=set)
    support_mass: float = 0.0
    status: ClaimStatus = ClaimStatus.UNKNOWN
    claim_type: ClaimType = ClaimType.UNSUPPORTED
    view_verdicts: list[bool] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ESBGNode):
            return NotImplemented
        return self.node_id == other.node_id


class EvidenceScopedBeliefGraph:
    """Evidence-Scoped Belief Graph (ESBG).

    A directed acyclic graph G = (V, ->, pi, sigma) where:
      - V: set of ESBGNode (each carrying a claim)
      - ->: dependency edges (u -> v means c_v depends on c_u)
      - pi: node -> claim mapping (stored in ESBGNode)
      - sigma: node -> evidence spans mapping (stored in ESBGNode)

    Edges u -> v indicate logical, definitional, or compositional dependency:
    claim c_v requires c_u.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, ESBGNode] = {}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def nodes(self) -> dict[str, ESBGNode]:
        return dict(self._nodes)

    def add_node(self, node: ESBGNode) -> None:
        """Add a claim node to the ESBG."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id!r} already exists")
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge: claim to_id depends on claim from_id.

        Raises ValueError if adding the edge would create a cycle,
        since the ESBG must remain a DAG.
        """
        for nid in (from_id, to_id):
            if nid not in self._nodes:
                raise ValueError(f"Node {nid!r} not in graph")

        self._graph.add_edge(from_id, to_id)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_edge(from_id, to_id)
            raise ValueError(
                f"Adding edge {from_id!r} -> {to_id!r} would create a cycle"
            )

    def get_node(self, node_id: str) -> ESBGNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} not found")
        return self._nodes[node_id]

    def get_dependencies(self, node_id: str) -> list[ESBGNode]:
        """Return the nodes that node_id depends on (its predecessors)."""
        return [self._nodes[pid] for pid in self._graph.predecessors(node_id)]

    def get_dependents(self, node_id: str) -> list[ESBGNode]:
        """Return the nodes that depend on node_id (its successors)."""
        return [self._nodes[sid] for sid in self._graph.successors(node_id)]

    def topological_order(self) -> list[ESBGNode]:
        """Return nodes in topological order (dependencies before dependents)."""
        return [self._nodes[nid] for nid in nx.topological_sort(self._graph)]

    def verified_subgraph(self, tau: float) -> set[str]:
        """Return V^tau: nodes with support_mass >= tau and status ENTAILED.

        V^tau = {v in V : m(pi(v)) >= tau AND z(v) = entailed}
        """
        return {
            nid
            for nid, node in self._nodes.items()
            if node.support_mass >= tau and node.status == ClaimStatus.ENTAILED
        }

    def all_node_ids(self) -> set[str]:
        return set(self._nodes.keys())

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def summary(self) -> dict:
        """Return a summary of the graph state."""
        status_counts = {s: 0 for s in ClaimStatus}
        type_counts = {t: 0 for t in ClaimType}
        for node in self._nodes.values():
            status_counts[node.status] += 1
            type_counts[node.claim_type] += 1
        return {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "status_counts": {s.value: c for s, c in status_counts.items()},
            "type_counts": {t.value: c for t, c in type_counts.items()},
            "mean_support_mass": (
                sum(n.support_mass for n in self._nodes.values()) / len(self._nodes)
                if self._nodes
                else 0.0
            ),
        }
