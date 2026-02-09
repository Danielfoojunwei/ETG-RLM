"""Logic-Step Verification for multi-hop reasoning chains.

Evaluates whether each step in a multi-hop reasoning chain is logically
valid and grounded in evidence. This is critical for datasets like
HotpotQA where answers require composing information across multiple
source documents.

The verification proceeds step-by-step:
    1. Decompose the reasoning chain into individual inference steps
    2. Verify each step is entailed by its premises + evidence
    3. Check that the chain is logically coherent (no contradictions)
    4. Compute step-level and chain-level accuracy

This captures a dimension that FactScore alone misses: whether the
*composition* of individually correct facts is itself valid.

Example:
    Step 1: "The Eiffel Tower is in Paris" (entailed by doc A)
    Step 2: "Paris is the capital of France" (entailed by doc B)
    Step 3: "The Eiffel Tower is in the capital of France" (valid composition)

References:
    [3] Yang et al., "HotpotQA: A Dataset for Diverse, Explainable
        Multi-hop Question Answering," EMNLP 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class StepValidity(Enum):
    """Validity status of a single reasoning step."""

    VALID = "valid"  # Step follows from premises + evidence
    INVALID = "invalid"  # Step does not follow
    UNSUPPORTED = "unsupported"  # Step has no evidence basis
    REDUNDANT = "redundant"  # Step is a tautology or repetition


@dataclass(frozen=True)
class ReasoningStep:
    """A single step in a multi-hop reasoning chain.

    Attributes:
        step_id: unique identifier for this step
        claim: the atomic claim asserted at this step
        premises: IDs of prior steps this step depends on
        evidence_spans: evidence spans supporting this step
    """

    step_id: str
    claim: AtomicClaim
    premises: tuple[str, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = ()


@dataclass
class StepVerificationResult:
    """Verification result for a single reasoning step."""

    step_id: str
    validity: StepValidity
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class ChainVerificationResult:
    """Verification result for an entire reasoning chain.

    Attributes:
        chain_valid: True if all steps are valid
        step_accuracy: fraction of steps that are valid
        n_steps: total number of reasoning steps
        n_valid: number of valid steps
        n_invalid: number of invalid steps
        n_unsupported: number of unsupported steps
        n_redundant: number of redundant steps
        chain_coherent: True if no contradictions between steps
        per_step: detailed per-step results
    """

    chain_valid: bool
    step_accuracy: float
    n_steps: int
    n_valid: int
    n_invalid: int
    n_unsupported: int
    n_redundant: int
    chain_coherent: bool
    per_step: list[StepVerificationResult] = field(default_factory=list)


@dataclass
class BatchChainResult:
    """Logic verification aggregated across multiple chains."""

    mean_step_accuracy: float
    chain_validity_rate: float
    mean_chain_coherence: float
    n_chains: int
    total_steps: int
    total_valid_steps: int
    per_chain: list[ChainVerificationResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StepVerifier(Protocol):
    """Verifies whether a reasoning step follows from its premises and evidence."""

    def verify_step(
        self,
        step: ReasoningStep,
        premise_claims: list[AtomicClaim],
    ) -> StepVerificationResult: ...


@runtime_checkable
class ChainDecomposer(Protocol):
    """Decomposes a multi-hop answer into a chain of reasoning steps."""

    def decompose(self, text: str) -> list[ReasoningStep]: ...


@runtime_checkable
class CoherenceChecker(Protocol):
    """Checks whether a set of claims is internally consistent (no contradictions)."""

    def check_coherence(self, claims: list[AtomicClaim]) -> bool: ...


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------


def verify_reasoning_chain(
    steps: list[ReasoningStep],
    verifier: StepVerifier,
    coherence_checker: CoherenceChecker | None = None,
) -> ChainVerificationResult:
    """Verify each step in a multi-hop reasoning chain.

    For each step s_i with premises {s_j : j in deps(i)}:
        1. Gather premise claims from prior steps
        2. Verify step entailment: premises + evidence |= s_i
        3. Record validity

    Then check overall chain coherence (optional).

    Args:
        steps: ordered list of reasoning steps
        verifier: step-level entailment verifier
        coherence_checker: optional coherence checker for contradiction detection

    Returns:
        ChainVerificationResult with step-level and chain-level metrics.
    """
    if not steps:
        return ChainVerificationResult(
            chain_valid=True,
            step_accuracy=1.0,
            n_steps=0,
            n_valid=0,
            n_invalid=0,
            n_unsupported=0,
            n_redundant=0,
            chain_coherent=True,
        )

    # Build step lookup for premise resolution
    step_map: dict[str, ReasoningStep] = {s.step_id: s for s in steps}

    per_step: list[StepVerificationResult] = []
    n_valid = 0
    n_invalid = 0
    n_unsupported = 0
    n_redundant = 0

    for step in steps:
        # Resolve premise claims
        premise_claims = [
            step_map[pid].claim
            for pid in step.premises
            if pid in step_map
        ]

        result = verifier.verify_step(step, premise_claims)
        per_step.append(result)

        if result.validity == StepValidity.VALID:
            n_valid += 1
        elif result.validity == StepValidity.INVALID:
            n_invalid += 1
        elif result.validity == StepValidity.UNSUPPORTED:
            n_unsupported += 1
        elif result.validity == StepValidity.REDUNDANT:
            n_redundant += 1

    step_accuracy = n_valid / len(steps)
    chain_valid = n_invalid == 0 and n_unsupported == 0

    # Coherence check
    if coherence_checker is not None:
        all_claims = [s.claim for s in steps]
        chain_coherent = coherence_checker.check_coherence(all_claims)
    else:
        # Without a checker, assume coherent if no invalid steps
        chain_coherent = n_invalid == 0

    return ChainVerificationResult(
        chain_valid=chain_valid,
        step_accuracy=step_accuracy,
        n_steps=len(steps),
        n_valid=n_valid,
        n_invalid=n_invalid,
        n_unsupported=n_unsupported,
        n_redundant=n_redundant,
        chain_coherent=chain_coherent,
        per_step=per_step,
    )


def aggregate_chain_results(
    results: list[ChainVerificationResult],
) -> BatchChainResult:
    """Aggregate chain verification results across multiple instances.

    Args:
        results: per-chain verification results

    Returns:
        BatchChainResult with means and totals.
    """
    if not results:
        return BatchChainResult(
            mean_step_accuracy=0.0,
            chain_validity_rate=0.0,
            mean_chain_coherence=0.0,
            n_chains=0,
            total_steps=0,
            total_valid_steps=0,
        )

    step_accuracies = [r.step_accuracy for r in results]
    chain_valid_count = sum(1 for r in results if r.chain_valid)
    coherence_count = sum(1 for r in results if r.chain_coherent)

    return BatchChainResult(
        mean_step_accuracy=sum(step_accuracies) / len(step_accuracies),
        chain_validity_rate=chain_valid_count / len(results),
        mean_chain_coherence=coherence_count / len(results),
        n_chains=len(results),
        total_steps=sum(r.n_steps for r in results),
        total_valid_steps=sum(r.n_valid for r in results),
        per_chain=results,
    )


# ---------------------------------------------------------------------------
# ESBG-native chain extraction
# ---------------------------------------------------------------------------


def extract_reasoning_chain_from_esbg(
    nodes: dict,
    edges: list[tuple[str, str]],
) -> list[ReasoningStep]:
    """Extract a reasoning chain from an ESBG's DAG structure.

    The ESBG's dependency edges u -> v naturally encode a reasoning chain:
    each node v with predecessors {u_1, ..., u_k} is a step that depends
    on premises {pi(u_1), ..., pi(u_k)}.

    Args:
        nodes: dict of node_id -> ESBGNode
        edges: list of (from_id, to_id) dependency edges

    Returns:
        List of ReasoningSteps in topological order.
    """
    # Build adjacency for premise lookup
    predecessors: dict[str, list[str]] = {nid: [] for nid in nodes}
    for from_id, to_id in edges:
        if to_id in predecessors:
            predecessors[to_id].append(from_id)

    # Topological sort (simple Kahn's algorithm)
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}
    for _, to_id in edges:
        if to_id in in_degree:
            in_degree[to_id] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    topo_order: list[str] = []

    while queue:
        queue.sort()  # deterministic ordering
        nid = queue.pop(0)
        topo_order.append(nid)
        for _, to_id in edges:
            if _ == nid and to_id in in_degree:
                in_degree[to_id] -= 1
                if in_degree[to_id] == 0:
                    queue.append(to_id)

    steps: list[ReasoningStep] = []
    for nid in topo_order:
        if nid not in nodes:
            continue
        node = nodes[nid]
        steps.append(ReasoningStep(
            step_id=nid,
            claim=node.claim,
            premises=tuple(predecessors.get(nid, [])),
            evidence_spans=tuple(node.evidence_spans),
        ))

    return steps
