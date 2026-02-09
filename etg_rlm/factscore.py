"""FactScore: Fine-grained Atomic Evaluation of Factual Precision.

Implements the FactScore methodology from Min et al. (2023):
    FactScore = (1/|A(y)|) * sum_{c in A(y)} 1[c is supported by E]

The pipeline:
    1. Atomic claim decomposition: A(y) -> {c_1, ..., c_m}
    2. Per-claim NLI verification against knowledge source E
    3. Aggregate into Claim-Precision (supported / total)
       and Claim-Recall (supported / reference claims)

This extends the basic faithfulness metrics with a principled,
reproducible scoring methodology aligned with the academic standard.

References:
    [8] Min et al., "FActScore: Fine-grained Atomic Evaluation of
        Factual Precision in Long Form Text Generation," EMNLP 2023.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan


# ---------------------------------------------------------------------------
# Protocols for pluggable components
# ---------------------------------------------------------------------------


@runtime_checkable
class AtomicFactExtractor(Protocol):
    """Extracts atomic facts from a generated response.

    Corresponds to the decomposition operator A(y) -> {c_1, ..., c_m}.
    In practice, this uses an LLM or rule-based system to split
    a paragraph into minimal, independently verifiable claims.
    """

    def extract_facts(self, text: str) -> list[AtomicClaim]: ...


@runtime_checkable
class NLIScorer(Protocol):
    """Scores whether a knowledge source entails an atomic claim.

    Returns a probability in [0, 1] that the evidence supports the claim.
    In practice, this wraps an NLI model (e.g., DeBERTa-v3-large)
    or an LLM prompted for entailment judgment.
    """

    def score(self, claim: AtomicClaim, evidence: list[EvidenceSpan]) -> float: ...


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


class ClaimScoreResult(NamedTuple):
    """Score for a single atomic claim."""

    claim_id: str
    claim_text: str
    score: float
    supported: bool
    evidence_used: list[str]  # doc_ids of evidence spans used


@dataclass
class FactScoreResult:
    """Aggregated FactScore result for one generation.

    Attributes:
        factscore: the FactScore = (supported claims) / (total claims)
        claim_precision: same as factscore (alias for clarity)
        claim_recall: (supported claims) / (reference claims), if available
        n_claims: total atomic claims extracted
        n_supported: claims scoring above threshold
        n_unsupported: claims scoring below threshold
        per_claim: detailed per-claim scores
    """

    factscore: float
    claim_precision: float
    claim_recall: float
    n_claims: int
    n_supported: int
    n_unsupported: int
    per_claim: list[ClaimScoreResult] = field(default_factory=list)

    @property
    def factual_density(self) -> float:
        """Average score across all claims (soft FactScore)."""
        if not self.per_claim:
            return 0.0
        return sum(c.score for c in self.per_claim) / len(self.per_claim)


@dataclass
class BatchFactScoreResult:
    """FactScore aggregated across multiple instances."""

    mean_factscore: float
    mean_claim_precision: float
    mean_claim_recall: float
    n_instances: int
    total_claims: int
    total_supported: int
    per_instance: list[FactScoreResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core FactScore computation
# ---------------------------------------------------------------------------


def compute_factscore(
    claims: list[AtomicClaim],
    evidence: list[EvidenceSpan],
    scorer: NLIScorer,
    threshold: float = 0.5,
    reference_claims: list[AtomicClaim] | None = None,
) -> FactScoreResult:
    """Compute FactScore for a set of atomic claims against evidence.

    FactScore = (1/|A(y)|) * sum_{c in A(y)} 1[score(c, E) >= threshold]

    Args:
        claims: atomic claims extracted from the generation A(y)
        evidence: knowledge source evidence spans E
        scorer: NLI scorer for (claim, evidence) -> [0, 1]
        threshold: decision boundary for supported vs. unsupported
        reference_claims: optional ground-truth claims for recall

    Returns:
        FactScoreResult with precision, recall, and per-claim details.
    """
    if not claims:
        return FactScoreResult(
            factscore=1.0,
            claim_precision=1.0,
            claim_recall=0.0 if reference_claims else float("nan"),
            n_claims=0,
            n_supported=0,
            n_unsupported=0,
        )

    per_claim: list[ClaimScoreResult] = []
    n_supported = 0

    for claim in claims:
        score = scorer.score(claim, evidence)
        supported = score >= threshold
        if supported:
            n_supported += 1

        evidence_used = list({s.doc_id for s in evidence})
        per_claim.append(ClaimScoreResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            score=score,
            supported=supported,
            evidence_used=evidence_used,
        ))

    factscore = n_supported / len(claims)

    # Recall: of reference claims, how many are covered by supported generated claims?
    if reference_claims is not None and len(reference_claims) > 0:
        ref_texts = {rc.text.lower().strip() for rc in reference_claims}
        supported_texts = {
            c.claim_text.lower().strip()
            for c in per_claim if c.supported
        }
        matched = len(ref_texts & supported_texts)
        claim_recall = matched / len(reference_claims)
    else:
        claim_recall = float("nan")

    return FactScoreResult(
        factscore=factscore,
        claim_precision=factscore,
        claim_recall=claim_recall,
        n_claims=len(claims),
        n_supported=n_supported,
        n_unsupported=len(claims) - n_supported,
        per_claim=per_claim,
    )


def aggregate_factscores(
    results: list[FactScoreResult],
) -> BatchFactScoreResult:
    """Aggregate FactScore results across multiple instances.

    Args:
        results: per-instance FactScore results

    Returns:
        BatchFactScoreResult with means and totals.
    """
    if not results:
        return BatchFactScoreResult(
            mean_factscore=0.0,
            mean_claim_precision=0.0,
            mean_claim_recall=0.0,
            n_instances=0,
            total_claims=0,
            total_supported=0,
        )

    import math

    scores = [r.factscore for r in results]
    precisions = [r.claim_precision for r in results]
    recalls = [r.claim_recall for r in results if not math.isnan(r.claim_recall)]

    return BatchFactScoreResult(
        mean_factscore=sum(scores) / len(scores),
        mean_claim_precision=sum(precisions) / len(precisions),
        mean_claim_recall=sum(recalls) / len(recalls) if recalls else float("nan"),
        n_instances=len(results),
        total_claims=sum(r.n_claims for r in results),
        total_supported=sum(r.n_supported for r in results),
        per_instance=results,
    )


# ---------------------------------------------------------------------------
# Decomposed FactScore (with retrieval per-claim)
# ---------------------------------------------------------------------------


@runtime_checkable
class PerClaimRetriever(Protocol):
    """Retrieves evidence specific to each atomic claim.

    Unlike bulk retrieval, this retrieves targeted evidence for each
    claim independently, matching the FactScore evaluation protocol.
    """

    def retrieve_for_claim(
        self, claim: AtomicClaim, corpus_id: str, top_k: int = 5
    ) -> list[EvidenceSpan]: ...


def compute_factscore_with_retrieval(
    claims: list[AtomicClaim],
    retriever: PerClaimRetriever,
    scorer: NLIScorer,
    corpus_id: str,
    top_k: int = 5,
    threshold: float = 0.5,
    reference_claims: list[AtomicClaim] | None = None,
) -> FactScoreResult:
    """Compute FactScore with per-claim retrieval (full pipeline).

    For each claim c_i:
        1. Retrieve top-k evidence spans for c_i
        2. Score entailment: score(c_i, retrieved_evidence)
        3. Aggregate into FactScore

    This matches the original FactScore protocol where retrieval
    is done per-claim rather than per-query.

    Args:
        claims: atomic claims A(y)
        retriever: per-claim evidence retriever
        scorer: NLI scorer
        corpus_id: evidence corpus identifier
        top_k: number of evidence spans per claim
        threshold: support decision boundary
        reference_claims: optional ground-truth claims for recall

    Returns:
        FactScoreResult with per-claim retrieval evidence.
    """
    if not claims:
        return FactScoreResult(
            factscore=1.0,
            claim_precision=1.0,
            claim_recall=0.0 if reference_claims else float("nan"),
            n_claims=0,
            n_supported=0,
            n_unsupported=0,
        )

    per_claim: list[ClaimScoreResult] = []
    n_supported = 0

    for claim in claims:
        evidence = retriever.retrieve_for_claim(claim, corpus_id, top_k)
        score = scorer.score(claim, evidence)
        supported = score >= threshold
        if supported:
            n_supported += 1

        per_claim.append(ClaimScoreResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            score=score,
            supported=supported,
            evidence_used=[s.doc_id for s in evidence],
        ))

    factscore = n_supported / len(claims)

    if reference_claims is not None and len(reference_claims) > 0:
        ref_texts = {rc.text.lower().strip() for rc in reference_claims}
        supported_texts = {
            c.claim_text.lower().strip()
            for c in per_claim if c.supported
        }
        matched = len(ref_texts & supported_texts)
        claim_recall = matched / len(reference_claims)
    else:
        claim_recall = float("nan")

    return FactScoreResult(
        factscore=factscore,
        claim_precision=factscore,
        claim_recall=claim_recall,
        n_claims=len(claims),
        n_supported=n_supported,
        n_unsupported=len(claims) - n_supported,
        per_claim=per_claim,
    )
