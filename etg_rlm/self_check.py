"""Self-CheckGPT: Zero-Resource Black-Box Hallucination Detection.

Implements the Self-CheckGPT methodology for hallucination detection
without requiring external knowledge sources. The key insight is that
hallucinated facts are inconsistent across multiple stochastic samples
from the same model, while grounded facts are stable.

Algorithm:
    1. Generate K stochastic samples {y_1, ..., y_K} for query q
    2. For each atomic claim c in the primary response y_0:
       a. Check whether c is consistent with each sample y_k
       b. Compute consistency score: score(c) = (1/K) sum 1[c consistent with y_k]
    3. Claims with low consistency scores are likely hallucinated

This serves as a baseline comparator for ETG. ETG's multi-view
verification uses independent retrieval + NLI, while Self-CheckGPT
uses only the model's own sampling distribution -- no external evidence.

The comparison validates ETG's thesis that *external evidence grounding*
provides stronger faithfulness guarantees than *behavioral self-consistency*.

References:
    [7] Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box
        Hallucination Detection for Generative Large Language Models,"
        EMNLP 2023.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StochasticGenerator(Protocol):
    """Generator that produces multiple stochastic samples."""

    def generate_samples(
        self, query: str, n_samples: int, temperature: float = 1.0
    ) -> list[str]: ...


@runtime_checkable
class ConsistencyChecker(Protocol):
    """Checks whether a claim is consistent with a sampled response.

    Returns a score in [0, 1] where 1 = fully consistent.
    """

    def check_consistency(self, claim: AtomicClaim, sample: str) -> float: ...


@runtime_checkable
class ClaimDecomposer(Protocol):
    """Decomposes text into atomic claims."""

    def decompose(self, text: str) -> list[AtomicClaim]: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SelfCheckMethod(Enum):
    """Consistency checking methods from Self-CheckGPT."""

    BERTSCORE = "bertscore"  # Soft semantic similarity
    NLI = "nli"  # NLI-based entailment
    PROMPTING = "prompting"  # LLM-based consistency prompting
    NGRAM = "ngram"  # N-gram overlap


@dataclass
class SelfCheckConfig:
    """Configuration for Self-CheckGPT.

    Attributes:
        n_samples: number of stochastic samples K
        temperature: sampling temperature for diversity
        method: consistency checking method
        threshold: decision boundary for hallucination
    """

    n_samples: int = 5
    temperature: float = 1.0
    method: SelfCheckMethod = SelfCheckMethod.NLI
    threshold: float = 0.5


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


class ClaimConsistencyResult(NamedTuple):
    """Consistency result for a single claim."""

    claim_id: str
    claim_text: str
    consistency_score: float
    is_hallucinated: bool
    per_sample_scores: list[float]


@dataclass
class SelfCheckResult:
    """Result of Self-CheckGPT on one generation.

    Attributes:
        hallucination_rate: fraction of claims detected as hallucinated
        mean_consistency: average consistency across all claims
        n_claims: total atomic claims checked
        n_hallucinated: claims below consistency threshold
        n_consistent: claims above consistency threshold
        n_samples_used: number of stochastic samples generated
        per_claim: detailed per-claim results
    """

    hallucination_rate: float
    mean_consistency: float
    n_claims: int
    n_hallucinated: int
    n_consistent: int
    n_samples_used: int
    per_claim: list[ClaimConsistencyResult] = field(default_factory=list)


@dataclass
class BatchSelfCheckResult:
    """Self-CheckGPT aggregated across multiple instances."""

    mean_hallucination_rate: float
    mean_consistency: float
    n_instances: int
    total_claims: int
    total_hallucinated: int
    per_instance: list[SelfCheckResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Self-CheckGPT pipeline
# ---------------------------------------------------------------------------


def self_check_claims(
    claims: list[AtomicClaim],
    samples: list[str],
    checker: ConsistencyChecker,
    threshold: float = 0.5,
) -> SelfCheckResult:
    """Run Self-CheckGPT consistency checking on a set of claims.

    For each claim c:
        score(c) = (1/K) * sum_{k=1}^{K} consistency(c, y_k)
        hallucinated = score(c) < threshold

    Args:
        claims: atomic claims from the primary generation
        samples: K stochastic sample responses
        checker: consistency checking function
        threshold: hallucination decision boundary

    Returns:
        SelfCheckResult with per-claim consistency scores.
    """
    if not claims:
        return SelfCheckResult(
            hallucination_rate=0.0,
            mean_consistency=1.0,
            n_claims=0,
            n_hallucinated=0,
            n_consistent=0,
            n_samples_used=len(samples),
        )

    per_claim: list[ClaimConsistencyResult] = []
    n_hallucinated = 0
    total_consistency = 0.0

    for claim in claims:
        per_sample_scores: list[float] = []
        for sample in samples:
            score = checker.check_consistency(claim, sample)
            per_sample_scores.append(score)

        consistency = (
            sum(per_sample_scores) / len(per_sample_scores)
            if per_sample_scores else 0.0
        )
        is_hallucinated = consistency < threshold
        if is_hallucinated:
            n_hallucinated += 1
        total_consistency += consistency

        per_claim.append(ClaimConsistencyResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            consistency_score=consistency,
            is_hallucinated=is_hallucinated,
            per_sample_scores=per_sample_scores,
        ))

    return SelfCheckResult(
        hallucination_rate=n_hallucinated / len(claims),
        mean_consistency=total_consistency / len(claims),
        n_claims=len(claims),
        n_hallucinated=n_hallucinated,
        n_consistent=len(claims) - n_hallucinated,
        n_samples_used=len(samples),
        per_claim=per_claim,
    )


def run_self_check_pipeline(
    query: str,
    primary_response: str,
    generator: StochasticGenerator,
    decomposer: ClaimDecomposer,
    checker: ConsistencyChecker,
    config: SelfCheckConfig | None = None,
) -> SelfCheckResult:
    """Full Self-CheckGPT pipeline: generate samples, decompose, check.

    Pipeline:
        1. Generate K stochastic samples from the model
        2. Decompose the primary response into atomic claims
        3. Check each claim's consistency against all samples
        4. Flag inconsistent claims as hallucinated

    Args:
        query: the input query
        primary_response: the primary generation y_0
        generator: stochastic sample generator
        decomposer: atomic claim extractor
        checker: consistency checking function
        config: Self-CheckGPT configuration

    Returns:
        SelfCheckResult with hallucination detection results.
    """
    cfg = config or SelfCheckConfig()

    # Step 1: Generate stochastic samples
    samples = generator.generate_samples(
        query, cfg.n_samples, cfg.temperature
    )

    # Step 2: Decompose primary response
    claims = decomposer.decompose(primary_response)

    # Step 3-4: Check consistency and flag hallucinations
    return self_check_claims(claims, samples, checker, cfg.threshold)


def aggregate_self_check_results(
    results: list[SelfCheckResult],
) -> BatchSelfCheckResult:
    """Aggregate Self-CheckGPT results across multiple instances.

    Args:
        results: per-instance Self-CheckGPT results

    Returns:
        BatchSelfCheckResult with means and totals.
    """
    if not results:
        return BatchSelfCheckResult(
            mean_hallucination_rate=0.0,
            mean_consistency=0.0,
            n_instances=0,
            total_claims=0,
            total_hallucinated=0,
        )

    hall_rates = [r.hallucination_rate for r in results]
    consistencies = [r.mean_consistency for r in results]

    return BatchSelfCheckResult(
        mean_hallucination_rate=sum(hall_rates) / len(hall_rates),
        mean_consistency=sum(consistencies) / len(consistencies),
        n_instances=len(results),
        total_claims=sum(r.n_claims for r in results),
        total_hallucinated=sum(r.n_hallucinated for r in results),
        per_instance=results,
    )
