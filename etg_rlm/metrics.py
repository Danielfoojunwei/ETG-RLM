"""Evaluation metrics for Evidence-Typed Generation (Section 3 of eval plan).

Primary metrics (Faithfulness):
    - Hallucination Rate: % of claims not supported by source documents
    - Claim Precision: TP / (TP + FP) -- fraction of generated claims that are supported
    - Claim Recall: TP / (TP + FN) -- fraction of possible true claims that were included

Secondary metrics (Quality & Efficiency):
    - ROUGE-L: n-gram overlap with reference answers
    - Latency: wall-clock time per token
    - Compute Cost: total verifier calls and FLOPs estimate
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from etg_rlm.core import AtomicClaim, ClaimStatus, ClaimType, EvidenceScopedBeliefGraph


# ---------------------------------------------------------------------------
# Faithfulness metrics (Section 3.1)
# ---------------------------------------------------------------------------


class FaithfulnessMetrics(NamedTuple):
    """Primary faithfulness metrics for a single evaluation instance."""

    hallucination_rate: float
    claim_precision: float
    claim_recall: float
    n_supported: int
    n_contradicted: int
    n_unsupported: int
    n_reference_claims: int


def compute_faithfulness(
    esbg: EvidenceScopedBeliefGraph,
    reference_claim_ids: set[str] | None = None,
) -> FaithfulnessMetrics:
    """Compute faithfulness metrics from a verified ESBG.

    Definitions:
        - Supported (TP): claim with type = Verified
        - Unsupported (FP): claim with type = Unsupported or Uncertain
        - Contradicted: claim with status = CONTRADICTED
        - Reference claims (for recall): the set of true claims extractable from E

    Args:
        esbg: the type-checked Evidence-Scoped Belief Graph
        reference_claim_ids: optional set of claim IDs representing ground-truth
            claims derivable from the source documents (for recall computation)

    Returns:
        FaithfulnessMetrics with hallucination rate, precision, and recall.
    """
    nodes = esbg.nodes
    n_total = len(nodes)
    if n_total == 0:
        return FaithfulnessMetrics(
            hallucination_rate=0.0,
            claim_precision=1.0,
            claim_recall=0.0,
            n_supported=0,
            n_contradicted=0,
            n_unsupported=0,
            n_reference_claims=len(reference_claim_ids) if reference_claim_ids else 0,
        )

    n_supported = 0
    n_contradicted = 0
    n_unsupported = 0

    for node in nodes.values():
        if node.claim_type == ClaimType.VERIFIED:
            n_supported += 1
        elif node.status == ClaimStatus.CONTRADICTED:
            n_contradicted += 1
        else:
            n_unsupported += 1

    hallucination_rate = (n_unsupported + n_contradicted) / n_total
    claim_precision = n_supported / n_total

    # Recall: of ground-truth claims, how many did we include and verify?
    if reference_claim_ids:
        verified_ids = {
            nid for nid, node in nodes.items()
            if node.claim_type == ClaimType.VERIFIED
        }
        true_positives = len(verified_ids & reference_claim_ids)
        claim_recall = true_positives / len(reference_claim_ids)
    else:
        claim_recall = float("nan")

    return FaithfulnessMetrics(
        hallucination_rate=hallucination_rate,
        claim_precision=claim_precision,
        claim_recall=claim_recall,
        n_supported=n_supported,
        n_contradicted=n_contradicted,
        n_unsupported=n_unsupported,
        n_reference_claims=len(reference_claim_ids) if reference_claim_ids else 0,
    )


# ---------------------------------------------------------------------------
# ROUGE-L (Section 3.2)
# ---------------------------------------------------------------------------


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of the longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


class ROUGELScore(NamedTuple):
    """ROUGE-L precision, recall, and F1."""

    precision: float
    recall: float
    f1: float


def rouge_l(hypothesis: str, reference: str) -> ROUGELScore:
    """Compute ROUGE-L score between hypothesis and reference texts.

    ROUGE-L uses the longest common subsequence (LCS) to measure
    n-gram overlap quality for summaries and long-form answers.

    Args:
        hypothesis: the generated answer text
        reference: the reference answer text

    Returns:
        ROUGELScore with precision, recall, and F1.
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return ROUGELScore(precision=0.0, recall=0.0, f1=0.0)

    lcs = _lcs_length(hyp_tokens, ref_tokens)

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return ROUGELScore(precision=precision, recall=recall, f1=f1)


# ---------------------------------------------------------------------------
# Latency & compute tracking (Section 3.2)
# ---------------------------------------------------------------------------


@dataclass
class LatencyMetrics:
    """Latency and compute cost metrics for a single run."""

    total_time_seconds: float
    n_tokens_generated: int
    ms_per_token: float
    n_verifier_calls: int
    n_retriever_calls: int

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_seconds == 0:
            return float("inf")
        return self.n_tokens_generated / self.total_time_seconds


class LatencyTracker:
    """Context manager for tracking generation latency."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0
        self._verifier_calls: int = 0
        self._retriever_calls: int = 0

    def __enter__(self) -> LatencyTracker:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.monotonic()

    def record_verifier_call(self) -> None:
        self._verifier_calls += 1

    def record_retriever_call(self) -> None:
        self._retriever_calls += 1

    def compute(self, n_tokens: int) -> LatencyMetrics:
        elapsed = self._end - self._start
        ms_per_token = (elapsed * 1000 / n_tokens) if n_tokens > 0 else 0.0
        return LatencyMetrics(
            total_time_seconds=elapsed,
            n_tokens_generated=n_tokens,
            ms_per_token=ms_per_token,
            n_verifier_calls=self._verifier_calls,
            n_retriever_calls=self._retriever_calls,
        )


# ---------------------------------------------------------------------------
# Aggregate metrics across a benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics across a full benchmark run."""

    n_instances: int
    mean_hallucination_rate: float
    mean_claim_precision: float
    mean_claim_recall: float
    mean_rouge_l_f1: float
    mean_ms_per_token: float
    total_verifier_calls: int
    per_instance: list[dict] = field(default_factory=list)


def aggregate_metrics(
    faithfulness_results: list[FaithfulnessMetrics],
    rouge_results: list[ROUGELScore],
    latency_results: list[LatencyMetrics],
) -> BenchmarkMetrics:
    """Aggregate per-instance metrics into benchmark-level statistics.

    Args:
        faithfulness_results: per-instance faithfulness metrics
        rouge_results: per-instance ROUGE-L scores
        latency_results: per-instance latency measurements

    Returns:
        BenchmarkMetrics with means and totals.
    """
    n = len(faithfulness_results)
    if n == 0:
        return BenchmarkMetrics(
            n_instances=0,
            mean_hallucination_rate=0.0,
            mean_claim_precision=0.0,
            mean_claim_recall=0.0,
            mean_rouge_l_f1=0.0,
            mean_ms_per_token=0.0,
            total_verifier_calls=0,
        )

    import math

    hall_rates = [f.hallucination_rate for f in faithfulness_results]
    precisions = [f.claim_precision for f in faithfulness_results]
    recalls = [f.claim_recall for f in faithfulness_results if not math.isnan(f.claim_recall)]
    rouge_f1s = [r.f1 for r in rouge_results] if rouge_results else []
    ms_per_tokens = [l.ms_per_token for l in latency_results] if latency_results else []
    total_verifier = sum(l.n_verifier_calls for l in latency_results) if latency_results else 0

    per_instance = []
    for i in range(n):
        entry: dict = {
            "hallucination_rate": faithfulness_results[i].hallucination_rate,
            "claim_precision": faithfulness_results[i].claim_precision,
        }
        if i < len(rouge_results):
            entry["rouge_l_f1"] = rouge_results[i].f1
        if i < len(latency_results):
            entry["ms_per_token"] = latency_results[i].ms_per_token
        per_instance.append(entry)

    return BenchmarkMetrics(
        n_instances=n,
        mean_hallucination_rate=sum(hall_rates) / n,
        mean_claim_precision=sum(precisions) / n,
        mean_claim_recall=sum(recalls) / len(recalls) if recalls else float("nan"),
        mean_rouge_l_f1=sum(rouge_f1s) / len(rouge_f1s) if rouge_f1s else 0.0,
        mean_ms_per_token=sum(ms_per_tokens) / len(ms_per_tokens) if ms_per_tokens else 0.0,
        total_verifier_calls=total_verifier,
        per_instance=per_instance,
    )
