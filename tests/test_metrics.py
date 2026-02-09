"""Tests for evaluation metrics (Section 3 of eval plan)."""

import math
import time

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.metrics import (
    BenchmarkMetrics,
    FaithfulnessMetrics,
    LatencyMetrics,
    LatencyTracker,
    ROUGELScore,
    aggregate_metrics,
    compute_faithfulness,
    rouge_l,
)


def _make_esbg(
    nodes: list[tuple[str, ClaimType, ClaimStatus]],
) -> EvidenceScopedBeliefGraph:
    """Helper: create an ESBG with typed nodes."""
    g = EvidenceScopedBeliefGraph()
    for nid, ctype, status in nodes:
        node = ESBGNode(
            node_id=nid,
            claim=AtomicClaim(claim_id=nid, text=f"Claim {nid}"),
            claim_type=ctype,
            status=status,
            support_mass=0.9 if ctype == ClaimType.VERIFIED else 0.1,
        )
        g.add_node(node)
    return g


class TestComputeFaithfulness:
    def test_all_verified(self):
        g = _make_esbg([
            ("c1", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
            ("c2", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
        ])
        result = compute_faithfulness(g)
        assert result.hallucination_rate == pytest.approx(0.0)
        assert result.claim_precision == pytest.approx(1.0)
        assert result.n_supported == 2

    def test_all_unsupported(self):
        g = _make_esbg([
            ("c1", ClaimType.UNSUPPORTED, ClaimStatus.UNKNOWN),
            ("c2", ClaimType.UNSUPPORTED, ClaimStatus.UNKNOWN),
        ])
        result = compute_faithfulness(g)
        assert result.hallucination_rate == pytest.approx(1.0)
        assert result.claim_precision == pytest.approx(0.0)

    def test_mixed(self):
        g = _make_esbg([
            ("c1", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
            ("c2", ClaimType.UNSUPPORTED, ClaimStatus.UNKNOWN),
            ("c3", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
            ("c4", ClaimType.UNSUPPORTED, ClaimStatus.CONTRADICTED),
        ])
        result = compute_faithfulness(g)
        assert result.hallucination_rate == pytest.approx(0.5)
        assert result.claim_precision == pytest.approx(0.5)
        assert result.n_supported == 2
        assert result.n_contradicted == 1
        assert result.n_unsupported == 1

    def test_with_reference_claims(self):
        g = _make_esbg([
            ("c1", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
            ("c2", ClaimType.VERIFIED, ClaimStatus.ENTAILED),
        ])
        # Reference has 4 claims, we verified 2 of them
        result = compute_faithfulness(g, reference_claim_ids={"c1", "c2", "c3", "c4"})
        assert result.claim_recall == pytest.approx(0.5)
        assert result.n_reference_claims == 4

    def test_empty_graph(self):
        g = EvidenceScopedBeliefGraph()
        result = compute_faithfulness(g)
        assert result.hallucination_rate == 0.0
        assert result.claim_precision == 1.0


class TestROUGEL:
    def test_identical_strings(self):
        result = rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)

    def test_no_overlap(self):
        result = rouge_l("hello world", "goodbye universe")
        assert result.f1 == pytest.approx(0.0)

    def test_partial_overlap(self):
        result = rouge_l("the cat sat", "the cat sat on the mat")
        assert result.precision == pytest.approx(1.0)  # all hyp tokens in ref
        assert result.recall == pytest.approx(0.5)  # 3/6 ref tokens matched
        assert result.f1 > 0.0

    def test_empty_hypothesis(self):
        result = rouge_l("", "the cat sat")
        assert result.f1 == pytest.approx(0.0)

    def test_empty_reference(self):
        result = rouge_l("the cat sat", "")
        assert result.f1 == pytest.approx(0.0)

    def test_case_insensitive(self):
        result = rouge_l("The Cat SAT", "the cat sat")
        assert result.f1 == pytest.approx(1.0)

    def test_subsequence_not_substring(self):
        # LCS should find "a c" = length 2, not requiring contiguity
        result = rouge_l("a b c", "a c")
        assert result.recall == pytest.approx(1.0)  # 2/2 ref tokens in LCS
        assert result.precision == pytest.approx(2.0 / 3.0)


class TestLatencyTracker:
    def test_basic_tracking(self):
        with LatencyTracker() as tracker:
            tracker.record_verifier_call()
            tracker.record_verifier_call()
            tracker.record_retriever_call()

        metrics = tracker.compute(n_tokens=100)
        assert metrics.total_time_seconds >= 0
        assert metrics.n_tokens_generated == 100
        assert metrics.n_verifier_calls == 2
        assert metrics.n_retriever_calls == 1
        assert metrics.ms_per_token >= 0

    def test_tokens_per_second(self):
        metrics = LatencyMetrics(
            total_time_seconds=2.0,
            n_tokens_generated=100,
            ms_per_token=20.0,
            n_verifier_calls=0,
            n_retriever_calls=0,
        )
        assert metrics.tokens_per_second == pytest.approx(50.0)


class TestAggregateMetrics:
    def test_basic_aggregation(self):
        faithfulness = [
            FaithfulnessMetrics(0.1, 0.9, 0.8, 9, 0, 1, 10),
            FaithfulnessMetrics(0.2, 0.8, 0.7, 8, 0, 2, 10),
        ]
        rouges = [
            ROUGELScore(0.8, 0.7, 0.75),
            ROUGELScore(0.9, 0.8, 0.85),
        ]
        latencies = [
            LatencyMetrics(1.0, 100, 10.0, 5, 3),
            LatencyMetrics(1.5, 150, 10.0, 7, 4),
        ]

        result = aggregate_metrics(faithfulness, rouges, latencies)
        assert result.n_instances == 2
        assert result.mean_hallucination_rate == pytest.approx(0.15)
        assert result.mean_claim_precision == pytest.approx(0.85)
        assert result.mean_claim_recall == pytest.approx(0.75)
        assert result.mean_rouge_l_f1 == pytest.approx(0.80)
        assert result.total_verifier_calls == 12

    def test_empty(self):
        result = aggregate_metrics([], [], [])
        assert result.n_instances == 0
        assert result.mean_hallucination_rate == 0.0
