"""Tests for the evaluation harness."""

import math

import pytest

from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.metrics import (
    FaithfulnessMetrics,
    LatencyMetrics,
    ROUGELScore,
)
from etg_rlm.evaluation import (
    ComparativeReport,
    EvalInstance,
    InstanceEvaluation,
    SystemReport,
    SystemResult,
    build_comparative_report,
    build_report,
    check_kpis,
    evaluate_instance,
)
from etg_rlm.metrics import BenchmarkMetrics


def _make_esbg_result(
    n_verified: int = 3, n_unsupported: int = 0
) -> EvidenceScopedBeliefGraph:
    g = EvidenceScopedBeliefGraph()
    for i in range(n_verified):
        g.add_node(ESBGNode(
            node_id=f"v{i}",
            claim=AtomicClaim(claim_id=f"v{i}", text=f"Verified {i}"),
            claim_type=ClaimType.VERIFIED,
            status=ClaimStatus.ENTAILED,
            support_mass=0.9,
        ))
    for i in range(n_unsupported):
        g.add_node(ESBGNode(
            node_id=f"u{i}",
            claim=AtomicClaim(claim_id=f"u{i}", text=f"Unsupported {i}"),
            claim_type=ClaimType.UNSUPPORTED,
            status=ClaimStatus.UNKNOWN,
            support_mass=0.1,
        ))
    return g


class TestEvaluateInstance:
    def test_with_esbg(self):
        esbg = _make_esbg_result(n_verified=3, n_unsupported=1)
        result = SystemResult(
            system_name="ETG",
            instance_id="q1",
            generated_text="text",
            final_text="Verified 0 Verified 1 Verified 2",
            esbg=esbg,
        )
        instance = EvalInstance(
            instance_id="q1",
            query="test",
            reference_answer="Verified 0 Verified 1 Verified 2",
        )
        ev = evaluate_instance(result, instance)
        assert ev.faithfulness.hallucination_rate == pytest.approx(0.25)
        assert ev.faithfulness.claim_precision == pytest.approx(0.75)
        assert ev.rouge.f1 == pytest.approx(1.0)

    def test_without_esbg(self):
        result = SystemResult(
            system_name="Baseline",
            instance_id="q1",
            generated_text="text",
            final_text="answer",
            n_claims=10,
            n_verified=8,
            n_rejected=2,
        )
        instance = EvalInstance(instance_id="q1", query="test")
        ev = evaluate_instance(result, instance)
        assert ev.faithfulness.hallucination_rate == pytest.approx(0.2)

    def test_with_reference_claims(self):
        esbg = _make_esbg_result(n_verified=2)
        result = SystemResult(
            system_name="ETG",
            instance_id="q1",
            generated_text="text",
            final_text="text",
            esbg=esbg,
        )
        instance = EvalInstance(
            instance_id="q1",
            query="test",
            reference_claim_ids=frozenset({"v0", "v1", "v2", "v3"}),
        )
        ev = evaluate_instance(result, instance)
        assert ev.faithfulness.claim_recall == pytest.approx(0.5)


class TestBuildReport:
    def test_aggregates_correctly(self):
        evals = [
            InstanceEvaluation(
                system_name="ETG",
                instance_id="q1",
                faithfulness=FaithfulnessMetrics(0.0, 1.0, 0.8, 5, 0, 0, 5),
                rouge=ROUGELScore(0.9, 0.8, 0.85),
                latency=LatencyMetrics(1.0, 100, 10.0, 5, 3),
            ),
            InstanceEvaluation(
                system_name="ETG",
                instance_id="q2",
                faithfulness=FaithfulnessMetrics(0.1, 0.9, 0.7, 4, 0, 1, 5),
                rouge=ROUGELScore(0.8, 0.7, 0.75),
                latency=LatencyMetrics(1.5, 150, 10.0, 7, 4),
            ),
        ]
        report = build_report(evals, "ETG")
        assert report.system_name == "ETG"
        assert report.metrics.n_instances == 2
        assert report.metrics.mean_hallucination_rate == pytest.approx(0.05)


class TestCheckKPIs:
    def test_passes_all_kpis(self):
        etg_report = SystemReport(
            system_name="ETG",
            metrics=BenchmarkMetrics(
                n_instances=10,
                mean_hallucination_rate=0.005,
                mean_claim_precision=0.995,
                mean_claim_recall=0.9,
                mean_rouge_l_f1=0.85,
                mean_ms_per_token=50.0,
                total_verifier_calls=100,
            ),
        )
        rag_report = SystemReport(
            system_name="Standard RAG",
            metrics=BenchmarkMetrics(
                n_instances=10,
                mean_hallucination_rate=0.15,
                mean_claim_precision=0.85,
                mean_claim_recall=0.9,
                mean_rouge_l_f1=0.80,
                mean_ms_per_token=20.0,
                total_verifier_calls=0,
            ),
        )
        kpis = check_kpis(etg_report, rag_report)
        assert kpis["hallucination_rate_below_1pct"] is True
        assert kpis["90pct_reduction_vs_rag"] is True
        assert kpis["rouge_maintained_vs_rag"] is True
        assert kpis["latency_below_500ms"] is True

    def test_fails_when_hallucination_high(self):
        etg_report = SystemReport(
            system_name="ETG",
            metrics=BenchmarkMetrics(
                n_instances=10,
                mean_hallucination_rate=0.05,
                mean_claim_precision=0.95,
                mean_claim_recall=0.9,
                mean_rouge_l_f1=0.85,
                mean_ms_per_token=50.0,
                total_verifier_calls=100,
            ),
        )
        kpis = check_kpis(etg_report)
        assert kpis["hallucination_rate_below_1pct"] is False


class TestComparativeReport:
    def test_to_json(self):
        report = ComparativeReport(
            systems=[
                SystemReport(
                    system_name="ETG",
                    metrics=BenchmarkMetrics(
                        n_instances=5,
                        mean_hallucination_rate=0.01,
                        mean_claim_precision=0.99,
                        mean_claim_recall=0.85,
                        mean_rouge_l_f1=0.82,
                        mean_ms_per_token=100.0,
                        total_verifier_calls=50,
                    ),
                ),
            ],
            kpi_check={"hallucination_rate_below_1pct": False},
        )
        json_str = report.to_json()
        assert "ETG" in json_str
        assert "hallucination_rate_below_1pct" in json_str

    def test_build_comparative(self):
        reports = [
            SystemReport(
                system_name="ETG",
                metrics=BenchmarkMetrics(5, 0.005, 0.995, 0.9, 0.85, 50.0, 100),
            ),
            SystemReport(
                system_name="Standard RAG",
                metrics=BenchmarkMetrics(5, 0.15, 0.85, 0.9, 0.80, 20.0, 0),
            ),
        ]
        comparative = build_comparative_report(reports)
        assert len(comparative.systems) == 2
        assert "hallucination_rate_below_1pct" in comparative.kpi_check
