"""Tests for the canonical benchmark runner orchestrator."""

import pytest

from etg_rlm.core import AtomicClaim, EvidenceSpan
from etg_rlm.factscore import FactScoreResult
from etg_rlm.citation_metrics import CitationMetricsResult
from etg_rlm.logic_verification import ChainVerificationResult
from etg_rlm.metrics import ROUGELScore, LatencyMetrics
from etg_rlm.benchmark_runner import (
    ModelType,
    BenchmarkDataset,
    BenchmarkInstance,
    ModelOutput,
    InstanceResult,
    DatasetResults,
    BenchmarkReport,
    aggregate_dataset_results,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubModelRunner:
    """Stub runner: returns fixed output."""

    def __init__(self, model: ModelType, factscore: float = 0.8):
        self.model = model
        self.factscore = factscore

    def run(self, instance: BenchmarkInstance) -> ModelOutput:
        return ModelOutput(
            model=self.model,
            instance_id=instance.instance_id,
            generated_text=f"Answer to {instance.query}",
            final_text=f"Verified answer to {instance.query}",
            claims=[AtomicClaim(claim_id="c1", text="A fact")],
        )


class StubMetricComputer:
    """Stub metric computer: returns configurable scores."""

    def __init__(self, factscore: float = 0.8, citation_prec: float = 0.9):
        self.factscore = factscore
        self.citation_prec = citation_prec

    def compute(
        self, output: ModelOutput, instance: BenchmarkInstance
    ) -> InstanceResult:
        return InstanceResult(
            model=output.model,
            dataset=instance.dataset,
            instance_id=instance.instance_id,
            factscore=FactScoreResult(
                factscore=self.factscore,
                claim_precision=self.factscore,
                claim_recall=0.7,
                n_claims=5,
                n_supported=4,
                n_unsupported=1,
            ),
            citation=CitationMetricsResult(
                citation_precision=self.citation_prec,
                citation_recall=0.8,
                n_total_citations=5,
                n_valid_citations=4,
                n_entailed_claims=5,
                n_cited_entailed=4,
            ),
            rouge=ROUGELScore(precision=0.7, recall=0.6, f1=0.65),
        )


# ---------------------------------------------------------------------------
# Test aggregate_dataset_results
# ---------------------------------------------------------------------------


class TestAggregateDatasetResults:
    def test_empty(self):
        result = aggregate_dataset_results(
            [], ModelType.ETG, BenchmarkDataset.TRUTHFUL_QA
        )
        assert result.n_instances == 0
        assert result.mean_factscore == 0.0

    def test_single_instance(self):
        instance = InstanceResult(
            model=ModelType.ETG,
            dataset=BenchmarkDataset.TRUTHFUL_QA,
            instance_id="i1",
            factscore=FactScoreResult(
                factscore=0.9, claim_precision=0.9, claim_recall=0.8,
                n_claims=5, n_supported=4, n_unsupported=1,
            ),
            citation=CitationMetricsResult(
                citation_precision=1.0, citation_recall=0.8,
                n_total_citations=5, n_valid_citations=5,
                n_entailed_claims=5, n_cited_entailed=4,
            ),
            rouge=ROUGELScore(0.7, 0.6, 0.65),
        )
        result = aggregate_dataset_results(
            [instance], ModelType.ETG, BenchmarkDataset.TRUTHFUL_QA
        )
        assert result.n_instances == 1
        assert result.mean_factscore == 0.9
        assert result.mean_citation_precision == 1.0
        assert result.mean_rouge_f1 == 0.65

    def test_multiple_instances(self):
        instances = [
            InstanceResult(
                model=ModelType.ETG,
                dataset=BenchmarkDataset.HOTPOT_QA,
                instance_id=f"i{i}",
                factscore=FactScoreResult(
                    factscore=fs, claim_precision=fs, claim_recall=0.5,
                    n_claims=3, n_supported=2, n_unsupported=1,
                ),
                rouge=ROUGELScore(0.5, 0.5, f1),
            )
            for i, (fs, f1) in enumerate([(0.8, 0.6), (1.0, 0.8)])
        ]
        result = aggregate_dataset_results(
            instances, ModelType.ETG, BenchmarkDataset.HOTPOT_QA
        )
        assert result.mean_factscore == pytest.approx(0.9)
        assert result.mean_rouge_f1 == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Test BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_add_and_get_result(self):
        report = BenchmarkReport()
        result = DatasetResults(
            model=ModelType.ETG,
            dataset=BenchmarkDataset.TRUTHFUL_QA,
            n_instances=100,
            mean_factscore=0.95,
        )
        report.add_result(result)
        retrieved = report.get_result(ModelType.ETG, BenchmarkDataset.TRUTHFUL_QA)
        assert retrieved is not None
        assert retrieved.mean_factscore == 0.95

    def test_missing_result_returns_none(self):
        report = BenchmarkReport()
        assert report.get_result(ModelType.ETG, BenchmarkDataset.ELI5) is None

    def test_compute_rankings(self):
        report = BenchmarkReport()

        # ETG is better
        report.add_result(DatasetResults(
            model=ModelType.ETG, dataset=BenchmarkDataset.TRUTHFUL_QA,
            n_instances=100, mean_factscore=0.95, mean_claim_precision=0.95,
            mean_citation_precision=0.90, mean_rouge_f1=0.70,
        ))
        report.add_result(DatasetResults(
            model=ModelType.ZERO_SHOT, dataset=BenchmarkDataset.TRUTHFUL_QA,
            n_instances=100, mean_factscore=0.60, mean_claim_precision=0.60,
            mean_citation_precision=0.40, mean_rouge_f1=0.65,
        ))

        report.compute_rankings()
        assert report.model_rankings["mean_factscore"][0] == "etg"
        assert report.model_rankings["mean_factscore"][1] == "zero_shot_gpt4"


# ---------------------------------------------------------------------------
# Test run_benchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_full_benchmark(self):
        models = {
            ModelType.ETG: StubModelRunner(ModelType.ETG, 0.95),
            ModelType.ZERO_SHOT: StubModelRunner(ModelType.ZERO_SHOT, 0.60),
        }
        instances = [
            BenchmarkInstance(
                instance_id="tq_001",
                dataset=BenchmarkDataset.TRUTHFUL_QA,
                query="What is gravity?",
            ),
            BenchmarkInstance(
                instance_id="tq_002",
                dataset=BenchmarkDataset.TRUTHFUL_QA,
                query="What is light?",
            ),
            BenchmarkInstance(
                instance_id="hp_001",
                dataset=BenchmarkDataset.HOTPOT_QA,
                query="Who founded the company?",
                is_multi_hop=True,
                n_hops=2,
            ),
        ]

        report = run_benchmark(models, instances, StubMetricComputer())

        # Should have 2 models x 2 datasets = 4 results
        assert len(report.results) == 4

        etg_tq = report.get_result(ModelType.ETG, BenchmarkDataset.TRUTHFUL_QA)
        assert etg_tq is not None
        assert etg_tq.n_instances == 2
        assert etg_tq.mean_factscore == pytest.approx(0.8)

        etg_hp = report.get_result(ModelType.ETG, BenchmarkDataset.HOTPOT_QA)
        assert etg_hp is not None
        assert etg_hp.n_instances == 1

    def test_empty_instances(self):
        models = {ModelType.ETG: StubModelRunner(ModelType.ETG)}
        report = run_benchmark(models, [], StubMetricComputer())
        assert len(report.results) == 0

    def test_rankings_computed(self):
        models = {
            ModelType.ETG: StubModelRunner(ModelType.ETG),
            ModelType.STANDARD_RAG: StubModelRunner(ModelType.STANDARD_RAG),
        }
        instances = [
            BenchmarkInstance(
                instance_id="i1",
                dataset=BenchmarkDataset.TRUTHFUL_QA,
                query="Q1",
            ),
        ]
        report = run_benchmark(models, instances, StubMetricComputer())
        assert "mean_factscore" in report.model_rankings


# ---------------------------------------------------------------------------
# Test model and dataset enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_model_types(self):
        assert len(ModelType) == 4
        assert ModelType.ETG.value == "etg"
        assert ModelType.SELF_CHECK_GPT.value == "self_check_gpt"

    def test_benchmark_datasets(self):
        assert len(BenchmarkDataset) == 5
        assert BenchmarkDataset.ELI5.value == "eli5"
        assert BenchmarkDataset.TRUTHFUL_QA.value == "truthfulqa"

    def test_benchmark_instance_multi_hop(self):
        inst = BenchmarkInstance(
            instance_id="hp_001",
            dataset=BenchmarkDataset.HOTPOT_QA,
            query="Multi-hop question",
            is_multi_hop=True,
            n_hops=3,
        )
        assert inst.is_multi_hop is True
        assert inst.n_hops == 3
