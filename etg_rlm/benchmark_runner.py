"""Canonical Benchmark Runner for the ETG evaluation framework.

Orchestrates the full evaluation pipeline across all models and datasets:

    Models:
        1. Zero-Shot GPT-4 (no retrieval, no verification)
        2. Standard RAG with Contriever (retrieval, no verification)
        3. Self-CheckGPT (zero-resource hallucination detection)
        4. ETG (evidence-typed generation with multi-view verification)

    Datasets:
        1. TruthfulQA (817 instances) -- truthfulness under adversarial priors
        2. HaluEval (1000 instances) -- hallucination detection benchmark
        3. HotpotQA (500 instances) -- multi-hop reasoning
        4. Natural Questions (1000 instances) -- open-domain factoid QA
        5. ELI5 (500 instances) -- long-form explanatory answers

    Metrics (per model x dataset):
        - FactScore (Claim Precision / Claim Recall)
        - Citation Precision / Citation Recall
        - Logic-Step Verification (for multi-hop datasets)
        - ROUGE-L F1
        - Latency (ms/token)
        - Self-CheckGPT consistency (for Self-CheckGPT baseline)

References:
    [1] Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," ACL 2022.
    [2] Li et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark," EMNLP 2023.
    [3] Yang et al., "HotpotQA," EMNLP 2018.
    [4] Gao et al., "ALCE: Attributed Language Model Evaluation," ACL 2023.
    [5] Rashkin et al., "Measuring Attribution in NLG Models," ACL 2022.
    [6] Kwiatkowski et al., "Natural Questions," TACL 2019.
    [7] Manakul et al., "SelfCheckGPT," EMNLP 2023.
    [8] Min et al., "FActScore," EMNLP 2023.
    [9] Fan et al., "ELI5: Long Form Question Answering," ACL 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan
from etg_rlm.factscore import FactScoreResult
from etg_rlm.citation_metrics import CitationMetricsResult
from etg_rlm.logic_verification import ChainVerificationResult
from etg_rlm.self_check import SelfCheckResult
from etg_rlm.metrics import ROUGELScore, LatencyMetrics


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """The four model configurations in the canonical evaluation."""

    ZERO_SHOT = "zero_shot_gpt4"
    STANDARD_RAG = "standard_rag_contriever"
    SELF_CHECK_GPT = "self_check_gpt"
    ETG = "etg"


class BenchmarkDataset(Enum):
    """The five benchmark datasets."""

    TRUTHFUL_QA = "truthfulqa"
    HALU_EVAL = "halueval"
    HOTPOT_QA = "hotpotqa"
    NATURAL_QUESTIONS = "natural_questions"
    ELI5 = "eli5"


# ---------------------------------------------------------------------------
# Evaluation instance and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkInstance:
    """A single evaluation instance from a benchmark dataset.

    Attributes:
        instance_id: unique identifier
        dataset: which benchmark this belongs to
        query: the input query/question
        reference_answer: gold-standard reference (for ROUGE)
        reference_claims: ground-truth atomic claims (for recall)
        corpus_id: evidence corpus to use
        is_multi_hop: whether this requires multi-step reasoning
        n_hops: number of reasoning hops (for multi-hop instances)
    """

    instance_id: str
    dataset: BenchmarkDataset
    query: str
    reference_answer: str = ""
    reference_claims: tuple[AtomicClaim, ...] = ()
    corpus_id: str = "default"
    is_multi_hop: bool = False
    n_hops: int = 1


@dataclass
class ModelOutput:
    """Output from running a model on a benchmark instance.

    Attributes:
        model: which model produced this output
        instance_id: which instance this corresponds to
        generated_text: the raw generated text
        final_text: text after any filtering/verification
        claims: extracted atomic claims
        evidence_spans: evidence spans used (if applicable)
        citations: (claim_id, span) pairs for citation metrics
        reasoning_steps: for multi-hop, the chain of reasoning
    """

    model: ModelType
    instance_id: str
    generated_text: str
    final_text: str
    claims: list[AtomicClaim] = field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    citations: list[tuple[str, EvidenceSpan]] = field(default_factory=list)
    reasoning_steps: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-instance evaluation result
# ---------------------------------------------------------------------------


@dataclass
class InstanceResult:
    """Complete evaluation result for one model on one instance.

    Attributes:
        model: model identifier
        dataset: dataset identifier
        instance_id: instance identifier
        factscore: FactScore metrics (claim precision/recall)
        citation: citation precision/recall (if applicable)
        logic: logic-step verification (if multi-hop)
        rouge: ROUGE-L scores
        latency: timing information
        self_check: Self-CheckGPT results (if applicable)
    """

    model: ModelType
    dataset: BenchmarkDataset
    instance_id: str
    factscore: FactScoreResult | None = None
    citation: CitationMetricsResult | None = None
    logic: ChainVerificationResult | None = None
    rouge: ROUGELScore | None = None
    latency: LatencyMetrics | None = None
    self_check: SelfCheckResult | None = None


# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------


@dataclass
class DatasetResults:
    """Aggregated results for one model on one dataset.

    Attributes:
        model: model identifier
        dataset: dataset identifier
        n_instances: number of instances evaluated
        mean_factscore: average FactScore
        mean_claim_precision: average claim precision
        mean_claim_recall: average claim recall
        mean_citation_precision: average citation precision
        mean_citation_recall: average citation recall
        mean_step_accuracy: average logic-step accuracy (multi-hop)
        mean_rouge_f1: average ROUGE-L F1
        mean_ms_per_token: average latency
        per_instance: detailed per-instance results
    """

    model: ModelType
    dataset: BenchmarkDataset
    n_instances: int = 0
    mean_factscore: float = 0.0
    mean_claim_precision: float = 0.0
    mean_claim_recall: float = 0.0
    mean_citation_precision: float = 0.0
    mean_citation_recall: float = 0.0
    mean_step_accuracy: float = 0.0
    mean_rouge_f1: float = 0.0
    mean_ms_per_token: float = 0.0
    per_instance: list[InstanceResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Full benchmark report: all models x all datasets.

    The canonical output of the evaluation framework.
    """

    results: dict[str, DatasetResults] = field(default_factory=dict)
    model_rankings: dict[str, list[str]] = field(default_factory=dict)

    def get_result(
        self, model: ModelType, dataset: BenchmarkDataset
    ) -> DatasetResults | None:
        key = f"{model.value}__{dataset.value}"
        return self.results.get(key)

    def add_result(self, result: DatasetResults) -> None:
        key = f"{result.model.value}__{result.dataset.value}"
        self.results[key] = result

    def compute_rankings(self) -> None:
        """Compute model rankings per metric across all datasets."""
        metrics = [
            "mean_factscore", "mean_claim_precision", "mean_citation_precision",
            "mean_rouge_f1",
        ]
        for metric in metrics:
            # Collect (model, mean_across_datasets) pairs
            model_scores: dict[str, list[float]] = {}
            for key, result in self.results.items():
                model_name = result.model.value
                score = getattr(result, metric, 0.0)
                model_scores.setdefault(model_name, []).append(score)

            avg_scores = {
                m: sum(s) / len(s) if s else 0.0
                for m, s in model_scores.items()
            }
            ranked = sorted(avg_scores, key=lambda m: avg_scores[m], reverse=True)
            self.model_rankings[metric] = ranked


# ---------------------------------------------------------------------------
# Runner protocol and orchestrator
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelRunner(Protocol):
    """Protocol for running a model on a benchmark instance."""

    def run(self, instance: BenchmarkInstance) -> ModelOutput: ...


@runtime_checkable
class MetricComputer(Protocol):
    """Protocol for computing all metrics on a model output."""

    def compute(
        self, output: ModelOutput, instance: BenchmarkInstance
    ) -> InstanceResult: ...


def aggregate_dataset_results(
    instance_results: list[InstanceResult],
    model: ModelType,
    dataset: BenchmarkDataset,
) -> DatasetResults:
    """Aggregate per-instance results into dataset-level statistics.

    Args:
        instance_results: results for each instance
        model: model identifier
        dataset: dataset identifier

    Returns:
        DatasetResults with aggregated means.
    """
    n = len(instance_results)
    if n == 0:
        return DatasetResults(model=model, dataset=dataset)

    # FactScore
    factscores = [r.factscore.factscore for r in instance_results if r.factscore]
    precisions = [r.factscore.claim_precision for r in instance_results if r.factscore]
    recalls = [
        r.factscore.claim_recall for r in instance_results
        if r.factscore and not _is_nan(r.factscore.claim_recall)
    ]

    # Citation
    cit_prec = [r.citation.citation_precision for r in instance_results if r.citation]
    cit_rec = [r.citation.citation_recall for r in instance_results if r.citation]

    # Logic
    step_acc = [r.logic.step_accuracy for r in instance_results if r.logic]

    # ROUGE
    rouge_f1 = [r.rouge.f1 for r in instance_results if r.rouge]

    # Latency
    ms_pt = [r.latency.ms_per_token for r in instance_results if r.latency]

    return DatasetResults(
        model=model,
        dataset=dataset,
        n_instances=n,
        mean_factscore=_safe_mean(factscores),
        mean_claim_precision=_safe_mean(precisions),
        mean_claim_recall=_safe_mean(recalls),
        mean_citation_precision=_safe_mean(cit_prec),
        mean_citation_recall=_safe_mean(cit_rec),
        mean_step_accuracy=_safe_mean(step_acc),
        mean_rouge_f1=_safe_mean(rouge_f1),
        mean_ms_per_token=_safe_mean(ms_pt),
        per_instance=instance_results,
    )


def run_benchmark(
    models: dict[ModelType, ModelRunner],
    instances: list[BenchmarkInstance],
    metric_computer: MetricComputer,
) -> BenchmarkReport:
    """Run the full canonical benchmark: all models x all instances.

    Args:
        models: mapping from model type to runner
        instances: all benchmark instances across all datasets
        metric_computer: computes all metrics for a model output

    Returns:
        BenchmarkReport with aggregated results and rankings.
    """
    report = BenchmarkReport()

    # Group instances by dataset
    by_dataset: dict[BenchmarkDataset, list[BenchmarkInstance]] = {}
    for inst in instances:
        by_dataset.setdefault(inst.dataset, []).append(inst)

    # Run each model on each dataset
    for model_type, runner in models.items():
        for dataset, dataset_instances in by_dataset.items():
            instance_results: list[InstanceResult] = []

            for inst in dataset_instances:
                output = runner.run(inst)
                result = metric_computer.compute(output, inst)
                instance_results.append(result)

            agg = aggregate_dataset_results(instance_results, model_type, dataset)
            report.add_result(agg)

    report.compute_rankings()
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_nan(x: float) -> bool:
    import math
    return math.isnan(x)
