"""Evaluation harness for comparative benchmarking of ETG vs. baselines.

Runs ETG and all baselines on a dataset, computes metrics, and produces
a structured comparison report. Corresponds to the full evaluation plan.

KPI targets from the plan:
    - Hallucination rate < 1% (>90% reduction vs. Standard RAG)
    - User preference > 75% vs. Standard RAG
    - Latency < 500ms/token for interactive use
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimType, EvidenceScopedBeliefGraph
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
from etg_rlm.baselines import BaselineConfig, BaselineResult, BaselineType


# ---------------------------------------------------------------------------
# Evaluation dataset protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalInstance:
    """A single evaluation instance (query + reference).

    Attributes:
        instance_id: unique identifier
        query: the input query
        reference_answer: gold-standard reference answer (for ROUGE-L)
        reference_claim_ids: IDs of ground-truth claims derivable from E (for recall)
        corpus_id: evidence corpus to use
    """

    instance_id: str
    query: str
    reference_answer: str = ""
    reference_claim_ids: frozenset[str] = frozenset()
    corpus_id: str = "default"


# ---------------------------------------------------------------------------
# System result (abstract over ETG and baselines)
# ---------------------------------------------------------------------------


@dataclass
class SystemResult:
    """Unified result from any system (ETG or baseline).

    Attributes:
        system_name: name of the system that produced this result
        instance_id: which eval instance this corresponds to
        generated_text: the raw generated text
        final_text: the text after any filtering/verification
        n_claims: total number of claims extracted
        n_verified: number of claims that passed verification
        n_rejected: number of claims that were rejected
        esbg: the ESBG (only for ETG, None for baselines)
    """

    system_name: str
    instance_id: str
    generated_text: str
    final_text: str
    n_claims: int = 0
    n_verified: int = 0
    n_rejected: int = 0
    esbg: EvidenceScopedBeliefGraph | None = None


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------


class InstanceEvaluation(NamedTuple):
    """Full evaluation of one instance from one system."""

    system_name: str
    instance_id: str
    faithfulness: FaithfulnessMetrics
    rouge: ROUGELScore
    latency: LatencyMetrics | None


# ---------------------------------------------------------------------------
# Comparative report
# ---------------------------------------------------------------------------


@dataclass
class SystemReport:
    """Aggregated evaluation report for a single system."""

    system_name: str
    metrics: BenchmarkMetrics
    per_instance: list[InstanceEvaluation] = field(default_factory=list)


@dataclass
class ComparativeReport:
    """Full comparative report across all systems.

    Attributes:
        systems: per-system aggregated reports
        kpi_check: whether KPI targets from the eval plan are met
    """

    systems: list[SystemReport] = field(default_factory=list)
    kpi_check: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        return {
            "systems": [
                {
                    "name": s.system_name,
                    "n_instances": s.metrics.n_instances,
                    "mean_hallucination_rate": round(s.metrics.mean_hallucination_rate, 4),
                    "mean_claim_precision": round(s.metrics.mean_claim_precision, 4),
                    "mean_claim_recall": round(s.metrics.mean_claim_recall, 4),
                    "mean_rouge_l_f1": round(s.metrics.mean_rouge_l_f1, 4),
                    "mean_ms_per_token": round(s.metrics.mean_ms_per_token, 2),
                    "total_verifier_calls": s.metrics.total_verifier_calls,
                }
                for s in self.systems
            ],
            "kpi_check": self.kpi_check,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def evaluate_instance(
    result: SystemResult,
    instance: EvalInstance,
    latency: LatencyMetrics | None = None,
) -> InstanceEvaluation:
    """Evaluate a single system result against an eval instance.

    Computes faithfulness metrics (from ESBG if available, else from
    claim counts), ROUGE-L against the reference answer, and attaches
    latency data if provided.
    """
    # Faithfulness
    if result.esbg is not None:
        ref_ids = set(instance.reference_claim_ids) if instance.reference_claim_ids else None
        faithfulness = compute_faithfulness(result.esbg, ref_ids)
    else:
        # For baselines without ESBG, compute from claim counts
        n_total = result.n_claims or 1
        faithfulness = FaithfulnessMetrics(
            hallucination_rate=(result.n_rejected / n_total) if n_total > 0 else 0.0,
            claim_precision=(result.n_verified / n_total) if n_total > 0 else 1.0,
            claim_recall=float("nan"),
            n_supported=result.n_verified,
            n_contradicted=0,
            n_unsupported=result.n_rejected,
            n_reference_claims=len(instance.reference_claim_ids),
        )

    # ROUGE-L
    rouge = rouge_l(result.final_text, instance.reference_answer) if instance.reference_answer else ROUGELScore(0.0, 0.0, 0.0)

    return InstanceEvaluation(
        system_name=result.system_name,
        instance_id=instance.instance_id,
        faithfulness=faithfulness,
        rouge=rouge,
        latency=latency,
    )


def build_report(
    evaluations: list[InstanceEvaluation],
    system_name: str,
) -> SystemReport:
    """Build an aggregated report for one system from instance evaluations."""
    faithfulness_list = [e.faithfulness for e in evaluations]
    rouge_list = [e.rouge for e in evaluations]
    latency_list = [e.latency for e in evaluations if e.latency is not None]

    metrics = aggregate_metrics(faithfulness_list, rouge_list, latency_list)

    return SystemReport(
        system_name=system_name,
        metrics=metrics,
        per_instance=evaluations,
    )


def check_kpis(
    etg_report: SystemReport,
    rag_report: SystemReport | None = None,
) -> dict[str, bool]:
    """Check whether the ETG system meets the KPI targets.

    KPIs from the evaluation plan:
        1. Hallucination rate < 1%
        2. >90% reduction in hallucination rate vs. Standard RAG
        3. ROUGE-L maintained or improved vs. baselines
        4. Latency < 500ms/token
    """
    kpis: dict[str, bool] = {}

    etg_hall = etg_report.metrics.mean_hallucination_rate
    kpis["hallucination_rate_below_1pct"] = etg_hall < 0.01

    if rag_report is not None:
        rag_hall = rag_report.metrics.mean_hallucination_rate
        if rag_hall > 0:
            reduction = 1.0 - (etg_hall / rag_hall)
            kpis["90pct_reduction_vs_rag"] = reduction >= 0.90
        else:
            kpis["90pct_reduction_vs_rag"] = True

        kpis["rouge_maintained_vs_rag"] = (
            etg_report.metrics.mean_rouge_l_f1 >= rag_report.metrics.mean_rouge_l_f1 * 0.95
        )

    kpis["latency_below_500ms"] = etg_report.metrics.mean_ms_per_token < 500.0

    return kpis


def build_comparative_report(
    system_reports: list[SystemReport],
) -> ComparativeReport:
    """Build a comparative report across all systems.

    Automatically identifies ETG and RAG reports for KPI checking.
    """
    etg_report = None
    rag_report = None

    for report in system_reports:
        name_lower = report.system_name.lower()
        if "etg" in name_lower or "ebrg" in name_lower:
            etg_report = report
        elif "standard rag" in name_lower or name_lower == "standard_rag":
            rag_report = report

    kpis: dict[str, bool] = {}
    if etg_report is not None:
        kpis = check_kpis(etg_report, rag_report)

    return ComparativeReport(
        systems=system_reports,
        kpi_check=kpis,
    )
