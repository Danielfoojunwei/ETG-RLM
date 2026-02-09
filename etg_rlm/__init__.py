"""Evidence-Typed Generation for Recursive Language Models.

Thesis: Hallucinations arise from read/write entanglement in next-token
decoding. ETG is an RLM-native inference framework that externalizes belief
into evidence-scoped graphs and restricts generation to well-typed, entailed
claims, yielding exponential suppression of hallucinations under
inference-time scaling.

Paper: "Evidence-Typed Generation: Faithfulness as a Type System
for Recursive Language Models"
"""

from etg_rlm.core import (
    EvidenceSpan,
    AtomicClaim,
    ClaimStatus,
    ClaimType,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.verification import (
    VerificationView,
    ViewResult,
    MultiViewVerifier,
)
from etg_rlm.type_system import (
    TypeThresholds,
    EvidenceTypeChecker,
)
from etg_rlm.policy import (
    ActionType,
    PolicyAction,
    RecursionPolicy,
    UtilityWeightedPolicy,
)
from etg_rlm.bounds import (
    hallucination_upper_bound,
    optimal_view_allocation,
    check_zero_confabulation,
    inference_time_scaling_law,
)
from etg_rlm.algorithm import (
    ebrg,
    constrained_decode,
    EBRGResult,
    ConstrainedDecodingResult,
)
from etg_rlm.pipeline import (
    ETGPipeline,
    ETGConfig,
)
from etg_rlm.metrics import (
    compute_faithfulness,
    rouge_l,
    aggregate_metrics,
    FaithfulnessMetrics,
    ROUGELScore,
    LatencyTracker,
)
from etg_rlm.baselines import (
    BaselineType,
    BaselineConfig,
    StandardLLMBaseline,
    StandardRAGBaseline,
    RAGVerifierBaseline,
    SelfCritiqueBaseline,
)
from etg_rlm.evaluation import (
    EvalInstance,
    evaluate_instance,
    build_report,
    build_comparative_report,
    check_kpis,
    ComparativeReport,
)
from etg_rlm.views import (
    ViewType,
    ViewConfig,
    create_view,
    create_default_view_suite,
)
from etg_rlm.datasets import (
    DatasetName,
    DatasetConfig,
    TaskType,
    ALL_DATASET_CONFIGS,
    get_dataset_config,
    total_eval_instances,
)
from etg_rlm.human_eval import (
    FaithfulnessRating,
    FaithfulnessAnnotation,
    PairwiseAnnotation,
    PreferenceDimension,
    PreferenceChoice,
    aggregate_faithfulness,
    aggregate_preferences,
    fleiss_kappa,
    HumanEvalSummary,
)
from etg_rlm.ablations import (
    AblationType,
    AblationConfig,
    RandomPolicy,
    all_ablation_configs,
    make_no_multi_view_config,
    make_no_constraint_config,
    make_threshold_sweep_configs,
    make_policy_ablation_config,
)
from etg_rlm.statistics import (
    paired_t_test,
    cohens_d,
    bootstrap_ci,
    bootstrap_paired_ci,
    full_analysis,
    PairedTTestResult,
    EffectSizeResult,
    BootstrapCIResult,
    StatisticalAnalysis,
)
from etg_rlm.factscore import (
    FactScoreResult,
    BatchFactScoreResult,
    ClaimScoreResult,
    compute_factscore,
    aggregate_factscores,
    compute_factscore_with_retrieval,
)
from etg_rlm.citation_metrics import (
    Citation,
    CitationMetricsResult,
    BatchCitationResult,
    compute_citation_metrics,
    aggregate_citation_metrics,
    extract_citations_from_esbg,
)
from etg_rlm.logic_verification import (
    StepValidity,
    ReasoningStep,
    ChainVerificationResult,
    BatchChainResult,
    verify_reasoning_chain,
    aggregate_chain_results,
    extract_reasoning_chain_from_esbg,
)
from etg_rlm.self_check import (
    SelfCheckConfig,
    SelfCheckMethod,
    SelfCheckResult,
    BatchSelfCheckResult,
    self_check_claims,
    run_self_check_pipeline,
    aggregate_self_check_results,
)
from etg_rlm.benchmark_runner import (
    ModelType,
    BenchmarkDataset,
    BenchmarkInstance,
    BenchmarkReport,
    DatasetResults,
    run_benchmark,
    aggregate_dataset_results,
)
from etg_rlm.reporting import (
    generate_markdown_report,
    generate_latex_table,
    generate_json_report,
    build_factscore_bar_chart,
    build_citation_heatmap,
    build_scaling_line_chart,
    BarChartSpec,
    LineChartSpec,
    HeatmapSpec,
)

__version__ = "0.1.0"

__all__ = [
    # Core (Section 4.1-4.2, Definition 1)
    "EvidenceSpan",
    "AtomicClaim",
    "ClaimStatus",
    "ClaimType",
    "ESBGNode",
    "EvidenceScopedBeliefGraph",
    # Verification (Section 4.3, Definitions 2-3)
    "VerificationView",
    "ViewResult",
    "MultiViewVerifier",
    # Type System (Section 4.4, Definition 4)
    "TypeThresholds",
    "EvidenceTypeChecker",
    # Policy (Section 4.5)
    "ActionType",
    "PolicyAction",
    "RecursionPolicy",
    "UtilityWeightedPolicy",
    # Bounds (Section 6, Propositions 1-3)
    "hallucination_upper_bound",
    "optimal_view_allocation",
    "check_zero_confabulation",
    "inference_time_scaling_law",
    # Algorithm (Section 5, Definition 5)
    "ebrg",
    "constrained_decode",
    "EBRGResult",
    "ConstrainedDecodingResult",
    # Pipeline (end-to-end)
    "ETGPipeline",
    "ETGConfig",
    # Metrics (eval plan Section 3)
    "compute_faithfulness",
    "rouge_l",
    "aggregate_metrics",
    "FaithfulnessMetrics",
    "ROUGELScore",
    "LatencyTracker",
    # Baselines (eval plan Section 2)
    "BaselineType",
    "BaselineConfig",
    "StandardLLMBaseline",
    "StandardRAGBaseline",
    "RAGVerifierBaseline",
    "SelfCritiqueBaseline",
    # Evaluation harness
    "EvalInstance",
    "evaluate_instance",
    "build_report",
    "build_comparative_report",
    "check_kpis",
    "ComparativeReport",
    # View factory (eval plan Section 1)
    "ViewType",
    "ViewConfig",
    "create_view",
    "create_default_view_suite",
    # Datasets (experimental design Section 1)
    "DatasetName",
    "DatasetConfig",
    "TaskType",
    "ALL_DATASET_CONFIGS",
    "get_dataset_config",
    "total_eval_instances",
    # Human evaluation (experimental design Section 2.2)
    "FaithfulnessRating",
    "FaithfulnessAnnotation",
    "PairwiseAnnotation",
    "PreferenceDimension",
    "PreferenceChoice",
    "aggregate_faithfulness",
    "aggregate_preferences",
    "fleiss_kappa",
    "HumanEvalSummary",
    # Ablation studies (experimental design Section 3)
    "AblationType",
    "AblationConfig",
    "RandomPolicy",
    "all_ablation_configs",
    "make_no_multi_view_config",
    "make_no_constraint_config",
    "make_threshold_sweep_configs",
    "make_policy_ablation_config",
    # Statistical analysis (experimental design Section 4)
    "paired_t_test",
    "cohens_d",
    "bootstrap_ci",
    "bootstrap_paired_ci",
    "full_analysis",
    "PairedTTestResult",
    "EffectSizeResult",
    "BootstrapCIResult",
    "StatisticalAnalysis",
    # FactScore (Min et al., EMNLP 2023)
    "FactScoreResult",
    "BatchFactScoreResult",
    "ClaimScoreResult",
    "compute_factscore",
    "aggregate_factscores",
    "compute_factscore_with_retrieval",
    # Citation metrics (Gao et al., ACL 2023; Rashkin et al., ACL 2022)
    "Citation",
    "CitationMetricsResult",
    "BatchCitationResult",
    "compute_citation_metrics",
    "aggregate_citation_metrics",
    "extract_citations_from_esbg",
    # Logic-step verification (multi-hop reasoning)
    "StepValidity",
    "ReasoningStep",
    "ChainVerificationResult",
    "BatchChainResult",
    "verify_reasoning_chain",
    "aggregate_chain_results",
    "extract_reasoning_chain_from_esbg",
    # Self-CheckGPT (Manakul et al., EMNLP 2023)
    "SelfCheckConfig",
    "SelfCheckMethod",
    "SelfCheckResult",
    "BatchSelfCheckResult",
    "self_check_claims",
    "run_self_check_pipeline",
    "aggregate_self_check_results",
    # Canonical benchmark runner
    "ModelType",
    "BenchmarkDataset",
    "BenchmarkInstance",
    "BenchmarkReport",
    "DatasetResults",
    "run_benchmark",
    "aggregate_dataset_results",
    # Reporting and visualization
    "generate_markdown_report",
    "generate_latex_table",
    "generate_json_report",
    "build_factscore_bar_chart",
    "build_citation_heatmap",
    "build_scaling_line_chart",
    "BarChartSpec",
    "LineChartSpec",
    "HeatmapSpec",
]
