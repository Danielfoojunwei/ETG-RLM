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
]
