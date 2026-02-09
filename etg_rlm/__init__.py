"""Evidence-Typed Generation for Recursive Language Models.

A formal framework that treats hallucination control as static type checking
over an Evidence-Scoped Belief Graph (ESBG). Claims are typed by their
multi-view support mass, and only well-typed (evidence-entailed) claims
are rendered into the final output.
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
)
from etg_rlm.pipeline import (
    ETGPipeline,
    ETGConfig,
)

__version__ = "0.1.0"

__all__ = [
    "EvidenceSpan",
    "AtomicClaim",
    "ClaimStatus",
    "ClaimType",
    "ESBGNode",
    "EvidenceScopedBeliefGraph",
    "VerificationView",
    "ViewResult",
    "MultiViewVerifier",
    "TypeThresholds",
    "EvidenceTypeChecker",
    "ActionType",
    "PolicyAction",
    "RecursionPolicy",
    "UtilityWeightedPolicy",
    "hallucination_upper_bound",
    "optimal_view_allocation",
    "ETGPipeline",
    "ETGConfig",
]
