"""Baseline configurations for ETG evaluation (Section 2 of eval plan).

Defines the four baselines that ETG is compared against. All baselines
use the same generator model to ensure a fair comparison of the
*framework*, not the underlying model.

Control 1: Standard LLM     -- zero-shot, no retrieval augmentation
Control 2: Standard RAG     -- generator + dense retriever (top-k)
Control 3: RAG + Verifier   -- RAG with post-hoc claim verification
Control 4: Self-Critique    -- single-view LLM self-check

Each baseline is defined as a configuration that can be instantiated
with concrete model implementations via the protocol interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, ClaimType, EvidenceSpan


class BaselineType(Enum):
    """The four baseline configurations from the evaluation plan."""

    STANDARD_LLM = "standard_llm"
    STANDARD_RAG = "standard_rag"
    RAG_VERIFIER = "rag_verifier"
    SELF_CRITIQUE = "self_critique"


@runtime_checkable
class Generator(Protocol):
    """Protocol for the generator LLM (e.g., Llama 3 70B)."""

    def generate(self, query: str, context: str | None = None) -> str:
        """Generate an answer to the query, optionally with retrieved context."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for dense/sparse retrieval (e.g., FAISS, BM25)."""

    def retrieve(self, query: str, corpus_id: str, top_k: int = 5) -> list[EvidenceSpan]:
        """Retrieve top-k evidence spans from the corpus."""
        ...


@runtime_checkable
class PostHocVerifier(Protocol):
    """Protocol for post-hoc claim verification (used in RAG+Verifier baseline)."""

    def verify_claims(
        self, claims: list[AtomicClaim], context: list[EvidenceSpan]
    ) -> list[tuple[AtomicClaim, ClaimStatus]]:
        """Verify each claim against the retrieved context."""
        ...


@runtime_checkable
class SelfCritiquer(Protocol):
    """Protocol for LLM self-critique (used in Self-Critique baseline)."""

    def critique(self, query: str, answer: str) -> str:
        """Ask the LLM to critique and revise its own answer."""
        ...


@dataclass
class BaselineConfig:
    """Configuration for a baseline run.

    Attributes:
        baseline_type: which baseline to run
        name: human-readable name for reporting
        top_k: number of retrieved passages (for RAG baselines)
        corpus_id: evidence corpus identifier
    """

    baseline_type: BaselineType
    name: str
    top_k: int = 5
    corpus_id: str = "default"


@dataclass
class BaselineResult:
    """Result of running a baseline.

    Attributes:
        config: the baseline configuration used
        query: the input query
        generated_text: the raw generated answer
        final_text: the answer after any post-hoc filtering
        claims: extracted atomic claims
        supported_claims: claims deemed supported (if verification was done)
        rejected_claims: claims deemed unsupported (if verification was done)
        retrieved_spans: evidence spans retrieved (if RAG was used)
    """

    config: BaselineConfig
    query: str
    generated_text: str
    final_text: str
    claims: list[AtomicClaim] = field(default_factory=list)
    supported_claims: list[AtomicClaim] = field(default_factory=list)
    rejected_claims: list[AtomicClaim] = field(default_factory=list)
    retrieved_spans: list[EvidenceSpan] = field(default_factory=list)


class BaselineRunner(ABC):
    """Abstract base for running a baseline configuration."""

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, query: str) -> BaselineResult:
        """Run the baseline on a query and return the result."""
        ...


class StandardLLMBaseline(BaselineRunner):
    """Control 1: Standard LLM (zero-shot, no retrieval).

    The base generator model with no retrieval augmentation.
    Establishes the base rate of hallucination.
    """

    def __init__(self, generator: Generator, config: BaselineConfig | None = None) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.STANDARD_LLM,
            name="Standard LLM (zero-shot)",
        ))
        self.generator = generator

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=text,
        )


class StandardRAGBaseline(BaselineRunner):
    """Control 2: Standard RAG (generator + dense retriever).

    The generator augmented with a simple dense retriever (top-k results).
    Represents the current industry standard for reducing hallucinations.
    """

    def __init__(
        self,
        generator: Generator,
        retriever: Retriever,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.STANDARD_RAG,
            name="Standard RAG",
        ))
        self.generator = generator
        self.retriever = retriever

    def run(self, query: str) -> BaselineResult:
        spans = self.retriever.retrieve(query, self.config.corpus_id, self.config.top_k)
        context = "\n".join(s.text for s in spans if s.text)
        text = self.generator.generate(query, context=context)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=text,
            retrieved_spans=spans,
        )


class RAGVerifierBaseline(BaselineRunner):
    """Control 3: RAG + Post-hoc Verifier.

    A standard RAG system where a verifier flags or retracts unsupported
    claims *after* the full text has been generated. Tests whether ETG's
    preventative constrained decoding is more effective than a corrective
    post-hoc check.
    """

    def __init__(
        self,
        generator: Generator,
        retriever: Retriever,
        claim_extractor: object,  # ClaimExtractor protocol
        verifier: PostHocVerifier,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.RAG_VERIFIER,
            name="RAG + Verifier",
        ))
        self.generator = generator
        self.retriever = retriever
        self.claim_extractor = claim_extractor
        self.verifier = verifier

    def run(self, query: str) -> BaselineResult:
        spans = self.retriever.retrieve(query, self.config.corpus_id, self.config.top_k)
        context = "\n".join(s.text for s in spans if s.text)
        text = self.generator.generate(query, context=context)

        # Extract claims then verify post-hoc
        claims = self.claim_extractor.extract(text)  # type: ignore[attr-defined]
        verdicts = self.verifier.verify_claims(claims, spans)

        supported = [c for c, s in verdicts if s == ClaimStatus.ENTAILED]
        rejected = [c for c, s in verdicts if s != ClaimStatus.ENTAILED]

        # Reconstruct text from supported claims only
        final = " ".join(c.text for c in supported) if supported else ""

        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=final,
            claims=claims,
            supported_claims=supported,
            rejected_claims=rejected,
            retrieved_spans=spans,
        )


class SelfCritiqueBaseline(BaselineRunner):
    """Control 4: Self-Critique (single-view LLM self-check).

    A single-view verification where the LLM is prompted to check its
    own claims. Tests whether ETG's multi-view, structurally enforced
    approach is more robust than behavioral prompting.
    """

    def __init__(
        self,
        generator: Generator,
        critiquer: SelfCritiquer,
        config: BaselineConfig | None = None,
    ) -> None:
        super().__init__(config or BaselineConfig(
            baseline_type=BaselineType.SELF_CRITIQUE,
            name="Self-Critique",
        ))
        self.generator = generator
        self.critiquer = critiquer

    def run(self, query: str) -> BaselineResult:
        text = self.generator.generate(query)
        revised = self.critiquer.critique(query, text)
        return BaselineResult(
            config=self.config,
            query=query,
            generated_text=text,
            final_text=revised,
        )


# ---------------------------------------------------------------------------
# Convenience: list all baseline configs
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = [
    BaselineConfig(
        baseline_type=BaselineType.STANDARD_LLM,
        name="Control 1: Standard LLM (zero-shot)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.STANDARD_RAG,
        name="Control 2: Standard RAG (top-k retrieval)",
    ),
    BaselineConfig(
        baseline_type=BaselineType.RAG_VERIFIER,
        name="Control 3: RAG + Post-hoc Verifier",
    ),
    BaselineConfig(
        baseline_type=BaselineType.SELF_CRITIQUE,
        name="Control 4: Self-Critique (single-view)",
    ),
]
