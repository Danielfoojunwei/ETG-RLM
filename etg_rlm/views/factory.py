"""Factory for constructing diverse verification views.

The implementation plan specifies N=5 diverse views:

    View 1: Dense retrieval (Sentence-BERT) + standard chunking (512 tokens)
    View 2: Sparse retrieval (BM25) + standard chunking (512 tokens)
    View 3: Dense retrieval + fine-grained chunking (128 tokens)
    View 4: Dense retrieval + query rewriting (T5-base paraphrase)
    View 5: Dense retrieval + negative sampling window (adversarial context)

Each view implements the VerificationView protocol from Definition 2:
    V_i : (E, c) -> (z_i, S_i)

Views differ by query rewriting, chunk boundaries, retriever randomness,
verifier prompting, and negative sampling windows to ensure that
support mass m(c) captures true stability of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan
from etg_rlm.verification import (
    EntailmentVerifier,
    EvidenceRetriever,
    QueryRewriter,
    VerificationView,
    ViewResult,
)


class ViewType(Enum):
    """The five view types from the implementation plan."""

    DENSE_STANDARD = "dense_standard"
    SPARSE_BM25 = "sparse_bm25"
    DENSE_FINE_CHUNK = "dense_fine_chunk"
    DENSE_QUERY_REWRITE = "dense_query_rewrite"
    DENSE_NEGATIVE_SAMPLE = "dense_negative_sample"


@dataclass(frozen=True)
class ViewConfig:
    """Configuration for constructing a verification view.

    Attributes:
        view_type: which view variant to build
        view_id: unique identifier for this view instance
        chunk_size: token count per chunk (default 512)
        top_k: number of retrieved spans
        retriever_seed: random seed for retriever (if applicable)
    """

    view_type: ViewType
    view_id: str
    chunk_size: int = 512
    top_k: int = 5
    retriever_seed: int = 42


# ---------------------------------------------------------------------------
# Concrete view implementations
# ---------------------------------------------------------------------------


class DenseStandardView(VerificationView):
    """View 1: Dense retrieval (Sentence-BERT) + standard chunking (512 tokens).

    The baseline view that uses a dense embedding model for retrieval
    with standard 512-token chunks. Provides robust coverage.
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        top_k: int = 5,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.top_k = top_k

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        spans = self.retriever.retrieve(claim.text, corpus_id)
        result = self.verifier.verify(claim, spans[:self.top_k])
        result.view_id = self.view_id
        return result


class SparseBM25View(VerificationView):
    """View 2: Sparse retrieval (BM25) + standard chunking.

    Uses traditional keyword-based BM25 retrieval. Catches claims where
    dense embeddings miss lexical matches (complementary to dense views).
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        top_k: int = 5,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.top_k = top_k

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        spans = self.retriever.retrieve(claim.text, corpus_id)
        result = self.verifier.verify(claim, spans[:self.top_k])
        result.view_id = self.view_id
        return result


class DenseFineChunkView(VerificationView):
    """View 3: Dense retrieval + fine-grained chunking (128 tokens).

    Smaller chunks improve precision for narrow factual claims that
    may be diluted in larger 512-token windows.
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        top_k: int = 5,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.top_k = top_k

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        spans = self.retriever.retrieve(claim.text, corpus_id)
        result = self.verifier.verify(claim, spans[:self.top_k])
        result.view_id = self.view_id
        return result


class DenseQueryRewriteView(VerificationView):
    """View 4: Dense retrieval + query rewriting (T5-base paraphrase).

    Paraphrases the claim before retrieval to overcome lexical gaps
    between the claim's phrasing and the source document's phrasing.
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        rewriter: QueryRewriter,
        top_k: int = 5,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.rewriter = rewriter
        self.top_k = top_k

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        rewritten = self.rewriter.rewrite(claim.text)
        spans = self.retriever.retrieve(rewritten, corpus_id)
        result = self.verifier.verify(claim, spans[:self.top_k])
        result.view_id = self.view_id
        return result


class DenseNegativeSampleView(VerificationView):
    """View 5: Dense retrieval + negative sampling window.

    Includes deliberately non-relevant passages in the verification
    context as adversarial distractors. Tests whether the verifier
    correctly rejects entailment when surrounded by noise.
    """

    def __init__(
        self,
        view_id: str,
        retriever: EvidenceRetriever,
        verifier: EntailmentVerifier,
        negative_retriever: EvidenceRetriever,
        top_k: int = 5,
        n_negatives: int = 3,
    ) -> None:
        super().__init__(view_id)
        self.retriever = retriever
        self.verifier = verifier
        self.negative_retriever = negative_retriever
        self.top_k = top_k
        self.n_negatives = n_negatives

    def verify(self, claim: AtomicClaim, corpus_id: str) -> ViewResult:
        positive_spans = self.retriever.retrieve(claim.text, corpus_id)
        # Retrieve negatives using a reversed/unrelated query
        negative_query = " ".join(reversed(claim.text.split()))
        negative_spans = self.negative_retriever.retrieve(negative_query, corpus_id)

        # Mix positives and negatives
        mixed = positive_spans[:self.top_k] + negative_spans[:self.n_negatives]
        result = self.verifier.verify(claim, mixed)
        result.view_id = self.view_id
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_view(
    config: ViewConfig,
    retriever: EvidenceRetriever,
    verifier: EntailmentVerifier,
    sparse_retriever: EvidenceRetriever | None = None,
    rewriter: QueryRewriter | None = None,
    negative_retriever: EvidenceRetriever | None = None,
) -> VerificationView:
    """Create a verification view from a configuration.

    Args:
        config: the view configuration
        retriever: primary dense retriever
        verifier: NLI entailment verifier
        sparse_retriever: BM25 retriever (for SPARSE_BM25 view)
        rewriter: query rewriter (for DENSE_QUERY_REWRITE view)
        negative_retriever: retriever for adversarial negatives

    Returns:
        A configured VerificationView instance.
    """
    if config.view_type == ViewType.DENSE_STANDARD:
        return DenseStandardView(config.view_id, retriever, verifier, config.top_k)

    elif config.view_type == ViewType.SPARSE_BM25:
        r = sparse_retriever or retriever
        return SparseBM25View(config.view_id, r, verifier, config.top_k)

    elif config.view_type == ViewType.DENSE_FINE_CHUNK:
        return DenseFineChunkView(config.view_id, retriever, verifier, config.top_k)

    elif config.view_type == ViewType.DENSE_QUERY_REWRITE:
        if rewriter is None:
            raise ValueError("Query rewriter required for DENSE_QUERY_REWRITE view")
        return DenseQueryRewriteView(config.view_id, retriever, verifier, rewriter, config.top_k)

    elif config.view_type == ViewType.DENSE_NEGATIVE_SAMPLE:
        nr = negative_retriever or retriever
        return DenseNegativeSampleView(
            config.view_id, retriever, verifier, nr, config.top_k
        )

    else:
        raise ValueError(f"Unknown view type: {config.view_type}")


def create_default_view_suite(
    retriever: EvidenceRetriever,
    verifier: EntailmentVerifier,
    sparse_retriever: EvidenceRetriever | None = None,
    rewriter: QueryRewriter | None = None,
    negative_retriever: EvidenceRetriever | None = None,
) -> list[VerificationView]:
    """Create the default N=5 view suite from the implementation plan.

    Returns views in this order:
        1. Dense + standard chunking (512)
        2. Sparse BM25 + standard chunking (512)
        3. Dense + fine-grained chunking (128)
        4. Dense + query rewriting
        5. Dense + negative sampling

    Views 4 and 5 are only included if the required components
    (rewriter, negative_retriever) are provided.
    """
    views: list[VerificationView] = []

    # View 1: Dense + standard
    views.append(DenseStandardView("v1_dense_512", retriever, verifier))

    # View 2: Sparse BM25
    r = sparse_retriever or retriever
    views.append(SparseBM25View("v2_bm25_512", r, verifier))

    # View 3: Dense + fine chunking
    views.append(DenseFineChunkView("v3_dense_128", retriever, verifier))

    # View 4: Dense + query rewrite (if rewriter available)
    if rewriter is not None:
        views.append(DenseQueryRewriteView(
            "v4_dense_rewrite", retriever, verifier, rewriter
        ))

    # View 5: Dense + negative sampling
    nr = negative_retriever or retriever
    views.append(DenseNegativeSampleView(
        "v5_dense_negative", retriever, verifier, nr
    ))

    return views
