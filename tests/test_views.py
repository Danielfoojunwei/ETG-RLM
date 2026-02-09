"""Tests for the diverse verification view factory."""

import pytest

from etg_rlm.core import AtomicClaim, ClaimStatus, EvidenceSpan
from etg_rlm.verification import ViewResult
from etg_rlm.views.factory import (
    ViewConfig,
    ViewType,
    create_view,
    create_default_view_suite,
    DenseStandardView,
    SparseBM25View,
    DenseFineChunkView,
    DenseQueryRewriteView,
    DenseNegativeSampleView,
)


class StubRetriever:
    def __init__(self, spans: list[EvidenceSpan] | None = None):
        self._spans = spans or [EvidenceSpan(doc_id="d1", start=0, end=10, text="evidence")]
        self.last_query: str = ""

    def retrieve(self, query: str, corpus_id: str) -> list[EvidenceSpan]:
        self.last_query = query
        return self._spans


class StubVerifier:
    def __init__(self, verdict: ClaimStatus = ClaimStatus.ENTAILED):
        self._verdict = verdict

    def verify(self, claim: AtomicClaim, spans: list[EvidenceSpan]) -> ViewResult:
        return ViewResult(
            verdict=self._verdict,
            spans=set(spans),
        )


class StubRewriter:
    def rewrite(self, query: str) -> str:
        return f"rewritten: {query}"


class TestDenseStandardView:
    def test_basic_verification(self):
        view = DenseStandardView("v1", StubRetriever(), StubVerifier())
        claim = AtomicClaim(claim_id="c1", text="Test")
        result = view.verify(claim, "corpus")
        assert result.verdict == ClaimStatus.ENTAILED
        assert result.view_id == "v1"


class TestSparseBM25View:
    def test_basic_verification(self):
        view = SparseBM25View("v2", StubRetriever(), StubVerifier())
        claim = AtomicClaim(claim_id="c1", text="Test")
        result = view.verify(claim, "corpus")
        assert result.verdict == ClaimStatus.ENTAILED


class TestDenseFineChunkView:
    def test_basic_verification(self):
        view = DenseFineChunkView("v3", StubRetriever(), StubVerifier())
        result = view.verify(AtomicClaim(claim_id="c1", text="Test"), "corpus")
        assert result.verdict == ClaimStatus.ENTAILED


class TestDenseQueryRewriteView:
    def test_rewrites_query(self):
        retriever = StubRetriever()
        rewriter = StubRewriter()
        view = DenseQueryRewriteView("v4", retriever, StubVerifier(), rewriter)
        view.verify(AtomicClaim(claim_id="c1", text="original"), "corpus")
        assert retriever.last_query == "rewritten: original"


class TestDenseNegativeSampleView:
    def test_includes_negatives(self):
        pos_spans = [EvidenceSpan(doc_id="d1", start=0, end=5, text="positive")]
        neg_spans = [EvidenceSpan(doc_id="d2", start=0, end=5, text="negative")]

        view = DenseNegativeSampleView(
            "v5",
            StubRetriever(pos_spans),
            StubVerifier(),
            StubRetriever(neg_spans),
            top_k=1,
            n_negatives=1,
        )
        result = view.verify(AtomicClaim(claim_id="c1", text="Test"), "corpus")
        # Should include both positive and negative spans
        assert len(result.spans) == 2


class TestCreateView:
    def test_dense_standard(self):
        config = ViewConfig(ViewType.DENSE_STANDARD, "v1")
        view = create_view(config, StubRetriever(), StubVerifier())
        assert isinstance(view, DenseStandardView)

    def test_sparse_bm25(self):
        config = ViewConfig(ViewType.SPARSE_BM25, "v2")
        view = create_view(config, StubRetriever(), StubVerifier())
        assert isinstance(view, SparseBM25View)

    def test_fine_chunk(self):
        config = ViewConfig(ViewType.DENSE_FINE_CHUNK, "v3")
        view = create_view(config, StubRetriever(), StubVerifier())
        assert isinstance(view, DenseFineChunkView)

    def test_query_rewrite_requires_rewriter(self):
        config = ViewConfig(ViewType.DENSE_QUERY_REWRITE, "v4")
        with pytest.raises(ValueError, match="Query rewriter required"):
            create_view(config, StubRetriever(), StubVerifier())

    def test_query_rewrite_with_rewriter(self):
        config = ViewConfig(ViewType.DENSE_QUERY_REWRITE, "v4")
        view = create_view(
            config, StubRetriever(), StubVerifier(), rewriter=StubRewriter()
        )
        assert isinstance(view, DenseQueryRewriteView)

    def test_negative_sample(self):
        config = ViewConfig(ViewType.DENSE_NEGATIVE_SAMPLE, "v5")
        view = create_view(config, StubRetriever(), StubVerifier())
        assert isinstance(view, DenseNegativeSampleView)


class TestDefaultViewSuite:
    def test_creates_at_least_4_views(self):
        views = create_default_view_suite(StubRetriever(), StubVerifier())
        # Without rewriter, should get 4 views (1, 2, 3, 5)
        assert len(views) == 4

    def test_creates_5_views_with_rewriter(self):
        views = create_default_view_suite(
            StubRetriever(), StubVerifier(), rewriter=StubRewriter()
        )
        assert len(views) == 5

    def test_view_ids_unique(self):
        views = create_default_view_suite(
            StubRetriever(), StubVerifier(), rewriter=StubRewriter()
        )
        ids = [v.view_id for v in views]
        assert len(ids) == len(set(ids))
