"""Tests for Self-CheckGPT hallucination detection."""

import pytest

from etg_rlm.core import AtomicClaim
from etg_rlm.self_check import (
    SelfCheckConfig,
    SelfCheckMethod,
    SelfCheckResult,
    ClaimConsistencyResult,
    self_check_claims,
    run_self_check_pipeline,
    aggregate_self_check_results,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class HighConsistencyChecker:
    """Stub: all claims are highly consistent with all samples."""

    def check_consistency(self, claim: AtomicClaim, sample: str) -> float:
        return 0.9


class LowConsistencyChecker:
    """Stub: all claims are inconsistent with all samples."""

    def check_consistency(self, claim: AtomicClaim, sample: str) -> float:
        return 0.1


class KeywordConsistencyChecker:
    """Stub: consistent if claim text appears in sample."""

    def check_consistency(self, claim: AtomicClaim, sample: str) -> float:
        return 0.9 if claim.text.lower() in sample.lower() else 0.1


class StubGenerator:
    """Stub: generates fixed samples."""

    def generate_samples(
        self, query: str, n_samples: int, temperature: float = 1.0
    ) -> list[str]:
        return [f"Sample {i} about Paris" for i in range(n_samples)]


class StubDecomposer:
    """Stub: returns fixed claims."""

    def decompose(self, text: str) -> list[AtomicClaim]:
        return [
            AtomicClaim(claim_id="c1", text="Paris"),
            AtomicClaim(claim_id="c2", text="Hallucinated fact"),
        ]


# ---------------------------------------------------------------------------
# Test self_check_claims
# ---------------------------------------------------------------------------


class TestSelfCheckClaims:
    def test_empty_claims(self):
        result = self_check_claims([], ["sample"], HighConsistencyChecker())
        assert result.hallucination_rate == 0.0
        assert result.mean_consistency == 1.0
        assert result.n_claims == 0

    def test_all_consistent(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Fact 1"),
            AtomicClaim(claim_id="c2", text="Fact 2"),
        ]
        samples = ["Sample 1", "Sample 2", "Sample 3"]
        result = self_check_claims(claims, samples, HighConsistencyChecker())
        assert result.hallucination_rate == 0.0
        assert result.n_hallucinated == 0
        assert result.n_consistent == 2
        assert result.mean_consistency == pytest.approx(0.9)

    def test_all_inconsistent(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Bad fact 1"),
            AtomicClaim(claim_id="c2", text="Bad fact 2"),
        ]
        samples = ["Sample 1", "Sample 2"]
        result = self_check_claims(claims, samples, LowConsistencyChecker())
        assert result.hallucination_rate == 1.0
        assert result.n_hallucinated == 2
        assert result.n_consistent == 0

    def test_mixed_consistency(self):
        claims = [
            AtomicClaim(claim_id="c1", text="Paris"),
            AtomicClaim(claim_id="c2", text="Hallucinated"),
        ]
        samples = ["Paris is the capital of France", "Paris has the Eiffel Tower"]
        result = self_check_claims(claims, samples, KeywordConsistencyChecker())
        assert result.n_consistent == 1  # "Paris" is consistent
        assert result.n_hallucinated == 1  # "Hallucinated" is not

    def test_custom_threshold(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact")]
        samples = ["Sample"]
        # HighConsistencyChecker returns 0.9
        result_high = self_check_claims(claims, samples, HighConsistencyChecker(), threshold=0.95)
        assert result_high.n_hallucinated == 1  # 0.9 < 0.95

        result_low = self_check_claims(claims, samples, HighConsistencyChecker(), threshold=0.5)
        assert result_low.n_hallucinated == 0  # 0.9 >= 0.5

    def test_per_sample_scores_recorded(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact")]
        samples = ["S1", "S2", "S3"]
        result = self_check_claims(claims, samples, HighConsistencyChecker())
        assert len(result.per_claim) == 1
        assert len(result.per_claim[0].per_sample_scores) == 3
        assert all(s == pytest.approx(0.9) for s in result.per_claim[0].per_sample_scores)

    def test_n_samples_recorded(self):
        claims = [AtomicClaim(claim_id="c1", text="Fact")]
        samples = ["S1", "S2", "S3", "S4", "S5"]
        result = self_check_claims(claims, samples, HighConsistencyChecker())
        assert result.n_samples_used == 5


# ---------------------------------------------------------------------------
# Test run_self_check_pipeline
# ---------------------------------------------------------------------------


class TestSelfCheckPipeline:
    def test_full_pipeline(self):
        result = run_self_check_pipeline(
            query="What is Paris?",
            primary_response="Paris is the capital of France",
            generator=StubGenerator(),
            decomposer=StubDecomposer(),
            checker=KeywordConsistencyChecker(),
            config=SelfCheckConfig(n_samples=3),
        )
        assert result.n_claims == 2
        assert result.n_samples_used == 3
        # "Paris" should be consistent (in "Sample X about Paris")
        # "Hallucinated fact" should be inconsistent
        assert result.n_consistent == 1
        assert result.n_hallucinated == 1

    def test_pipeline_with_default_config(self):
        result = run_self_check_pipeline(
            query="Query",
            primary_response="Some response",
            generator=StubGenerator(),
            decomposer=StubDecomposer(),
            checker=HighConsistencyChecker(),
        )
        # Default config: n_samples=5
        assert result.n_samples_used == 5
        assert result.n_claims == 2


# ---------------------------------------------------------------------------
# Test aggregate_self_check_results
# ---------------------------------------------------------------------------


class TestAggregateSelfCheck:
    def test_empty(self):
        result = aggregate_self_check_results([])
        assert result.n_instances == 0

    def test_multiple_instances(self):
        results = [
            SelfCheckResult(
                hallucination_rate=0.2, mean_consistency=0.8,
                n_claims=5, n_hallucinated=1, n_consistent=4, n_samples_used=3,
            ),
            SelfCheckResult(
                hallucination_rate=0.4, mean_consistency=0.6,
                n_claims=5, n_hallucinated=2, n_consistent=3, n_samples_used=3,
            ),
        ]
        agg = aggregate_self_check_results(results)
        assert agg.mean_hallucination_rate == pytest.approx(0.3)
        assert agg.mean_consistency == pytest.approx(0.7)
        assert agg.total_claims == 10
        assert agg.total_hallucinated == 3


# ---------------------------------------------------------------------------
# Test SelfCheckConfig
# ---------------------------------------------------------------------------


class TestSelfCheckConfig:
    def test_default_config(self):
        cfg = SelfCheckConfig()
        assert cfg.n_samples == 5
        assert cfg.temperature == 1.0
        assert cfg.method == SelfCheckMethod.NLI
        assert cfg.threshold == 0.5

    def test_custom_config(self):
        cfg = SelfCheckConfig(
            n_samples=10,
            temperature=0.7,
            method=SelfCheckMethod.BERTSCORE,
            threshold=0.3,
        )
        assert cfg.n_samples == 10
        assert cfg.method == SelfCheckMethod.BERTSCORE
