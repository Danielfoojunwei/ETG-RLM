"""Tests for logic-step verification of multi-hop reasoning chains."""

import pytest

from etg_rlm.core import AtomicClaim, ClaimStatus, ESBGNode, EvidenceSpan
from etg_rlm.logic_verification import (
    StepValidity,
    ReasoningStep,
    StepVerificationResult,
    ChainVerificationResult,
    verify_reasoning_chain,
    aggregate_chain_results,
    extract_reasoning_chain_from_esbg,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class AlwaysValidVerifier:
    """Stub: all steps are valid."""

    def verify_step(
        self, step: ReasoningStep, premise_claims: list[AtomicClaim]
    ) -> StepVerificationResult:
        return StepVerificationResult(
            step_id=step.step_id,
            validity=StepValidity.VALID,
            confidence=1.0,
        )


class AlwaysInvalidVerifier:
    """Stub: all steps are invalid."""

    def verify_step(
        self, step: ReasoningStep, premise_claims: list[AtomicClaim]
    ) -> StepVerificationResult:
        return StepVerificationResult(
            step_id=step.step_id,
            validity=StepValidity.INVALID,
            confidence=0.0,
        )


class EvidenceBasedVerifier:
    """Stub: valid if step has evidence, unsupported otherwise."""

    def verify_step(
        self, step: ReasoningStep, premise_claims: list[AtomicClaim]
    ) -> StepVerificationResult:
        if step.evidence_spans:
            return StepVerificationResult(
                step_id=step.step_id, validity=StepValidity.VALID, confidence=0.9
            )
        return StepVerificationResult(
            step_id=step.step_id, validity=StepValidity.UNSUPPORTED, confidence=0.1
        )


class AlwaysCoherent:
    """Stub: always coherent."""

    def check_coherence(self, claims: list[AtomicClaim]) -> bool:
        return True


class NeverCoherent:
    """Stub: never coherent (contradictions found)."""

    def check_coherence(self, claims: list[AtomicClaim]) -> bool:
        return False


# ---------------------------------------------------------------------------
# Test verify_reasoning_chain
# ---------------------------------------------------------------------------


class TestVerifyReasoningChain:
    def test_empty_chain(self):
        result = verify_reasoning_chain([], AlwaysValidVerifier())
        assert result.chain_valid is True
        assert result.n_steps == 0
        assert result.step_accuracy == 1.0

    def test_all_valid_chain(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Premise"),
                evidence_spans=(EvidenceSpan(doc_id="d1", start=0, end=50),),
            ),
            ReasoningStep(
                step_id="s2",
                claim=AtomicClaim(claim_id="c2", text="Conclusion"),
                premises=("s1",),
            ),
        ]
        result = verify_reasoning_chain(steps, AlwaysValidVerifier())
        assert result.chain_valid is True
        assert result.step_accuracy == 1.0
        assert result.n_valid == 2
        assert result.n_invalid == 0

    def test_all_invalid_chain(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Bad step"),
            ),
        ]
        result = verify_reasoning_chain(steps, AlwaysInvalidVerifier())
        assert result.chain_valid is False
        assert result.step_accuracy == 0.0
        assert result.n_invalid == 1

    def test_mixed_chain(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Grounded"),
                evidence_spans=(EvidenceSpan(doc_id="d1", start=0, end=50),),
            ),
            ReasoningStep(
                step_id="s2",
                claim=AtomicClaim(claim_id="c2", text="Ungrounded"),
                premises=("s1",),
            ),
        ]
        result = verify_reasoning_chain(steps, EvidenceBasedVerifier())
        assert result.n_valid == 1
        assert result.n_unsupported == 1
        assert result.step_accuracy == pytest.approx(0.5)
        assert result.chain_valid is False

    def test_multi_hop_chain(self):
        """Test a 3-hop reasoning chain: s1 -> s2 -> s3."""
        span = EvidenceSpan(doc_id="d1", start=0, end=50)
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Paris is in France"),
                evidence_spans=(span,),
            ),
            ReasoningStep(
                step_id="s2",
                claim=AtomicClaim(claim_id="c2", text="France is in Europe"),
                premises=("s1",),
                evidence_spans=(span,),
            ),
            ReasoningStep(
                step_id="s3",
                claim=AtomicClaim(claim_id="c3", text="Paris is in Europe"),
                premises=("s1", "s2"),
                evidence_spans=(span,),
            ),
        ]
        result = verify_reasoning_chain(steps, AlwaysValidVerifier())
        assert result.chain_valid is True
        assert result.n_steps == 3
        assert result.n_valid == 3

    def test_coherence_check(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="A fact"),
            ),
        ]
        # Valid steps but incoherent
        result = verify_reasoning_chain(
            steps, AlwaysValidVerifier(), NeverCoherent()
        )
        assert result.chain_coherent is False

    def test_coherence_passes(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="A fact"),
            ),
        ]
        result = verify_reasoning_chain(
            steps, AlwaysValidVerifier(), AlwaysCoherent()
        )
        assert result.chain_coherent is True

    def test_per_step_results(self):
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Fact"),
                evidence_spans=(EvidenceSpan(doc_id="d1", start=0, end=50),),
            ),
            ReasoningStep(
                step_id="s2",
                claim=AtomicClaim(claim_id="c2", text="Ungrounded"),
            ),
        ]
        result = verify_reasoning_chain(steps, EvidenceBasedVerifier())
        assert len(result.per_step) == 2
        assert result.per_step[0].validity == StepValidity.VALID
        assert result.per_step[1].validity == StepValidity.UNSUPPORTED

    def test_premise_resolution(self):
        """Verify that premises are correctly resolved from prior steps."""
        steps = [
            ReasoningStep(
                step_id="s1",
                claim=AtomicClaim(claim_id="c1", text="Base"),
            ),
            ReasoningStep(
                step_id="s2",
                claim=AtomicClaim(claim_id="c2", text="Derived"),
                premises=("s1",),
            ),
        ]

        class PremiseCheckVerifier:
            def verify_step(self, step, premise_claims):
                if step.step_id == "s2":
                    assert len(premise_claims) == 1
                    assert premise_claims[0].text == "Base"
                return StepVerificationResult(
                    step_id=step.step_id, validity=StepValidity.VALID
                )

        result = verify_reasoning_chain(steps, PremiseCheckVerifier())
        assert result.chain_valid is True


# ---------------------------------------------------------------------------
# Test aggregate_chain_results
# ---------------------------------------------------------------------------


class TestAggregateChainResults:
    def test_empty(self):
        result = aggregate_chain_results([])
        assert result.n_chains == 0

    def test_multiple_chains(self):
        results = [
            ChainVerificationResult(
                chain_valid=True, step_accuracy=1.0,
                n_steps=3, n_valid=3, n_invalid=0,
                n_unsupported=0, n_redundant=0, chain_coherent=True,
            ),
            ChainVerificationResult(
                chain_valid=False, step_accuracy=0.5,
                n_steps=4, n_valid=2, n_invalid=2,
                n_unsupported=0, n_redundant=0, chain_coherent=False,
            ),
        ]
        agg = aggregate_chain_results(results)
        assert agg.mean_step_accuracy == pytest.approx(0.75)
        assert agg.chain_validity_rate == pytest.approx(0.5)
        assert agg.mean_chain_coherence == pytest.approx(0.5)
        assert agg.total_steps == 7
        assert agg.total_valid_steps == 5


# ---------------------------------------------------------------------------
# Test extract_reasoning_chain_from_esbg
# ---------------------------------------------------------------------------


class TestExtractFromESBG:
    def test_linear_chain(self):
        """Test extraction from a simple linear DAG: n1 -> n2 -> n3."""
        nodes = {
            "n1": ESBGNode(
                node_id="n1",
                claim=AtomicClaim(claim_id="c1", text="Step 1"),
                evidence_spans={EvidenceSpan(doc_id="d1", start=0, end=50)},
            ),
            "n2": ESBGNode(
                node_id="n2",
                claim=AtomicClaim(claim_id="c2", text="Step 2"),
            ),
            "n3": ESBGNode(
                node_id="n3",
                claim=AtomicClaim(claim_id="c3", text="Step 3"),
            ),
        }
        edges = [("n1", "n2"), ("n2", "n3")]
        steps = extract_reasoning_chain_from_esbg(nodes, edges)

        assert len(steps) == 3
        assert steps[0].step_id == "n1"
        assert steps[1].step_id == "n2"
        assert steps[2].step_id == "n3"
        assert steps[1].premises == ("n1",)
        assert steps[2].premises == ("n2",)

    def test_empty_graph(self):
        steps = extract_reasoning_chain_from_esbg({}, [])
        assert len(steps) == 0

    def test_independent_nodes(self):
        nodes = {
            "n1": ESBGNode(
                node_id="n1",
                claim=AtomicClaim(claim_id="c1", text="Fact 1"),
            ),
            "n2": ESBGNode(
                node_id="n2",
                claim=AtomicClaim(claim_id="c2", text="Fact 2"),
            ),
        }
        steps = extract_reasoning_chain_from_esbg(nodes, [])
        assert len(steps) == 2
        assert all(s.premises == () for s in steps)

    def test_diamond_dag(self):
        """Test extraction from a diamond DAG: n1 -> n2, n1 -> n3, n2 -> n4, n3 -> n4."""
        nodes = {
            f"n{i}": ESBGNode(
                node_id=f"n{i}",
                claim=AtomicClaim(claim_id=f"c{i}", text=f"Claim {i}"),
            )
            for i in range(1, 5)
        }
        edges = [("n1", "n2"), ("n1", "n3"), ("n2", "n4"), ("n3", "n4")]
        steps = extract_reasoning_chain_from_esbg(nodes, edges)

        assert len(steps) == 4
        # n4 should come last and have premises from n2 and n3
        assert steps[-1].step_id == "n4"
        assert set(steps[-1].premises) == {"n2", "n3"}
