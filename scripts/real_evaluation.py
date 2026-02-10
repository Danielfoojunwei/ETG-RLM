#!/usr/bin/env python3
"""Real empirical evaluation of ETG framework.

NO simulations. NO mocks. NO fake data.

Uses:
- Real dataset: TruthfulQA (Lin et al., ACL 2022) from HuggingFace
- Real NLI model: cross-encoder/nli-deberta-v3-small (DeBERTa v3)
- Real ETG pipeline: ESBG construction, multi-view verification, type-checking
- Real metrics: FactScore (claim-level precision), hallucination detection accuracy

Methodology:
  TruthfulQA provides questions with correct_answers and incorrect_answers.
  For each question:
    1. Treat correct answers as "faithful claims" and incorrect as "hallucinated claims"
    2. Use best_answer as evidence (ground truth)
    3. Run real NLI model to verify each claim against evidence
    4. Apply ETG multi-view verification with diverse input formulations
    5. Apply type-checking with threshold tau
    6. Measure: does ETG correctly accept true claims and reject hallucinations?

This produces real precision, recall, F1, and FactScore numbers from a real
model on a real benchmark.
"""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from etg_rlm.core import (
    AtomicClaim, EvidenceSpan, ESBGNode, EvidenceScopedBeliefGraph,
    ClaimStatus, ClaimType,
)
from etg_rlm.verification import MultiViewVerifier, VerificationView
from etg_rlm.type_system import TypeThresholds, EvidenceTypeChecker
from etg_rlm.bounds import hallucination_upper_bound, kl_bernoulli


# ---------------------------------------------------------------------------
# Real NLI Model
# ---------------------------------------------------------------------------

class RealNLIModel:
    """Wrapper around a real cross-encoder NLI model."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        print(f"Loading NLI model: {model_name} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # Label mapping for this model: 0=contradiction, 1=entailment, 2=neutral
        self.label_names = ["contradiction", "entailment", "neutral"]
        print("NLI model loaded.", flush=True)

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run NLI inference. Returns {contradiction, entailment, neutral} probs."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return {l: float(p) for l, p in zip(self.label_names, probs)}

    def batch_predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict[str, float]]:
        """Batch NLI inference for efficiency."""
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            inputs = self.tokenizer(
                premises, hypotheses,
                return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            for j in range(len(batch)):
                results.append({l: float(probs[j][k]) for k, l in enumerate(self.label_names)})
        return results


# ---------------------------------------------------------------------------
# Real Verification Views (diverse NLI formulations)
# ---------------------------------------------------------------------------

class RealNLIView(VerificationView):
    """A real verification view backed by a real NLI model.

    Different views use different input formulations to create diversity:
    - Direct: premise=evidence, hypothesis=claim
    - Reversed: premise=claim, hypothesis=evidence (different NLI behavior)
    - Contextualized: prepend question to evidence
    - Negation-aware: check if negation is contradicted
    - Truncated: use first half of evidence only
    """

    def __init__(self, nli: RealNLIModel, view_type: str, view_id: str,
                 entailment_threshold: float = 0.33):
        self._nli = nli
        self._view_type = view_type
        self._id = view_id
        self._threshold = entailment_threshold

    @property
    def view_id(self) -> str:
        return self._id

    def verify(self, evidence_text: str, claim_text: str, **kwargs) -> tuple:
        """Run real NLI verification with this view's formulation."""
        question = kwargs.get("question", "")

        if self._view_type == "direct":
            premise = evidence_text
            hypothesis = claim_text
        elif self._view_type == "contextualized":
            premise = f"Question: {question} Answer: {evidence_text}"
            hypothesis = claim_text
        elif self._view_type == "reversed":
            # Reverse premise/hypothesis — NLI models behave differently
            premise = claim_text
            hypothesis = evidence_text
        elif self._view_type == "truncated":
            # Use only first half of evidence
            words = evidence_text.split()
            premise = " ".join(words[:max(len(words) // 2, 5)])
            hypothesis = claim_text
        elif self._view_type == "paraphrased":
            # Add "It is true that" prefix to hypothesis
            premise = evidence_text
            hypothesis = f"It is true that {claim_text.lower()}"
        else:
            premise = evidence_text
            hypothesis = claim_text

        probs = self._nli.predict(premise, hypothesis)

        entailment_prob = probs["entailment"]
        contradiction_prob = probs["contradiction"]

        if entailment_prob > self._threshold:
            status = ClaimStatus.ENTAILED
        elif contradiction_prob > 0.5:
            status = ClaimStatus.CONTRADICTED
        else:
            status = ClaimStatus.UNKNOWN

        span = EvidenceSpan(doc_id="evidence", start=0, end=len(evidence_text), text=evidence_text)
        spans = {span} if status == ClaimStatus.ENTAILED else set()

        return status, spans, entailment_prob


# ---------------------------------------------------------------------------
# Real ETG Pipeline
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim through ETG."""
    claim_text: str
    is_correct: bool  # ground truth
    support_mass: float
    claim_type: ClaimType
    per_view_entailment: list[float]
    per_view_verdicts: list[bool]
    accepted_by_etg: bool


@dataclass
class InstanceResult:
    """Result of evaluating one TruthfulQA instance."""
    question: str
    evidence: str
    claim_results: list[ClaimVerificationResult]
    n_correct_claims: int
    n_incorrect_claims: int
    n_correct_accepted: int  # true positives
    n_correct_rejected: int  # false negatives
    n_incorrect_accepted: int  # false positives (hallucinations that passed)
    n_incorrect_rejected: int  # true negatives


def run_etg_on_instance(
    question: str,
    evidence: str,
    correct_claims: list[str],
    incorrect_claims: list[str],
    views: list[RealNLIView],
    tau: float,
) -> InstanceResult:
    """Run real ETG pipeline on one TruthfulQA instance."""

    all_claims = [(c, True) for c in correct_claims] + [(c, False) for c in incorrect_claims]
    claim_results = []

    for claim_text, is_correct in all_claims:
        # Run each view — real NLI inference
        verdicts = []
        entailment_probs = []
        for view in views:
            status, spans, ent_prob = view.verify(evidence, claim_text, question=question)
            verdicts.append(status == ClaimStatus.ENTAILED)
            entailment_probs.append(ent_prob)

        # Compute support mass (Definition 3)
        support_mass = sum(1 for v in verdicts if v) / len(verdicts)

        # Apply type system (Definition 4)
        if support_mass >= tau:
            claim_type = ClaimType.VERIFIED
        elif support_mass > 0:
            claim_type = ClaimType.UNCERTAIN
        else:
            claim_type = ClaimType.UNSUPPORTED

        accepted = claim_type == ClaimType.VERIFIED

        claim_results.append(ClaimVerificationResult(
            claim_text=claim_text,
            is_correct=is_correct,
            support_mass=support_mass,
            claim_type=claim_type,
            per_view_entailment=entailment_probs,
            per_view_verdicts=verdicts,
            accepted_by_etg=accepted,
        ))

    n_correct = sum(1 for c in claim_results if c.is_correct)
    n_incorrect = sum(1 for c in claim_results if not c.is_correct)
    n_correct_accepted = sum(1 for c in claim_results if c.is_correct and c.accepted_by_etg)
    n_correct_rejected = sum(1 for c in claim_results if c.is_correct and not c.accepted_by_etg)
    n_incorrect_accepted = sum(1 for c in claim_results if not c.is_correct and c.accepted_by_etg)
    n_incorrect_rejected = sum(1 for c in claim_results if not c.is_correct and not c.accepted_by_etg)

    return InstanceResult(
        question=question,
        evidence=evidence,
        claim_results=claim_results,
        n_correct_claims=n_correct,
        n_incorrect_claims=n_incorrect,
        n_correct_accepted=n_correct_accepted,
        n_correct_rejected=n_correct_rejected,
        n_incorrect_accepted=n_incorrect_accepted,
        n_incorrect_rejected=n_incorrect_rejected,
    )


# ---------------------------------------------------------------------------
# Baseline: No Verification (accept all claims)
# ---------------------------------------------------------------------------

def run_no_verification_baseline(
    correct_claims: list[str],
    incorrect_claims: list[str],
) -> dict:
    """Baseline: accept all claims without verification."""
    n_correct = len(correct_claims)
    n_incorrect = len(incorrect_claims)
    total = n_correct + n_incorrect
    return {
        "n_correct_accepted": n_correct,
        "n_incorrect_accepted": n_incorrect,
        "precision": n_correct / total if total > 0 else 0,
        "recall": 1.0,  # all correct claims accepted
        "hallucination_rate": n_incorrect / total if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Baseline: Single-View NLI (one NLI check)
# ---------------------------------------------------------------------------

def run_single_view_baseline(
    question: str,
    evidence: str,
    correct_claims: list[str],
    incorrect_claims: list[str],
    nli: RealNLIModel,
    threshold: float = 0.33,
) -> dict:
    """Baseline: single NLI check, accept if entailment > threshold."""
    n_correct_accepted = 0
    n_incorrect_accepted = 0

    for claim in correct_claims:
        probs = nli.predict(evidence, claim)
        if probs["entailment"] > threshold:
            n_correct_accepted += 1

    for claim in incorrect_claims:
        probs = nli.predict(evidence, claim)
        if probs["entailment"] > threshold:
            n_incorrect_accepted += 1

    n_correct = len(correct_claims)
    n_incorrect = len(incorrect_claims)
    total_accepted = n_correct_accepted + n_incorrect_accepted

    return {
        "n_correct_accepted": n_correct_accepted,
        "n_incorrect_accepted": n_incorrect_accepted,
        "precision": n_correct_accepted / total_accepted if total_accepted > 0 else 1.0,
        "recall": n_correct_accepted / n_correct if n_correct > 0 else 0,
        "hallucination_rate": n_incorrect_accepted / total_accepted if total_accepted > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REAL EMPIRICAL EVALUATION OF ETG FRAMEWORK")
    print("No simulations. No mocks. Real data, real models, real numbers.")
    print("=" * 70)
    print()

    # 1. Load real dataset
    print("[1/5] Loading TruthfulQA dataset from HuggingFace...")
    ds = load_dataset("truthful_qa", "generation")["validation"]
    print(f"  Loaded {len(ds)} instances")
    print(f"  Fields: {list(ds[0].keys())}")
    print()

    # 2. Load real NLI model
    print("[2/5] Loading real NLI model...")
    nli = RealNLIModel("cross-encoder/nli-deberta-v3-small")
    print()

    # 3. Create real diverse verification views
    print("[3/5] Creating 5 real verification views...")
    view_configs = [
        ("direct", "V1-Direct"),
        ("contextualized", "V2-Contextualized"),
        ("reversed", "V3-Reversed"),
        ("truncated", "V4-Truncated"),
        ("paraphrased", "V5-Paraphrased"),
    ]
    views = [RealNLIView(nli, vtype, vid) for vtype, vid in view_configs]
    print(f"  Views: {[v.view_id for v in views]}")
    print()

    # 4. Run evaluation on ALL 817 TruthfulQA instances
    print("[4/5] Running real evaluation on all 817 TruthfulQA instances...")
    print("  This uses real NLI inference on every claim — will take time on CPU.")
    print()

    tau = 0.6  # ETG threshold: need >=3/5 views to agree

    # Accumulators
    etg_results: list[InstanceResult] = []
    baseline_no_verif = {"n_correct_accepted": 0, "n_incorrect_accepted": 0, "total_correct": 0, "total_incorrect": 0}
    baseline_single = {"n_correct_accepted": 0, "n_incorrect_accepted": 0, "total_correct": 0, "total_incorrect": 0}

    # Also track per-view agreement for multi-view analysis
    per_view_correct_entailed = [0] * len(views)
    per_view_incorrect_entailed = [0] * len(views)
    total_correct_claims = 0
    total_incorrect_claims = 0

    start_time = time.time()
    n_instances = len(ds)

    for idx in range(n_instances):
        instance = ds[idx]
        question = instance["question"]
        best_answer = instance["best_answer"]
        correct_answers = instance["correct_answers"]
        incorrect_answers = instance["incorrect_answers"]

        # Use best_answer as evidence (ground truth)
        evidence = best_answer

        # Filter out very short answers (< 3 chars) — not meaningful claims
        correct_claims = [a for a in correct_answers if len(a.strip()) >= 3]
        incorrect_claims = [a for a in incorrect_answers if len(a.strip()) >= 3]

        if not correct_claims and not incorrect_claims:
            continue

        # --- Run ETG (real multi-view verification) ---
        result = run_etg_on_instance(
            question=question,
            evidence=evidence,
            correct_claims=correct_claims,
            incorrect_claims=incorrect_claims,
            views=views,
            tau=tau,
        )
        etg_results.append(result)

        # --- Accumulate baseline: no verification ---
        baseline_no_verif["n_correct_accepted"] += len(correct_claims)
        baseline_no_verif["n_incorrect_accepted"] += len(incorrect_claims)
        baseline_no_verif["total_correct"] += len(correct_claims)
        baseline_no_verif["total_incorrect"] += len(incorrect_claims)

        # --- Accumulate baseline: single-view NLI ---
        for claim in correct_claims:
            probs = nli.predict(evidence, claim)
            if probs["entailment"] > 0.33:
                baseline_single["n_correct_accepted"] += 1
        for claim in incorrect_claims:
            probs = nli.predict(evidence, claim)
            if probs["entailment"] > 0.33:
                baseline_single["n_incorrect_accepted"] += 1
        baseline_single["total_correct"] += len(correct_claims)
        baseline_single["total_incorrect"] += len(incorrect_claims)

        # --- Track per-view statistics ---
        for cr in result.claim_results:
            if cr.is_correct:
                total_correct_claims += 1
                for v_idx, verdict in enumerate(cr.per_view_verdicts):
                    if verdict:
                        per_view_correct_entailed[v_idx] += 1
            else:
                total_incorrect_claims += 1
                for v_idx, verdict in enumerate(cr.per_view_verdicts):
                    if verdict:
                        per_view_incorrect_entailed[v_idx] += 1

        # Progress
        if (idx + 1) % 50 == 0 or idx == n_instances - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            eta = (n_instances - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_instances}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    total_time = time.time() - start_time

    # =========================================================================
    # 5. Compute and report REAL metrics
    # =========================================================================
    print()
    print("[5/5] Computing real metrics...")
    print()
    print("=" * 70)
    print("RESULTS: REAL EMPIRICAL EVALUATION ON TRUTHFULQA (817 instances)")
    print(f"NLI Model: cross-encoder/nli-deberta-v3-small (DeBERTa v3)")
    print(f"Dataset: TruthfulQA (Lin et al., ACL 2022) — {n_instances} questions")
    print(f"Views: 5 real NLI views (direct, contextualized, reversed, truncated, paraphrased)")
    print(f"Threshold: tau = {tau}")
    print(f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print()

    # --- ETG aggregate metrics ---
    etg_correct_accepted = sum(r.n_correct_accepted for r in etg_results)
    etg_correct_rejected = sum(r.n_correct_rejected for r in etg_results)
    etg_incorrect_accepted = sum(r.n_incorrect_accepted for r in etg_results)
    etg_incorrect_rejected = sum(r.n_incorrect_rejected for r in etg_results)

    etg_total_accepted = etg_correct_accepted + etg_incorrect_accepted
    etg_total_claims = etg_correct_accepted + etg_correct_rejected + etg_incorrect_accepted + etg_incorrect_rejected

    etg_precision = etg_correct_accepted / etg_total_accepted if etg_total_accepted > 0 else 1.0
    etg_recall = etg_correct_accepted / (etg_correct_accepted + etg_correct_rejected) if (etg_correct_accepted + etg_correct_rejected) > 0 else 0
    etg_f1 = 2 * etg_precision * etg_recall / (etg_precision + etg_recall) if (etg_precision + etg_recall) > 0 else 0
    etg_hallucination_rate = etg_incorrect_accepted / etg_total_accepted if etg_total_accepted > 0 else 0
    etg_factscore = etg_precision  # FactScore = fraction of accepted claims that are correct

    # Hallucination detection metrics
    etg_detection_accuracy = (etg_incorrect_rejected + etg_correct_accepted) / etg_total_claims if etg_total_claims > 0 else 0
    etg_false_positive_rate = etg_incorrect_accepted / (etg_incorrect_accepted + etg_incorrect_rejected) if (etg_incorrect_accepted + etg_incorrect_rejected) > 0 else 0

    # --- No-verification baseline ---
    nv_total = baseline_no_verif["n_correct_accepted"] + baseline_no_verif["n_incorrect_accepted"]
    nv_precision = baseline_no_verif["n_correct_accepted"] / nv_total if nv_total > 0 else 0
    nv_hallucination_rate = baseline_no_verif["n_incorrect_accepted"] / nv_total if nv_total > 0 else 0

    # --- Single-view baseline ---
    sv_total_accepted = baseline_single["n_correct_accepted"] + baseline_single["n_incorrect_accepted"]
    sv_precision = baseline_single["n_correct_accepted"] / sv_total_accepted if sv_total_accepted > 0 else 1.0
    sv_recall = baseline_single["n_correct_accepted"] / baseline_single["total_correct"] if baseline_single["total_correct"] > 0 else 0
    sv_hallucination_rate = baseline_single["n_incorrect_accepted"] / sv_total_accepted if sv_total_accepted > 0 else 0
    sv_f1 = 2 * sv_precision * sv_recall / (sv_precision + sv_recall) if (sv_precision + sv_recall) > 0 else 0
    sv_fpr = baseline_single["n_incorrect_accepted"] / baseline_single["total_incorrect"] if baseline_single["total_incorrect"] > 0 else 0

    # =========================================================================
    # Print results
    # =========================================================================

    print("--- MAIN RESULTS: Claim-Level Metrics ---")
    print()
    print(f"{'Metric':<35} {'No Verif':>10} {'Single NLI':>10} {'ETG (N=5)':>10}")
    print("-" * 70)
    print(f"{'FactScore (claim precision):':<35} {nv_precision:>10.4f} {sv_precision:>10.4f} {etg_precision:>10.4f}")
    print(f"{'Recall (correct claims kept):':<35} {'1.0000':>10} {sv_recall:>10.4f} {etg_recall:>10.4f}")
    print(f"{'F1 Score:':<35} {2*nv_precision/(1+nv_precision):>10.4f} {sv_f1:>10.4f} {etg_f1:>10.4f}")
    print(f"{'Hallucination Rate:':<35} {nv_hallucination_rate:>10.4f} {sv_hallucination_rate:>10.4f} {etg_hallucination_rate:>10.4f}")
    print(f"{'False Positive Rate:':<35} {'1.0000':>10} {sv_fpr:>10.4f} {etg_false_positive_rate:>10.4f}")
    print(f"{'Detection Accuracy:':<35} {nv_precision:>10.4f} {'—':>10} {etg_detection_accuracy:>10.4f}")
    print()

    print("--- CONFUSION MATRIX: ETG (N=5 views, tau=0.6) ---")
    print()
    print(f"  {'':>25} {'Predicted Faithful':>20} {'Predicted Halluc.':>20}")
    print(f"  {'Actually Correct:':<25} {etg_correct_accepted:>20d} {etg_correct_rejected:>20d}")
    print(f"  {'Actually Incorrect:':<25} {etg_incorrect_accepted:>20d} {etg_incorrect_rejected:>20d}")
    print()
    print(f"  Total claims evaluated: {etg_total_claims}")
    print(f"  Total correct claims: {etg_correct_accepted + etg_correct_rejected}")
    print(f"  Total incorrect claims: {etg_incorrect_accepted + etg_incorrect_rejected}")
    print()

    print("--- PER-VIEW ANALYSIS ---")
    print()
    print(f"  {'View':<25} {'TPR (correct entailed)':>25} {'FPR (incorrect entailed)':>25}")
    print("  " + "-" * 75)
    for v_idx, (vtype, vid) in enumerate(view_configs):
        tpr = per_view_correct_entailed[v_idx] / total_correct_claims if total_correct_claims > 0 else 0
        fpr = per_view_incorrect_entailed[v_idx] / total_incorrect_claims if total_incorrect_claims > 0 else 0
        print(f"  {vid:<25} {tpr:>25.4f} {fpr:>25.4f}")
    print()

    # Multi-view analysis: how does support mass distribute?
    print("--- SUPPORT MASS DISTRIBUTION ---")
    print()
    correct_masses = [cr.support_mass for r in etg_results for cr in r.claim_results if cr.is_correct]
    incorrect_masses = [cr.support_mass for r in etg_results for cr in r.claim_results if not cr.is_correct]

    for label, masses in [("Correct claims", correct_masses), ("Incorrect claims", incorrect_masses)]:
        if not masses:
            continue
        avg = sum(masses) / len(masses)
        mass_0 = sum(1 for m in masses if m == 0) / len(masses)
        mass_low = sum(1 for m in masses if 0 < m < tau) / len(masses)
        mass_high = sum(1 for m in masses if m >= tau) / len(masses)
        print(f"  {label} (n={len(masses)}):")
        print(f"    Mean support mass: {avg:.4f}")
        print(f"    m=0 (all views reject):  {mass_0:.1%}")
        print(f"    0<m<{tau} (below threshold): {mass_low:.1%}")
        print(f"    m>={tau} (accepted by ETG):  {mass_high:.1%}")
        print()

    # ETG improvement over baselines
    print("--- IMPROVEMENT ANALYSIS ---")
    print()
    hallu_reduction_vs_no_verif = (nv_hallucination_rate - etg_hallucination_rate) / nv_hallucination_rate * 100 if nv_hallucination_rate > 0 else 0
    hallu_reduction_vs_single = (sv_hallucination_rate - etg_hallucination_rate) / sv_hallucination_rate * 100 if sv_hallucination_rate > 0 else 0
    print(f"  Hallucination rate reduction vs no verification: {hallu_reduction_vs_no_verif:.1f}%")
    print(f"  Hallucination rate reduction vs single-view NLI: {hallu_reduction_vs_single:.1f}%")
    print(f"  FactScore improvement vs no verification: +{etg_precision - nv_precision:.4f}")
    print(f"  FactScore improvement vs single-view NLI: +{etg_precision - sv_precision:.4f}")
    print()

    # Theoretical bound comparison
    print("--- THEORETICAL vs EMPIRICAL ---")
    print()
    empirical_fpr = etg_false_positive_rate
    # Average per-view FPR
    avg_alpha = sum(per_view_incorrect_entailed[i] / total_incorrect_claims for i in range(len(views))) / len(views) if total_incorrect_claims > 0 else 0
    theoretical_bound = hallucination_upper_bound(n_views=len(views), tau=tau, alpha=max(avg_alpha, 0.001))
    print(f"  Empirical per-view FPR (alpha): {avg_alpha:.4f}")
    print(f"  Proposition 1 bound (N={len(views)}, tau={tau}, alpha={avg_alpha:.4f}): {theoretical_bound:.6f}")
    print(f"  Empirical multi-view FPR: {empirical_fpr:.4f}")
    print(f"  Bound holds: {empirical_fpr <= theoretical_bound + 0.001}")
    print()

    # Threshold sweep
    print("--- THRESHOLD SWEEP (real data) ---")
    print()
    print(f"  {'tau':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Halluc Rate':>12} {'Accepted':>10}")
    print("  " + "-" * 65)
    for sweep_tau in [0.2, 0.4, 0.6, 0.8, 1.0]:
        tp = sum(1 for r in etg_results for cr in r.claim_results if cr.is_correct and cr.support_mass >= sweep_tau)
        fp = sum(1 for r in etg_results for cr in r.claim_results if not cr.is_correct and cr.support_mass >= sweep_tau)
        fn = sum(1 for r in etg_results for cr in r.claim_results if cr.is_correct and cr.support_mass < sweep_tau)
        total_acc = tp + fp
        p = tp / total_acc if total_acc > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        h = fp / total_acc if total_acc > 0 else 0
        print(f"  {sweep_tau:<8.1f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {h:>12.4f} {total_acc:>10d}")
    print()

    # Save full results as JSON
    output = {
        "dataset": "TruthfulQA (Lin et al., ACL 2022)",
        "n_instances": n_instances,
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "n_views": len(views),
        "tau": tau,
        "total_claims_evaluated": etg_total_claims,
        "total_correct_claims": etg_correct_accepted + etg_correct_rejected,
        "total_incorrect_claims": etg_incorrect_accepted + etg_incorrect_rejected,
        "inference_time_seconds": total_time,
        "results": {
            "no_verification": {
                "factscore": round(nv_precision, 4),
                "hallucination_rate": round(nv_hallucination_rate, 4),
                "recall": 1.0,
            },
            "single_view_nli": {
                "factscore": round(sv_precision, 4),
                "hallucination_rate": round(sv_hallucination_rate, 4),
                "recall": round(sv_recall, 4),
                "f1": round(sv_f1, 4),
            },
            "etg_multiview": {
                "factscore": round(etg_precision, 4),
                "hallucination_rate": round(etg_hallucination_rate, 4),
                "recall": round(etg_recall, 4),
                "f1": round(etg_f1, 4),
                "false_positive_rate": round(etg_false_positive_rate, 4),
                "detection_accuracy": round(etg_detection_accuracy, 4),
            },
        },
        "confusion_matrix": {
            "true_positive": etg_correct_accepted,
            "false_negative": etg_correct_rejected,
            "false_positive": etg_incorrect_accepted,
            "true_negative": etg_incorrect_rejected,
        },
        "per_view_tpr": {
            vid: round(per_view_correct_entailed[i] / total_correct_claims, 4) if total_correct_claims > 0 else 0
            for i, (_, vid) in enumerate(view_configs)
        },
        "per_view_fpr": {
            vid: round(per_view_incorrect_entailed[i] / total_incorrect_claims, 4) if total_incorrect_claims > 0 else 0
            for i, (_, vid) in enumerate(view_configs)
        },
        "theoretical": {
            "empirical_alpha": round(avg_alpha, 4),
            "proposition_1_bound": round(theoretical_bound, 6),
            "empirical_fpr": round(empirical_fpr, 4),
        },
    }

    output_path = Path(__file__).parent.parent / "results" / "real_evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to: {output_path}")
    print()

    print("=" * 70)
    print("EVALUATION COMPLETE — ALL NUMBERS ARE REAL")
    print(f"Real model: DeBERTa v3 (cross-encoder/nli-deberta-v3-small)")
    print(f"Real dataset: TruthfulQA ({n_instances} questions, {etg_total_claims} total claims)")
    print(f"Real inference: {total_time:.1f}s on CPU")
    print("=" * 70)


if __name__ == "__main__":
    main()
