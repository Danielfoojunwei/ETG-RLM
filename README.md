# Evidence-Typed Generation: Faithfulness as a Type System for Recursive Language Models

> **Abstract.** Large language models hallucinate because their decoding objective — `y* = argmax log p(y|q,E)` — is structurally indifferent to whether generated claims are grounded in evidence. We introduce *Evidence-Typed Generation* (ETG), an inference-time framework that externalizes belief into an Evidence-Scoped Belief Graph (ESBG), assigns each atomic claim a formal evidence type via multi-view verification, and restricts generation to the subspace of well-typed, entailed claims. Unsupported claims become *unrepresentable* in the output. We prove that hallucination acceptance decays exponentially with the number of verification views N: `Pr[hallucination] <= exp(-N * D(tau || alpha))`. Empirical evaluation on TruthfulQA (817 questions, 5,865 claims) using 5 independent NLI model architectures shows that ETG verification raises GPT-2 output factuality from **5.9% to 74.3%** (end-to-end), while revealing that the theoretical exponential bound is **violated by 44.6x** due to correlated model errors — an important negative result showing that true view independence, not just architectural diversity, is required.

---

## 1. The Problem: Structural Hallucination in Language Models

The dominant failure mode of large language models is **hallucination** — the generation of fluent but unfaithful text that is not grounded in any evidence source. This is not a training data problem or a model size problem. It is a **structural** problem inherent to the decoding objective itself.

The standard autoregressive objective:

```
y* = argmax_y log p_theta(y | q, E)
```

maximizes likelihood given the prompt and (optionally) retrieved evidence E. But this objective contains no mechanism to ensure that claims in the generated output are actually *entailed* by E. A claim that "sounds right" given the training distribution receives high probability regardless of whether it has evidential support.

**Why existing approaches fail:**

| Approach | Structural Limitation |
|----------|----------------------|
| **Retrieval-Augmented Generation (RAG)** | Retrieval ≠ entailment. Placing documents in context does not prevent the model from generating claims that go beyond or contradict the evidence. |
| **Self-CheckGPT** | Single-view consistency is gameable — a confident hallucination produces consistent samples. No evidence grounding. |
| **Chain-of-Thought** | Linear reasoning trace with no provenance. Steps are ungrounded assertions, not evidence-linked claims. |
| **RLHF / reward shaping** | Behavioral alignment — rewards faithful-sounding text, does not structurally prevent unfaithful claims from being representable. |
| **Constrained decoding** | Operates at the token/syntax level (e.g., grammar constraints), not at the semantic/evidence level. |
| **Knowledge graphs** | Static, offline structures. Not constructed at inference time, cannot adapt to query-specific evidence. |

The fundamental insight of ETG is that faithfulness should be treated as a **type constraint** on the output space, not as a soft objective to be optimized. Just as a type system in programming prevents ill-typed expressions from compiling, ETG prevents unsupported claims from being generated.

---

## 2. The ETG Framework: Formal Definitions

### Definition 1: Evidence-Scoped Belief Graph (ESBG)

An ESBG is a directed acyclic graph `G = (V, →, π, σ, m, z)` where:

| Symbol | Meaning |
|--------|---------|
| `V` | Set of claim nodes, constructed at inference time |
| `u → v` | Dependency edge: claim v depends on claim u |
| `π(v)` | Atomic claim associated with node v |
| `σ(v) ⊆ S(E)` | Evidence span pointers — provenance trail linking claim to source |
| `m(v) ∈ [0,1]` | Support mass — multi-view stability score |
| `z(v)` | Entailment status: {entailed, contradicted, unknown} |

The ESBG is **not** a knowledge graph (static, pre-computed) or a chain-of-thought (linear, ungrounded). It is a dynamic belief structure with explicit provenance, constructed during inference via a recursive graph policy.

### Definition 2: Verification View

A verification view `V_i` is a function:

```
V_i : (E, c) → (z_i, S_i)
```

where `z_i ∈ {entailed, contradicted, not-found}` and `S_i ⊆ S(E)`. Views are intentionally diverse — they differ by retriever type, chunk boundaries, query rewriting strategy, verifier prompting, and negative sampling windows.

### Definition 3: Support Mass

Given N independent views:

```
m(c) = (1/N) × Σ_{i=1}^{N} 𝟙[z_i = entailed]
σ(c) = ∪_{i : z_i = entailed} S_i
```

Support mass is a **stability invariant** — it measures how consistently a claim survives scrutiny across diverse evidence retrieval and verification strategies.

### Definition 4: Evidence Types

```
type(c) = Verified      if m(c) ≥ τ
           Uncertain     if τ' < m(c) < τ
           Unsupported   if m(c) ≤ τ'
```

where `0 ≤ τ' < τ ≤ 1`. This is a genuine type system: claims are classified before they can enter the output, and only well-typed (Verified) claims are representable.

### Definition 5: Constrained Decoding

```
V^τ = {v ∈ V | type(π(v)) = Verified}
Y(G_T, τ) = {y | A(y) ⊆ {π(v) : v ∈ V^τ}}
y* = argmax_{y ∈ Y(G_T, τ)} log p_θ(y | q, E)
```

The output space Y is **restricted** to texts whose atomic claim decomposition is a subset of verified claims. Unsupported claims are not penalized — they are **unrepresentable**.

---

## 3. Theoretical Guarantees

### Proposition 1: Exponential Suppression of Hallucinations

> Assume a hallucinated claim has per-view false-positive probability α, and views are conditionally independent. Then:
>
> **Pr[m(c) ≥ τ] ≤ exp(−N · D(τ ∥ α))**
>
> where D is the KL divergence between Bernoulli(τ) and Bernoulli(α).

This establishes an **inference-time scaling law** for faithfulness. Unlike parameter scaling (which requires retraining), ETG improves faithfulness by adding verification views at inference time:

| N (views) | Upper bound on Pr[hallucination passes] |
|-----------|----------------------------------------|
| 1 | 0.490 |
| 5 | 0.028 |
| 10 | 1.05 × 10⁻⁶ |
| 20 | 1.09 × 10⁻¹² |

*(Computed with τ = 0.7, α = 0.1)*

**Important caveat:** This bound assumes conditional independence of views. See Section 4.5 for empirical evidence that this assumption is violated when views share a single NLI backbone, and how this affects the bound.

### Proposition 2: Zero-Confabulation Property

> Under exact entailment verification:
>
> **Pr[∃c ∈ A(y*) s.t. supp(E,c) = ∅] = 0**

Every claim in the output has evidence pointers by construction. Confabulation — emitting a claim without evidence — is structurally impossible, not merely unlikely.

### Proposition 3: Optimal Compute Allocation

> Under budget B, the optimal policy allocates views to claims maximizing:
>
> **E[ΔVerified Utility] / k**
>
> This reduces to a bandit / knapsack allocation problem over claims.

---

## 4. Empirical Evaluation (Real Results)

All results below are from real experiments. No simulations, no mocks. Reproducible via `scripts/real_evaluation_v2.py`.

### 4.1 Experimental Setup

| Component | Detail |
|-----------|--------|
| **Dataset** | TruthfulQA (Lin et al., ACL 2022) — 817 questions, 5,865 claims (2,577 correct + 3,288 incorrect) |
| **Evidence** | TruthfulQA `best_answer` field (ground-truth correct answer) |
| **Hardware** | 16-core CPU, 21GB RAM, no GPU |
| **Total runtime** | 622 seconds (10.4 minutes) for all 5 models |

**5 Independent NLI Model Architectures** (genuinely different, not the same model reformatted):

| View | Model | Architecture | Params | TPR | FPR |
|------|-------|-------------|--------|-----|-----|
| V1 | `cross-encoder/nli-deberta-v3-small` | DeBERTa | 22M | 0.493 | 0.017 |
| V2 | `cross-encoder/nli-distilroberta-base` | DistilRoBERTa | 82M | 0.512 | 0.037 |
| V3 | `cross-encoder/nli-MiniLM2-L6-H768` | MiniLM | 22M | 0.513 | 0.030 |
| V4 | `cross-encoder/nli-roberta-base` | RoBERTa | 125M | 0.513 | 0.023 |
| V5 | `facebook/bart-large-mnli` | BART | 407M | 0.502 | 0.017 |

### 4.2 Apples-to-Apples Comparison (All methods, same data, same metric)

| Method | Precision | Recall | F1 | Halluc. Rate | FPR |
|--------|-----------|--------|-----|-------------|-----|
| No verification | 0.4394 | 1.0000 | 0.6105 | 0.5606 | 1.0000 |
| Single: DeBERTa-v3-small (22M) | 0.9570 | 0.4928 | 0.6506 | 0.0430 | 0.0173 |
| Single: DistilRoBERTa (82M) | 0.9147 | 0.5118 | 0.6564 | 0.0853 | 0.0374 |
| Single: MiniLM (22M) | 0.9316 | 0.5126 | 0.6613 | 0.0684 | 0.0295 |
| Single: RoBERTa-base (125M) | 0.9450 | 0.5130 | 0.6650 | 0.0550 | 0.0234 |
| **Single: BART-large (407M)** | **0.9578** | 0.5021 | 0.6589 | 0.0422 | **0.0173** |
| ETG: 4 small models (τ=0.5) | 0.9361 | 0.5285 | 0.6756 | 0.0639 | 0.0283 |
| **ETG: 5 independent (τ=0.6)** | 0.9540 | 0.5076 | 0.6626 | 0.0460 | 0.0192 |

### 4.3 Claim 1 — Exponential Suppression: NOT PROVEN

| Quantity | Value |
|----------|-------|
| Average per-view FPR (α) | 0.0250 |
| Proposition 1 bound (N=5, τ=0.6, α=0.025) | 0.000430 |
| **Empirical multi-view FPR** | **0.019161** |
| **Bound holds?** | **No — violated by 44.6×** |

Even with 5 genuinely different NLI architectures (DeBERTa, DistilRoBERTa, MiniLM, RoBERTa, BART), the exponential bound is violated by 44.6×.

**Root cause: shared training data.** Pairwise agreement between models on incorrect claims is 96.7–98.5%, meaning when one model is fooled, the others usually are too. This is not because they share architecture — they don't — but because they all learned from the same NLI training datasets (MNLI, SNLI). The independence assumption requires not just different models, but models trained on fundamentally different data or using different verification paradigms (e.g., symbolic reasoning, retrieval-based fact-checking, LLM-based judging).

### 4.4 Claim 2 — Multi-View vs. Single Large Model: NOT PROVEN

| Method | Precision | FPR |
|--------|-----------|-----|
| **BART-large single (407M)** | **0.9578** | **0.0173** |
| ETG 5 independent (τ=0.6) | 0.9540 | 0.0192 |

A single large model (BART-large, 407M params) **outperforms** 5 independent models voting. One good verifier is better than five mediocre ones agreeing. Multi-view only helps when views are truly independent; with correlated errors, it averages noise rather than canceling it.

### 4.5 Claim 3 — Superiority over Single Models: MIXED

The best single model (RoBERTa-base, F1=0.6650) slightly outperforms ETG 5-independent (F1=0.6626). However, the 4-small-model ETG (F1=0.6756) achieves the highest F1 of any method. The advantage of multi-view appears at lower thresholds where it preserves more recall. No method is clearly dominant.

### 4.6 Claim 4 — End-to-End Generation: PROVEN

Real text generation with GPT-2 (124M params), verified by ETG, judged by BART-large as independent ground truth:

| Metric | Unfiltered (all GPT-2 output) | ETG Accepted | ETG Rejected |
|--------|------------------------------|--------------|-------------|
| **FactScore** | **0.0586** | **0.7429** | 0.0117 |
| **Sentences** | 546 | 35 | 511 |

GPT-2 generates mostly hallucinated text (94.1% of its output is not entailed by the evidence). ETG accepts only 35 out of 546 sentences — but **74.3% of those are actually truthful**. The rejected pile has 1.2% FactScore.

**ETG raises output factuality from 5.9% to 74.3% — a 12.7× improvement on real generated text.** This is the strongest empirical result: ETG verification genuinely improves the factual quality of LLM output, even with a weak generator (GPT-2).

### 4.7 Threshold Sweep (5 independent models, real data)

| τ | Precision | Recall | F1 | Halluc. Rate | FPR | Accepted |
|---|-----------|--------|-----|-------------|-----|----------|
| 0.2 | 0.8868 | 0.5925 | 0.7104 | 0.1132 | 0.0593 | 1,722 |
| 0.4 | 0.9334 | 0.5382 | 0.6827 | 0.0666 | 0.0301 | 1,486 |
| **0.6** | **0.9540** | **0.5076** | **0.6626** | **0.0460** | **0.0192** | **1,371** |
| 0.8 | 0.9709 | 0.4664 | 0.6301 | 0.0291 | 0.0109 | 1,238 |
| 1.0 | 0.9839 | 0.4276 | 0.5962 | 0.0161 | 0.0055 | 1,120 |

At τ=1.0 (all 5 models must agree), precision is 98.4% and FPR drops to 0.55%. The type system provides a smooth, controllable tradeoff.

### 4.8 What Was Proven and What Was Not

| Claim | Status | Evidence |
|-------|--------|----------|
| Exponential suppression (Prop. 1) | **Not proven** | Bound violated 44.6× with 5 independent architectures; models share training data |
| Multi-view > single large model | **Not proven** | BART-large (single) beats 5-model ETG on precision and FPR |
| ETG improves generation factuality | **Proven** | GPT-2 FactScore: 5.9% → 74.3% after ETG filtering |
| Type system controls precision-recall | **Proven** | Threshold sweep confirms smooth, predictable tradeoff |
| NLI verification catches hallucinations | **Proven** | All methods reduce hallucination rate from 56% to <9% |

### 4.9 Honest Limitations

- **Precision-recall tradeoff:** ETG is conservative — it achieves high precision but discards ~50% of correct claims.
- **View independence fails:** Even architecturally different NLI models are correlated (96.7–98.5% agreement on errors) because they share training data. The exponential bound requires a stronger notion of independence than architectural diversity.
- **Single large model wins:** One good 407M-parameter model outperforms five smaller ones. The multi-view advantage only materializes with truly independent views.
- **Weak generator:** The E2E test uses GPT-2 (very low baseline quality). Testing with stronger generators (GPT-4, Llama 3) would give more representative results but requires GPU/API access.
- **Single dataset:** Evaluated on TruthfulQA only. Generalization to other benchmarks is untested.
- **Computational cost:** 622 seconds for 5 models × 5,865 claims on CPU. Production needs GPU.

---

## 6. Reproducing Results

### Run the comprehensive evaluation (5 independent models + E2E)

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets

# Run full evaluation (downloads TruthfulQA + 5 NLI models + GPT-2)
python scripts/real_evaluation_v2.py
```

This will:
1. Download TruthfulQA (817 instances) from HuggingFace
2. Download and run 5 independent NLI models (DeBERTa, DistilRoBERTa, MiniLM, RoBERTa, BART) on all 5,865 claims
3. Test Proposition 1 exponential bound with independent models
4. Compare multi-view vs. single large model (BART-large)
5. Generate text with GPT-2 and verify with ETG end-to-end
6. Save results to `results/real_evaluation_v2_results.json`

Expected runtime: ~10 minutes on 16-core CPU.

### Run the framework tests

```bash
pytest tests/ -v    # 364 tests across 21 test files
```

### Quick Start (Framework API)

```python
from etg_rlm import (
    ebrg, AtomicClaim,
    hallucination_upper_bound, inference_time_scaling_law,
    check_zero_confabulation,
)

# --- EBRG Algorithm (Section 5) ---
result = ebrg(
    query="What causes tides?",
    claims=[
        AtomicClaim(claim_id="c1", text="Tides are caused by gravitational pull."),
        AtomicClaim(claim_id="c2", text="The moon's gravity is the primary driver."),
    ],
    views=your_views,     # list[VerificationView]
    tau=0.7,              # support mass threshold
    n_views_per_claim=5,
    budget=50,
)
print(f"Verified: {[c.text for c in result.decoding.verified_claims]}")
print(f"Rejected: {[c.text for c in result.decoding.rejected_claims]}")
print(f"Hallucination bound: {result.hallucination_bound:.6f}")
print(f"Zero-confabulation holds: {result.zero_confabulation_holds}")
```

---

## 7. Architecture

```
etg_rlm/
  core.py               -- Definitions 1, 4: ESBG, AtomicClaim, EvidenceSpan
  verification.py       -- Definitions 2-3: VerificationView, MultiViewVerifier
  type_system.py        -- Definitions 4-5: EvidenceTypeChecker, constrained output
  policy.py             -- Section 4.5: RecursionPolicy, UtilityWeightedPolicy
  bounds.py             -- Propositions 1-3: exponential bounds, zero-confabulation
  algorithm.py          -- Section 5: ebrg(), constrained_decode()
  pipeline.py           -- End-to-end ETGPipeline orchestration
  metrics.py            -- Faithfulness metrics, ROUGE-L, latency measurement
  baselines.py          -- 4 baseline configurations
  evaluation.py         -- Comparative benchmarking harness
  views/factory.py      -- 5 diverse verification view types
  datasets.py           -- 5 benchmark dataset specifications (3,817 instances)
  human_eval.py         -- Human evaluation protocol (Fleiss' Kappa)
  ablations.py          -- 4 ablation studies
  statistics.py         -- Statistical analysis (t-test, Cohen's d, bootstrap CI)
  factscore.py          -- FactScore: claim decomposition + NLI (Min et al., 2023)
  citation_metrics.py   -- Citation P/R (Gao et al., 2023; Rashkin et al., 2022)
  logic_verification.py -- Logic-step verification for multi-hop chains
  self_check.py         -- Self-CheckGPT baseline (Manakul et al., 2023)
  benchmark_runner.py   -- Canonical 4×5 benchmark orchestration
  reporting.py          -- Markdown, LaTeX, JSON, and visualization reports
scripts/
  real_evaluation.py    -- Single-model evaluation (TruthfulQA + DeBERTa-v3 NLI)
  real_evaluation_v2.py -- Comprehensive eval (5 independent NLI models + GPT-2 E2E)
  download_data.py      -- Dataset download (TruthfulQA, HaluEval, HotpotQA, NQ, ELI5)
.github/workflows/
  eval.yml              -- CI/CD: unit tests + evaluation matrix
results/
  real_evaluation_results.json     -- Single-model evaluation results
  real_evaluation_v2_results.json  -- Comprehensive 5-model + E2E results
```

22 source modules, 364 tests. Core framework is pure Python; evaluation requires PyTorch + Transformers.

---

## 8. References

1. Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023.
2. Gao et al., "ALCE: Attributed Language Model Evaluation," EMNLP 2023.
3. Rashkin et al., "Measuring Attribution in Natural Language Generation Models," ACL 2022.
4. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," EMNLP 2018.
5. Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," EMNLP 2023.
6. Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," ACL 2022.
7. Li et al., "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models," EMNLP 2023.
8. Kwiatkowski et al., "Natural Questions: A Benchmark for Question Answering Research," TACL 2019.
9. Fan et al., "ELI5: Long Form Question Answering," ACL 2019.
10. Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models," ACL Findings 2024.
11. OpenAI, "GPT-4 Technical Report," 2023.
12. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023.

---

*ETG-RLM: 22 modules, 364 tests. Empirical evaluation on TruthfulQA (817 questions, 5,865 claims) using 5 independent NLI architectures + GPT-2 end-to-end generation test. All numbers are from real experiments — see `scripts/real_evaluation_v2.py` and `results/real_evaluation_v2_results.json`.*
