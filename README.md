# ETG-RLM

**Evidence-Typed Generation: Faithfulness as a Type System for Recursive Language Models**

> We show that hallucinations arise from read/write entanglement in next-token decoding, and introduce Evidence-Typed Generation -- an RLM-native inference framework that externalizes belief into evidence-scoped graphs and restricts generation to well-typed, entailed claims, yielding exponential suppression of hallucinations under inference-time scaling.

## Paper Structure (implemented)

| Section | Topic | Module |
|---------|-------|--------|
| 4.1 | Problem setup, claim extraction | `core.py` |
| 4.2 | Evidence-Scoped Belief Graph (Definition 1) | `core.py` |
| 4.3 | Multi-view verification (Definitions 2-3) | `verification.py` |
| 4.4 | Evidence typing system (Definition 4) | `type_system.py` |
| 4.5 | Recursive graph construction policy | `policy.py` |
| 4.6 | Constrained decoding (Definition 5) | `algorithm.py`, `type_system.py` |
| 5 | EBRG algorithm (pseudocode) | `algorithm.py` |
| 6 | Propositions 1-3 (theoretical bounds) | `bounds.py` |

## Formal Definitions

### Definition 1: Evidence-Scoped Belief Graph (ESBG)

A tuple `G = (V, ->, pi, sigma, m, z)` where:
- **V**: set of nodes
- **pi(v)**: atomic claim associated with node v
- **sigma(v) subset S(E)**: evidence span pointers
- **m(v) in [0,1]**: support mass
- **z(v)**: entailment status {entailed, contradicted, unknown}
- **u -> v**: dependency edge (claim v depends on claim u)

The graph is a DAG, constructed at inference time.

### Definition 2: Verification View

A function `V_i : (E, c) -> (z_i, S_i)` where z_i is the entailment verdict and S_i is the supporting span set. Views differ by query rewriting, chunk boundaries, retriever randomness, verifier prompting, and negative sampling windows.

### Definition 3: Support Mass

```
m(c) = (1/N) * sum_{i=1}^{N} 1[z_i = entailed]
sigma(c) = union_{i : z_i = entailed} S_i
```

### Definition 4: Evidence Types

```
type(c) =
    Verified      if m(c) >= tau
    Uncertain     if tau' < m(c) < tau
    Unsupported   if m(c) <= tau'
```

### Definition 5: Constrained Decoding

```
V^tau = {v in V | type(pi(v)) = Verified}
Y(G_T, tau) = {y | A(y) subset {pi(v) : v in V^tau}}
y* = argmax_{y in Y(G_T, tau)} log p_theta(y | q, E)
```

Unsupported claims are **unrepresentable** in the output space.

## Theoretical Properties

### Proposition 1: Exponential Suppression of Hallucinations

```
Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))
```

Increasing N yields exponential decay in hallucination acceptance -- an **inference-time scaling law** for faithfulness.

### Proposition 2: Zero-Confabulation Property

Under exact entailment verification: `Pr[exists c in A(y*) s.t. supp(E,c) = empty] = 0`. Hallucination is eliminated by construction, not by reward shaping.

### Proposition 3: Optimal Compute Allocation

Under budget B, the optimal policy allocates views to claims maximizing `E[Delta Verified Utility] / k`. This reduces to a bandit / knapsack allocation problem.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

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
    views=your_views,     # list of VerificationView implementations
    tau=0.7,              # support mass threshold
    n_views_per_claim=5,  # views per claim
    budget=50,
)
print(f"Verified: {[c.text for c in result.decoding.verified_claims]}")
print(f"Rejected: {[c.text for c in result.decoding.rejected_claims]}")
print(f"Hallucination bound: {result.hallucination_bound:.6f}")
print(f"Zero-confabulation holds: {result.zero_confabulation_holds}")

# --- Proposition 1: Inference-time scaling law ---
scaling = inference_time_scaling_law(tau=0.7, alpha=0.1, max_n=50)
for n, bound in zip(scaling.n_views_sequence, scaling.bounds_sequence):
    print(f"  N={n:2d}  Pr[hallucination passes] <= {bound:.2e}")

# --- Theoretical bounds ---
bound = hallucination_upper_bound(n_views=10, tau=0.7, alpha=0.1)
print(f"Bound with N=10: {bound:.6f}")
```

## Evaluation Framework

### Baselines (4 controls)

| Control | Description | Purpose |
|---------|-------------|---------|
| Standard LLM | Llama 3 70B zero-shot | Base hallucination rate |
| Standard RAG | Generator + dense retriever (top-k) | Industry standard |
| RAG + Verifier | RAG with post-hoc claim verification | Corrective vs. preventative |
| Self-Critique | Single-view LLM self-check | Behavioral vs. structural |

### Verification Views (N=5)

| View | Retriever | Chunking | Special |
|------|-----------|----------|---------|
| V1 | Dense (Sentence-BERT) | 512 tokens | Baseline |
| V2 | Sparse (BM25) | 512 tokens | Lexical complement |
| V3 | Dense | 128 tokens | Fine-grained precision |
| V4 | Dense | 512 tokens | Query rewriting (T5) |
| V5 | Dense | 512 tokens | Negative sampling |

### KPI Targets

| KPI | Target |
|-----|--------|
| Hallucination rate | < 1% |
| Reduction vs. RAG | > 90% |
| ROUGE-L vs. baseline | Maintained (>= 95%) |
| Latency | < 500ms/token |

## Architecture

```
etg_rlm/
  core.py          -- Definitions 1, 4: EvidenceSpan, AtomicClaim, ESBG
  verification.py  -- Definitions 2-3: VerificationView, MultiViewVerifier
  type_system.py   -- Definitions 4-5: EvidenceTypeChecker, constrained output space
  policy.py        -- Section 4.5: RecursionPolicy, UtilityWeightedPolicy
  bounds.py        -- Propositions 1-3: hallucination bounds, zero-confabulation, allocation
  algorithm.py     -- Section 5: ebrg(), constrained_decode()
  pipeline.py      -- End-to-end ETGPipeline orchestration
  metrics.py       -- Hallucination rate, claim precision/recall, ROUGE-L, latency
  baselines.py     -- 4 baseline configurations (Standard LLM, RAG, RAG+Verifier, Self-Critique)
  evaluation.py    -- Comparative benchmarking harness with KPI checking
  views/factory.py -- 5 diverse verification view types with factory
  datasets.py      -- 5 dataset specifications (NQ, HotpotQA, TruthfulQA, HaluEval, XSum)
  human_eval.py    -- Human evaluation protocol (faithfulness rating, pairwise preference, Fleiss' Kappa)
  ablations.py     -- 4 ablation studies (NoMultiView, NoConstraint, Threshold-Sweep, PolicyAblation)
  statistics.py    -- Statistical analysis (paired t-test, Cohen's d, bootstrapped CIs)
  factscore.py     -- FactScore metrics: atomic claim decomposition + NLI scoring (Min et al., 2023)
  citation_metrics.py -- Citation Precision/Recall (Gao et al., 2023; Rashkin et al., 2022)
  logic_verification.py -- Logic-Step Verification for multi-hop reasoning chains
  self_check.py    -- Self-CheckGPT baseline: zero-resource hallucination detection (Manakul et al., 2023)
  benchmark_runner.py -- Canonical benchmark runner: all models x all datasets orchestration
  reporting.py     -- Report generation: markdown tables, LaTeX, JSON, visualization specs
scripts/
  download_data.py -- Dataset download script (TruthfulQA, HaluEval, HotpotQA, NQ, ELI5)
.github/workflows/
  eval.yml         -- GitHub Actions CI/CD: unit tests + evaluation matrix (4 models x 5 datasets)
```

## Experimental Design

### Datasets (5 benchmarks, 3,817 total instances)

| Dataset | Task | N | Purpose |
|---------|------|---|---------|
| Natural Questions | Factual QA | 1,000 | Factual extraction from long documents |
| HotpotQA | Multi-hop QA | 500 | Dependency construction across sources |
| TruthfulQA | Truthfulness | 817 | Resistance to plausible misconceptions |
| HaluEval | Hallucination detection | 1,000 | Direct hallucination measurement |
| XSum | Summarization | 500 | Faithful compression |

### Ablation Studies

| Ablation | Configuration | Purpose |
|----------|---------------|---------|
| ETG-NoMultiView | N=1 single view | Multi-view stability importance |
| ETG-NoConstraint | ESBG built, no constraint | Constrained decoding vs. scoring |
| ETG-Threshold-Sweep | tau in {0.5, 0.6, 0.7, 0.8, 0.9} | Precision-recall trade-off |
| ETG-PolicyAblation | Random claim selection | Recursive policy importance |

### Statistical Analysis

- **Paired t-test**: H0 = no hallucination reduction vs. RAG (alpha = 0.05)
- **Cohen's d**: Effect size magnitude (target d > 0.8 = large)
- **Bootstrap CIs**: 95% confidence intervals (10,000 resamples)

## Canonical Evaluation Framework

### Advanced Metrics

| Metric | Description | Reference |
|--------|-------------|-----------|
| FactScore | Atomic claim precision via NLI | Min et al., EMNLP 2023 |
| Citation Precision | Fraction of citations that are valid | Gao et al., ACL 2023 |
| Citation Recall | Fraction of entailed claims with citations | Rashkin et al., ACL 2022 |
| Logic-Step Verification | Step-level accuracy in multi-hop chains | Yang et al., EMNLP 2018 |
| Self-CheckGPT | Zero-resource consistency checking | Manakul et al., EMNLP 2023 |

### Benchmark Models (4 configurations)

| Model | Description | Evidence Source |
|-------|-------------|-----------------|
| Zero-Shot GPT-4 | No retrieval, no verification | Parametric only |
| Standard RAG (Contriever) | Dense retrieval, no verification | Retrieved passages |
| Self-CheckGPT | Stochastic sampling consistency | Self-consistency |
| ETG (Ours) | Multi-view verification + type-checked decoding | Evidence-scoped graph |

### CI/CD Pipeline

The GitHub Actions workflow runs a 4x5 evaluation matrix:
- **4 models**: Zero-Shot, Standard RAG, Self-CheckGPT, ETG
- **5 datasets**: TruthfulQA, HaluEval, HotpotQA, NQ, ELI5
- Produces aggregated reports with markdown tables and LaTeX snippets

## Running Tests

```bash
pytest tests/ -v    # 346 tests
```

## Why This Is Fundamentally New

| Prior approach | Limitation |
|----------------|------------|
| RAG | Retrieval != entailment |
| Self-check | Single-view, gameable |
| Chain-of-thought | Linear, ungrounded |
| Constrained decoding | Syntax-level constraints |
| Knowledge graphs | Static, offline |

ETG introduces:
1. **ESBG**: A dynamic, evidence-scoped belief DAG (not chain-of-thought)
2. **Support mass**: A multi-view stability invariant with exponential filtering
3. **Evidence-Typed Decoding**: Faithfulness as a type constraint, not a reward
4. **Inference-time scaling law**: Provable hallucination suppression via N
5. **Zero-confabulation by construction**: Mechanism design, not behavioral alignment
