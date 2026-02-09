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
```

## Running Tests

```bash
pytest tests/ -v    # 100 tests
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
