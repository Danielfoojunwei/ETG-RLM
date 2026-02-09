# ETG-RLM

**Evidence-Typed Generation: Faithfulness as a Type System for Recursive Language Models**

A formal framework that treats hallucination control as static type checking over an Evidence-Scoped Belief Graph (ESBG). Claims extracted from model outputs are typed by their multi-view support mass, and only well-typed (evidence-entailed) claims are rendered into the final output.

## Core Concepts

### Evidence-Scoped Belief Graph (ESBG)

A directed acyclic graph `G = (V, ->, pi, sigma)` where each node carries:
- **pi(v)**: an atomic claim extracted from the generated text
- **sigma(v)**: evidence spans in the source corpus that support the claim
- **m(v)**: support mass computed across N independent verification views
- **z(v)**: entailment status (entailed, contradicted, unknown)

### Multi-View Support Mass

N independent verification views each check a claim against the evidence corpus. Views can differ by query rewrite strategy, chunking scheme, retriever model, or verifier prompting.

```
m(c) = (1/N) * sum_{i=1}^{N} 1[z_i = entailed]
```

### Evidence Type System

Claims are typed by their support mass:

| Type | Condition | Renderable |
|------|-----------|------------|
| Verified | `m(c) >= tau` | Yes |
| Uncertain | `tau' < m(c) < tau` | Configurable |
| Unsupported | `m(c) <= tau'` | No |

The type-checker rejects any answer containing Unsupported claims.

### Hallucination Upper Bound (Proposition 1)

If the verifier has false-positive rate alpha per view, any unsupported claim has:

```
Pr[m(c) >= tau] <= exp(-N * D(tau || alpha))
```

where D is KL divergence for Bernoulli distributions. Unsupported claims become **exponentially unlikely** to pass the support-mass gate as N increases.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from etg_rlm import (
    ETGPipeline,
    ETGConfig,
    MultiViewVerifier,
    EvidenceTypeChecker,
    hallucination_upper_bound,
    required_views_for_bound,
)

# Compute theoretical bounds
bound = hallucination_upper_bound(n_views=10, tau=0.7, alpha=0.1)
print(f"Hallucination bound with N=10: {bound:.6f}")

n_needed = required_views_for_bound(target_prob=0.001, tau=0.7, alpha=0.1)
print(f"Views needed for Pr < 0.001: {n_needed}")

# Build a pipeline (bring your own claim extractor, retriever, verifier)
config = ETGConfig(
    tau=0.7,
    tau_prime=0.3,
    verification_budget=50,
    min_views_per_claim=3,
)
pipeline = ETGPipeline(
    claim_extractor=your_extractor,
    views=your_views,
    config=config,
)
result = pipeline.run(query="What is X?", generated_text="X is Y because Z.")
print(f"Verified claims: {len(result.verified_claims)}")
print(f"Rejected claims: {len(result.rejected_claims)}")
print(f"Output: {result.rendered_text}")
```

## Architecture

```
etg_rlm/
  core.py          -- EvidenceSpan, AtomicClaim, ESBGNode, EvidenceScopedBeliefGraph
  verification.py  -- VerificationView, MultiViewVerifier, support mass computation
  type_system.py   -- TypeThresholds, EvidenceTypeChecker (Verified/Uncertain/Unsupported)
  policy.py        -- RecursionPolicy, UtilityWeightedPolicy, GreedyBudgetPolicy
  bounds.py        -- hallucination_upper_bound, required_views_for_bound, optimal_view_allocation
  pipeline.py      -- ETGPipeline, ETGConfig (end-to-end orchestration)
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Contributions

1. **ESBG**: An auditable belief DAG as intermediate representation
2. **Support mass**: A multi-view scalar invariant with exponential filtering guarantees
3. **ETG type system**: Faithfulness as a type constraint, not a reward signal
4. **Inference-time scaling law**: Hallucination probability decreases exponentially with N
