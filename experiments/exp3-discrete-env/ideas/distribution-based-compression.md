# Distribution-Based Gradient Compression

## Background: Dirac's Delta Function

### What is it?

The Dirac delta function δ(x) represents a point mass — all probability concentrated at a single point. It's not a regular function but a distribution (in the mathematical sense):

```
δ(x) = ∞   when x = 0
δ(x) = 0   when x ≠ 0
∫ δ(x) dx = 1
```

Think of it as: "I observed exactly this value, with 100% certainty."

### Intuition

Imagine a dartboard:
- A **normal distribution** N(μ, σ²) says "darts land in a spread around μ"
- A **Dirac delta** δ(x - g) says "the dart landed exactly at g"

The question becomes: **how surprising is it that the dart landed at g, given what we expected?**

### Measuring Surprise with KL Divergence

KL divergence measures how different two distributions are. Between a Dirac delta (observation) and a normal distribution (expectation):

```
KL(δ(g) || N(μ, σ²)) = -log(σ) + (g - μ)² / (2σ²) + const
```

The dominant term is `(g - μ)² / σ²` — the **z-score squared**. This tells us:

| Scenario | z-score | Surprise | Meaning |
|---|---|---|---|
| g ≈ μ, any σ | Low | Low | Observed value matches expectation |
| g far from μ, large σ | Medium | Medium | Unusual but within expected variance |
| g far from μ, small σ | High | High | This dimension was stable, now it changed |

### Application to Gradient Compression

Each gradient value g[i] is an observation (Dirac delta). The running stats μ[i], σ[i] are the expected distribution. The z-score tells us:

- **Low z:** This gradient is business-as-usual for this dimension. Safe to accumulate/skip.
- **High z:** Something changed at this dimension. The accumulated buffer is likely stale — flush it and send the fresh value.

The key insight is that surprise is **relative**. A small gradient in a usually-stable dimension is more informative than a large gradient in a noisy dimension.

## Problem with Current Approach

The current `accumulate-grads` method uses magnitude-based thresholding (90th percentile). This has two weaknesses:

1. A gradient of 0.01 in a dimension that's always ~0.01 is **not informative** but gets sent if above threshold
2. A gradient of 0.001 in a dimension that's usually ~0.1 **is informative** (something changed) but gets skipped

Magnitude alone doesn't capture whether a gradient carries new information.

## Idea: Information-Theoretic Filtering

Maintain a running distribution (μ, σ²) per gradient dimension. Use the surprise (z-score) to decide what to send.

### Algorithm

```
For each gradient update:
  1. Compute z-score per dimension: z[i] = |g[i] - μ[i]| / σ[i]
  2. Send only where z[i] > threshold (surprising/informative)
  3. Receiver reconstructs unsent dimensions using μ[i] (not zero)
  4. Update running stats: μ ← αμ + (1-α)g, σ² ← ασ² + (1-α)(g-μ)²
```

### Why Dirac's Delta

Each observed gradient is a point mass δ(g). The KL divergence between δ(g) and the prior N(μ,σ²) is:

```
KL(δ(g) || N(μ,σ²)) ∝ (g - μ)² / σ²
```

This is the z-score squared. High z = high information content = worth sending.

### Key Differences from Current Approach

| | Magnitude-based (current) | Distribution-based (proposed) |
|---|---|---|
| What to send | Top 10% by |grad| | Gradients with z > threshold |
| Unsent values | Receiver uses 0 | Receiver uses μ (better approx) |
| Adapts to patterns | No | Yes — learns per-dimension norms |
| Compression ratio | Fixed (always 10%) | Variable (depends on how surprising the batch is) |

## Hybrid Approach: Accumulate + Surprise Circuit Breaker

Keep the proven accumulate-grads method but add distribution-based surprise detection as a "circuit breaker."

### How it works

1. **Accumulate gradients as normal** — buffer collects gradients, sends top 10% by magnitude
2. **Maintain per-dimension stats in parallel** — track μ, σ² with EMA (α ≈ 0.999 for ~3 iteration window)
3. **On each gradient update, check for surprise** — compute z-score per dimension
4. **If z > threshold at a dimension:**
   - Clear that dimension's accumulation buffer (it's stale)
   - Send the raw gradient value for that dimension immediately
5. **Otherwise:** continue accumulating as normal

### Why this helps

The current accumulate-grads degrades because the buffer accumulates stale or wrong-direction gradients. By the time they cross the magnitude threshold, the signal is outdated. The surprise detector catches direction changes early and flushes the stale value.

### Compression behavior

- **Stable training phases:** Few surprises → compression ratio stays near 10% (same as current)
- **Rapid policy changes:** Many surprises → more data sent temporarily, but this is correct — the model is changing fast and needs fresh gradients
- Adaptive compression that matches the training dynamics

### Potential experiment

```
--gradient_compression_technique accumulate-grads-surprise
--accumulate_grads_percentile 0.90
--surprise_z_threshold 2.0
--surprise_ema_alpha 0.999
--warm_start_steps 30000
```

Compare against baseline and plain accumulate-grads on:
- Episodic return curve
- Total data transferred
- Compression ratio over time (should be variable — high compression when stable, low when policy shifts)

## Open Questions

- What z-threshold gives comparable compression to 90p? (z=1.28 ≈ 90th percentile of normal)
- How should α (EMA decay for stats) be set? Too fast = noisy estimates, too slow = stale
- Should we update stats with ALL gradients or only the sent ones?
- Does variable compression ratio cause issues? Could enforce a fixed budget
- How to handle early training when σ estimates are unreliable? (warm start?)
- The running mean needs to be shared: machine1 computes it, machine0 needs it for reconstruction. This adds communication overhead — is it worth it?
