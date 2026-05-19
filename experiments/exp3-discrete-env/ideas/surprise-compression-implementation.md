# Implementation: Surprise-Based Gradient Compression

## Overview

A new gradient compression technique (`--gradient_compression_technique surprise`) that selects which gradients to send based on statistical surprise (z-score) rather than magnitude. Unsent positions are reconstructed by sampling from the learned per-dimension distribution instead of using zeros.

## How It Works

### Per-Dimension Running Stats

We maintain running mean μ and variance σ² per feature dimension (shape `(4096,)`) using exponential moving average (EMA):

```
μ_new = α * μ_old + (1 - α) * grad_per_dim
σ²_new = α * σ²_old + (1 - α) * (grad_per_dim - μ_old) * (grad_per_dim - μ_new)
```

Where `grad_per_dim = feature_grads.mean(dim=0)` averages across the batch dimension first. This is necessary because minibatch indices are shuffled each iteration — per-element stats across minibatches would track different samples and be meaningless.

The stats are `(4096,)` each — only 32KB, cheap to sync.

### Z-Score Selection

For each element in the full `(minibatch_size, 4096)` gradient tensor, compute surprise using the per-dimension stats broadcast across the batch:

```
z[i, j] = |grad[i, j] - μ[j]| / σ[j]
```

Select the top N% elements by z-score (default 10%, controlled by `--surprise_compress_percentile 0.90`). Send these as sparse tensors `{indices, values, shape}` — same format as the existing `accumulate-grads` technique.

### Why Z-Score Over Magnitude

Magnitude-based selection (used in `accumulate-grads`) has two weaknesses:

1. A gradient of 0.01 in a dimension that's always ~0.01 is **not informative** but gets sent if above threshold
2. A gradient of 0.001 in a dimension that's usually ~0.1 **is informative** (something changed) but gets skipped

Z-score captures whether a gradient carries *new information* relative to what that dimension usually sees.

### Reconstruction on Machine 0

Instead of zero-filling unsent positions (like `accumulate-grads`), machine 0 reconstructs by:

1. Sample full tensor from `N(μ[j], σ[j])` — shape `(minibatch_size, 4096)`
2. Overwrite the 10% positions where actual gradient values were received

This gives unsent dimensions a reasonable approximation rather than zero, while sent dimensions get exact values.

### Stats Sync

Machine 1 (which computes gradients) is the authority on running stats. Machine 0 needs them for reconstruction. Stats are synced by piggybacking on the return dict every `surprise_sync_interval` minibatches:

- First compressed minibatch always syncs (counter starts at 0)
- Default interval is 10, but `--surprise_sync_interval 1` sends every minibatch
- Bandwidth overhead: 8,192 floats (2 × 4096) per sync — negligible vs ~52K gradient values

### Warm Start

During the warm start period (`--warm_start_steps 30000`):
- Full gradients are sent (no compression)
- Running stats are accumulated on machine 1 in the background
- When compression activates, stats are well-initialized

### Connection to Information Theory

The ideas doc (`distribution-based-compression.md`) motivates this approach using Dirac's delta and KL divergence. Each observed gradient is a point mass δ(g), and the KL divergence against the running distribution N(μ, σ²) is:

```
KL(δ(g) || N(μ, σ²)) = -log(σ) + (g - μ)² / (2σ²) + const
```

The dominant term is `(g - μ)² / σ²` — the z-score squared. We use the z-score directly rather than the full KL. The additional `-log(σ)` term would add a constant bias per dimension favoring low-variance dimensions, but the z-score already captures most of this effect since small σ amplifies any deviation.

## Files Changed

### `args.py`

New arguments:
- `surprise_compress_percentile: float = 0.90` — send top 10% most surprising by z-score
- `surprise_compress_ema_alpha: float = 0.5` — EMA weight on old stats (0.5 = equal weight history vs current)
- `surprise_sync_interval: int = 10` — sync stats to machine 0 every N minibatches

### `network.py`

- Added `running_mean` and `running_var` state to `ActorCriticNetwork.__init__`
- Added warm-start stats accumulation block before technique branches
- Added `elif surprise_compress:` branch in `backward_and_step`:
  - Computes z-scores on full `(minibatch, 4096)` tensor
  - Selects top N% elements, creates sparse dict with `'technique': 'surprise'`
  - Periodically includes `running_mean` and `running_std` for sync
  - Updates running stats with EMA
  - Logs z-score percentiles (p25, p50, p90, p99) via `buffer_stats`

### `machine0.py`

- Added `local_running_mean`, `local_running_std`, `surprise_minibatch_counter` state
- Added `use_surprise` and `surprise_warmup` flags
- Extended RPC call with surprise params
- Added surprise reconstruction branch: samples `N(μ, σ)`, overwrites sent positions
- Generified `buffer_stats` logging to handle both accumulate-grads and surprise keys

## Usage

```bash
MASTER_PORT=29500 WANDB_API_KEY=... bash run_experiment.sh \
  --cuda \
  --gradient_compression_technique surprise \
  --surprise_compress_percentile 0.90 \
  --surprise_compress_ema_alpha 0.5 \
  --surprise_sync_interval 1 \
  --warm_start_steps 30000
```

## Experiment 7 Results

Run `1777156828` on RTX 3090, stopped at iteration 148/244 (~604K steps).

- Warm start completed normally, compression activated at 30K steps
- Returns started climbing around 100K steps, reaching peaks of 373 by 600K steps
- Still showing high variance in episodic returns at time of stopping
- W&B: https://wandb.ai/faizsameerahmed96-san-jose-state-university/data-paralellism-rl-2026-02/runs/2hb78tlq
