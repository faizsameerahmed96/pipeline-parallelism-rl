# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS297/CS298 research project at SJSU implementing distributed pipeline-parallel PPO (Proximal Policy Optimization) training with gradient compression. The core research question is how gradient compression affects communication bandwidth vs. training performance in a two-machine distributed RL setup.

## Environment Setup

- **Python:** 3.11 (see `.python-version`)
- **Package manager:** `uv`
- **Install dependencies:** `uv sync`

## Running Experiments

The active experiment is in `experiments/exp2-act-grad-acc/`. Training requires two machines communicating via PyTorch RPC.

**With Docker Compose (recommended):**
```bash
cd experiments/exp2-act-grad-acc
WANDB_API_KEY=<key> docker-compose up
```

**Without Docker (two terminals / two machines):**
```bash
# Machine 1 first (worker/remote):
python experiments/exp2-act-grad-acc/machine1.py

# Machine 0 (master/actor):
python experiments/exp2-act-grad-acc/machine0.py \
  --env_id CarRacing-v3 \
  --gradient_compression_technique accumulate-grads \
  --accumulate_grads_percentile 0.99 \
  --total_timesteps 500000
```

**Generating report plots:**
```bash
python report/generate_plots.py
```

## Architecture

### Distributed Design

Two machines communicate via PyTorch RPC (`torch.distributed.rpc`):

- **Machine 0** (`machine0.py`): Master node. Runs the `CNNNetwork` (feature extractor), interacts with the environment, orchestrates the training loop, and calls remote methods on Machine 1.
- **Machine 1** (`machine1.py`): Worker node. Hosts the `ActorCriticNetwork` (policy + value heads). Receives CNN features, computes forward pass, and performs backward + optimizer step via RPC.

RPC calls use the pattern:
```python
_remote_method(ActorCriticNetwork.backward_and_step, remote_rref, features, ...)
```

### Network Architecture (`network.py`)

- **`CNNNetwork`**: Conv2d layers → 4096-dim feature vector. Runs on Machine 0.
- **`ActorCriticNetwork`**: Linear layers → actor head (continuous actions via tanh-squashed Normal) + critic head (state value). Runs on Machine 1.

### Gradient Compression

Three modes controlled by `--gradient_compression_technique`:
1. **`none`**: Full feature gradients sent from Machine 1 → Machine 0.
2. **`stats-based`**: Filters gradients by percentile threshold.
3. **`accumulate-grads`**: Accumulates gradients across minibatches; only sends values above `--accumulate_grads_percentile` threshold in sparse format `{indices, values, shape}`. Compression starts after `--warm_start_steps` (default: 30,000).

Machine 0 reconstructs the sparse tensor before the CNN backward pass.

### Algorithm: PPO

Key hyperparameters (configurable via CLI with `tyro`):
- `total_timesteps`: 500,000
- `learning_rate`: 5e-5
- `num_steps`: 128 (rollout length per env)
- `num_envs`: 32
- `update_epochs`: 10
- `batch_size`: num_envs × num_steps = 4,096

### Environment (`env.py`)

Wraps `CarRacing-v3` from Gymnasium with:
- Grayscale conversion
- Frame stacking (4 frames)
- Observation normalization to [0, 1]

### Checkpointing

Machine 0 saves/loads checkpoints for both CNN and ActorCritic networks separately. Both must be from the same iteration. Checkpoints stored under `runs/`.

## Experiment Comparison

`exp1-cloud-setup/` — initial baseline setup
`exp2-act-grad-acc/` — current experiment with gradient accumulation compression
