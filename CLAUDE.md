# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS297/CS298 research project at SJSU. The goal is to research strategies to reduce the amount of data transferred between machines during distributed RL training, without significantly degrading policy performance.

## Environment Setup

- **Python:** 3.11 (see `.python-version`)
- **Package manager:** `uv`
- **Install dependencies:** `uv sync`

## Running Experiments
The active experiment is in `experiments/exp2-act-grad-acc/`. Training requires two machines communicating via PyTorch RPC.

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

### Gradient Compression

Three modes controlled by `--gradient_compression_technique`:
1. **`none`**: Full feature gradients sent from Machine 1 → Machine 0.
2. **`stats-based`**: Filters gradients by percentile threshold.
3. **`accumulate-grads`**: Accumulates gradients across minibatches; only sends values above `--accumulate_grads_percentile` threshold in sparse format `{indices, values, shape}`. Compression starts after `--warm_start_steps` (default: 30,000).

Machine 0 reconstructs the sparse tensor before the CNN backward pass.

### Algorithm

Training uses PPO (Proximal Policy Optimization). Hyper-parameters are configurable via CLI with `tyro`.

### Checkpointing

Machine 0 saves/loads checkpoints for both CNN and ActorCritic networks separately. Both must be from the same iteration. Checkpoints stored under `runs/`.

## Experiment Comparison

`exp1-cloud-setup/` — initial baseline setup
`exp2-act-grad-acc/` — current experiment with gradient accumulation compression
