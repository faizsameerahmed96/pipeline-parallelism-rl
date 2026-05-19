# Exp3 — Discrete Environment (Enduro)

## Remote Server

```
ssh root@213.173.108.214 -p 13273
```

## Goal

Validate that the gradient compression technique (accumulate-grads + buffer decay) generalizes to a discrete action space environment.

## Changes from exp2

1. **env.py** — Added `ResizeObservation(env, 96)` to resize Atari 210x160 frames to 96x96 (keeps CNN output at 4096). Added `import ale_py; gym.register_envs(ale_py)` for ALE namespace registration. Added `GrayscaleObservation` + `FrameStackObservation(4)`.
2. **network.py** — Replaced continuous Normal+TanhTransform actor with discrete `Categorical` distribution. Removed `actor_logstd` parameter.
3. **args.py** — Default env_id changed to `ALE/Enduro-v5`, switched to standard Atari PPO hyperparams.
4. **machine0.py** — Action tensor shape changed to `(num_steps, num_envs)` (scalar actions), b_actions flattened with `.long()`. Fixed episode logging for multi-env. Disabled deterministic algorithms.

## Remote Setup

```bash
pip3 install --break-system-packages gymnasium[atari] ale-py AutoROM opencv-python-headless tyro wandb tensorboard
AutoROM --accept-license
```

## Failed Environment Attempts (SpaceInvaders, Breakout)

Tried SpaceInvaders and Breakout before settling on Enduro. Neither converged well with the pipeline-parallel setup.

| Env | Exp | LR | num_envs | Best Avg | Issue |
|---|---|---|---|---|---|
| SpaceInvaders | 1 | 4e-5 | 1 | ~200 | Too slow (49 SPS), flat returns |
| SpaceInvaders | 2 | 2.5e-4 | 8 | ~279 | Peaked then declined — LR too high |
| SpaceInvaders | 3 | 1e-4 | 8 | ~260 | Stable plateau, no further improvement |
| SpaceInvaders | 4 | 1e-4 | 32 | ~180 | Slower learning, larger minibatches |
| Breakout | 5 | 1e-4 | 8 | ~2.5 | LR too low |
| Breakout | 6 | 2.5e-4 | 8 | ~4.6 | Far below benchmark (~100-400) |

**Conclusion:** SpaceInvaders plateaued at ~260 (benchmark ~1189). Breakout barely learned at all. The RPC overhead appears to degrade gradient quality too much for these environments. Switched to Enduro for its dense reward signal.

## Current: Enduro Baseline

**Environment:** ALE/Enduro-v5 — driving game, Discrete(9) actions, dense rewards (+1 per car passed).

**Hyperparams:**

| Param | Value |
|---|---|
| learning_rate | 2.5e-4 |
| num_envs | 8 |
| num_steps | 128 |
| num_minibatches | 4 |
| update_epochs | 4 |
| clip_coef | 0.1 |
| ent_coef | 0.01 |
| total_timesteps | 10M |
| save_model_freq | 100 |

### Experiment 7 — Enduro Baseline (lr=2.5e-4, 8 envs)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1778979097` | no compression, 10M steps, lr=2.5e-4, num_envs=8 | **COMPLETED** | root@213.173.108.214:13273 | 29500 |

W&B: https://wandb.ai/faizsameerahmed96-san-jose-state-university/data-paralellism-rl-2026-02/runs/f2i9e8ie

Logs:
```bash
ssh -p 13273 root@213.173.108.214 "tail -20 /tmp/machine0_29500.log"
```

**Progress (avg return over 40 episodes):**

| Steps | Avg Return |
|-------|-----------|
| 0.5M | 9 |
| 1.0M | 104 |
| 1.5M | 148 |
| 2.0M | 217 |
| 2.5M | 254 |
| 3.0M | 239 |
| 3.5M | 326 |
| 4.0M | 292 |
| 4.5M | 337 |
| 4.8M | 376 |
| 5.3M | 391 |
| 6.0M | 416 |
| 6.5M | 427 |
| 7.6M | 427 |
| 8.4M | 491 |
| 8.7M | 472 |
| 9.5M | 500 |
| 9.8M | 494 |

**Peak: ~500 avg return at 9.5M steps. COMPLETED 10M steps.**

**Note:** First run crashed at iter 4151/9765 due to disk quota (checkpoints saved every 10 iters filled 94GB). Fixed by deleting old runs and changing save_model_freq to 100. Restarted fresh.

### Experiment 8 — Enduro with Compression (90p, 0.99 decay)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| TBD | accumulate-grads, 90th percentile, 0.99 decay, 10M steps, lr=2.5e-4, num_envs=8, warm_start=500k | **COMPLETED** | root@213.173.108.214:13273 | 29501 |

Logs:
```bash
ssh -p 13273 root@213.173.108.214 "tail -20 /tmp/machine0_29501.log"
```

Same hyperparams as Exp 7 baseline, but with gradient compression enabled. Warm start at 500k steps (compression activates ~iter 488). Running in parallel on port 29501.

**Note:** First attempt crashed at iter 31 due to variable shadowing bug (`values` in sparse reconstruction overwrote rollout buffer). Fixed by renaming to `sparse_values`/`sparse_indices`.

**Progress (avg return over 40 episodes):**

| Steps | Avg Return | Note |
|-------|-----------|------|
| 0.5M | 9 | warm start (full grads) |
| 1.0M | 104 | compression active |
| 1.5M | 147 | |
| 2.0M | 217 | |
| 2.5M | 254 | |
| 2.8M | 309 | peak |
| 3.0M | 239 | |
| 3.2M | 271 | |
| 3.4M | 274 | |
| 3.5M | 265 | |
| 3.7M | 244 | |
| 3.9M | 54 | **COLLAPSE** |
| 4.0M | 34 | |
| 4.1M | 23 | |
| 4.3M | 17 | |
| 4.5M | 39 | |
| 4.7M | 48 | |
| 5.3M | 150 | recovering |
| 5.9M | 249 | |
| 6.5M | 316 | |
| 6.7M | 334 | 2nd peak |
| 6.9M | 262 | oscillating |
| 7.5M | 270 | |
| 7.9M | 330 | |
| 8.4M | 131 | 2nd dip |
| 8.9M | 270 | |
| 9.3M | 352 | 3rd peak |
| 9.5M | 241 | |
| 9.8M | 150 | |
| 9.9M | 174 | final |

**COMPLETED.** Policy collapsed at ~3.8M, recovered, then oscillated between 130-350 for the rest of training. Never stabilized or matched baseline.

**Comparison:**

| | Baseline (Exp 7) | Compression (Exp 8) |
|---|---|---|
| Peak avg return | **500** | **352** |
| Final avg return | **494** | **174** |
| Stability | Steady climb, stable at 400-500 | Volatile — collapse at 3.8M, oscillations throughout |

**Conclusion:** 0.99 decay + 90p compression degrades performance significantly on Enduro. The accumulation buffer drifts from true gradients, causing periodic collapses. Root cause: 0.99 decay is applied only 16× per iteration (vs 320× in CarRacing), so 85% of old values remain (vs 4%). Stale gradients accumulate and corrupt CNN updates.

### Experiment 9 — Enduro with Compression (90p, 0.82 decay)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| TBD | accumulate-grads, 90th percentile, 0.82 decay, 10M steps, lr=2.5e-4, num_envs=8, warm_start=500k | **KILLED** | root@213.173.108.214:13273 | 29500 |

Decay of 0.82 chosen to match CarRacing's effective decay: 0.82^16 ≈ 0.04 (4% remaining after 1 iteration), same as 0.99^320 in CarRacing. **Result:** 0 returns after 2.5M steps — decay too aggressive, gradients wiped out before they could have any effect. Killed.

### Experiment 10 — Enduro with Compression (90p, 0.95 decay)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| TBD | accumulate-grads, 90th percentile, 0.95 decay, 10M steps, lr=2.5e-4, num_envs=8, warm_start=500k | **KILLED** | root@213.173.108.214:13273 | 29500 |

0.95^16 ≈ 0.44 (44% remaining after 1 iteration). **Result:** 0 returns after 1.6M steps — decay still too aggressive. Killed.

### Experiment 11 — Enduro with Compression (90p, 0.98 decay)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| TBD | accumulate-grads, 90th percentile, 0.98 decay, 10M steps, lr=2.5e-4, num_envs=8, warm_start=500k | **RUNNING** | root@213.173.108.214:13273 | 29502 |

Logs:
```bash
ssh -p 13273 root@213.173.108.214 "tail -20 /tmp/machine0_29502.log"
```

0.98^16 ≈ 0.72 (72% remaining after 1 iteration). Closer to 0.99 (85%) which learned but collapsed.

**Decay search summary:**

| Decay | Remaining/iter | Result |
|-------|---------------|--------|
| 0.99 | 85% | Learned (peak 352), collapsed at 3.8M |
| 0.98 | 72% | Running |
| 0.95 | 44% | No learning |
| 0.82 | 4% | No learning |
