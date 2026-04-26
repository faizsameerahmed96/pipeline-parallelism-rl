
# Experiment 2: Gradient Accumulation Compression — Plan

## Proven Config

`num_envs=1, num_steps=4096, lr=0.00004, num_minibatches=32, total_timesteps=1,000,000` (244 iterations)

## Experiment 1 — Baseline

| Run ID | Config | Status | Server |
|---|---|---|---|
| `1776624639` | No compression | **CRASHED iter 169/244 (NaN in policy)** | root@213.192.2.109:40164 |

Logs (started with old script):
```bash
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/training.log"
```

## Experiment 2 — Gradient Compression (accumulate-grads, 90p, decay=1.0)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1776628684` | accumulate-grads, 90p, decay=1.0 | **CRASHED iter 172/244 (CUDA TensorPipe error)** | root@213.192.2.109:40164 | 29501 |

Logs:
```bash
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/machine0_29501.log"
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/machine1_29501.log"
```

## Experiment 3 — Gradient Compression with Decay (accumulate-grads, 90p, decay=0.99)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1776629538` | accumulate-grads, 90p, decay=0.99 | OOM at iter 10 | root@213.192.2.109:40164 | 29502 |
| `1776636519` | accumulate-grads, 90p, decay=0.99 | **STOPPED at iter 121/244** | root@213.192.2.109:40164 | 29502 |

Logs:
```bash
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/machine0_29502.log"
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/machine1_29502.log"
```

## Experiment 4 — Baseline (750k steps)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1776981725` | No compression, total_timesteps=750000 | **COMPLETED** | root@213.192.2.86:40092 | 29500 |

Logs:
```bash
ssh -p 40092 root@213.192.2.86 "tail -20 /tmp/machine0_29500.log"
```

## Experiment 5 — Gradient Compression (accumulate-grads, 90p, decay=1.0, 750k steps)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1776998261` | accumulate-grads, 90p, decay=1.0, total_timesteps=750000 | **COMPLETED** | root@213.192.2.86:40092 | 29500 |

Logs:
```bash
ssh -p 40092 root@213.192.2.86 "tail -20 /tmp/machine0_29500.log"
```

## Experiment 6 — Gradient Compression with Decay (accumulate-grads, 90p, decay=0.99, 750k steps)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1777001996` | accumulate-grads, 90p, decay=0.99, total_timesteps=750000 | **COMPLETED** | root@213.192.2.86:40092 | 29501 |

Logs:
```bash
ssh -p 40092 root@213.192.2.86 "tail -20 /tmp/machine0_29501.log"
```

## Experiment 7 — Surprise Compression (90p z-score, sync every minibatch)

| Run ID | Config | Status | Server | Port |
|---|---|---|---|---|
| `1777156828` | surprise, 90p z-score, ema=0.5, sync=1 | **STOPPED at iter 148/244** | root@213.192.2.120:40178 | 29500 |

Surprise hyperparams:
- `--gradient_compression_technique surprise`
- `--surprise_compress_percentile 0.90` (send top 10% most surprising by z-score)
- `--surprise_compress_ema_alpha 0.5` (equal weight history vs current gradient)
- `--surprise_sync_interval 1` (sync mean/std to machine0 every minibatch)
- `--warm_start_steps 30000` (build up stats before compressing)

```bash
MASTER_PORT=29500 WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff nohup bash run_experiment.sh \
  --cuda \
  --gradient_compression_technique surprise \
  --surprise_compress_percentile 0.90 \
  --surprise_compress_ema_alpha 0.5 \
  --surprise_sync_interval 1 \
  --warm_start_steps 30000 \
  > /dev/null 2>&1 &
```

Logs:
```bash
ssh -p 40178 root@213.192.2.120 "tail -20 /tmp/machine0_29500.log"
ssh -p 40178 root@213.192.2.120 "tail -20 /tmp/machine1_29500.log"
```

## General Tracking

```bash
# Check processes
ssh -p 40092 root@213.192.2.86 "pgrep -af 'python3.*machine[01].py'"

# Sync results locally
rsync -avz -e "ssh -p 40092" root@213.192.2.86:/workspace/runs/ experiments/exp2-act-grad-acc/runs/
```

W&B: https://wandb.ai/faizsameerahmed96-san-jose-state-university/data-paralellism-rl-2026-02
