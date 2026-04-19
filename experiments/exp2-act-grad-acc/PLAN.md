# Experiment 2: Gradient Accumulation Compression — Plan

## Proven Config

`num_envs=1, num_steps=4096, lr=0.00004, num_minibatches=32, total_timesteps=1,000,000` (244 iterations)

## Experiment 1 — Baseline

| Run ID | Config | Status | Server |
|---|---|---|---|
| `1776624639` | No compression | **RUNNING** | root@213.192.2.109:40164 |

Logs (started with old script):
```bash
ssh -p 40164 root@213.192.2.109 "tail -20 /tmp/training.log"
```

## Experiment 2 — Gradient Compression (accumulate-grads, 90p, decay=1.0)

| Run ID | Config | Status | Server |
|---|---|---|---|
| TBD | accumulate-grads, 90p, decay=1.0 | PLANNED | — |

Logs (new script uses port-based log files):
```bash
# machine0 (master)
ssh -p PORT root@HOST "tail -20 /tmp/machine0_MASTER_PORT.log"

# machine1 (RPC worker)
ssh -p PORT root@HOST "tail -20 /tmp/machine1_MASTER_PORT.log"
```

## General Tracking

```bash
# Check processes
ssh -p 40164 root@213.192.2.109 "pgrep -af 'python3.*machine[01].py'"

# Sync results locally
rsync -avz -e "ssh -p 40164" root@213.192.2.109:/workspace/runs/ experiments/exp2-act-grad-acc/runs/
```

W&B: https://wandb.ai/faizsameerahmed96-san-jose-state-university/data-paralellism-rl-2026-02
