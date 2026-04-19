# CLAUDE.md — exp2-act-grad-acc

## Experiments

Always check `PLAN.md` first before running experiments. It contains the current runs, their status, and how to check logs. Update it when starting or completing a run.

## Running Experiments Remotely

Training requires a GPU server (tested on RTX 3090). Remote servers are rented and change frequently — the SSH address/port will differ each time.

### Setup a new server

1. `setup_remote.sh` — installs system deps (swig, build-essential) and all Python packages. Safe to re-run.
2. The remote is typically a Docker container with PyTorch pre-installed, so `pip install --break-system-packages` is used.
3. Clone or update the repo: `git clone https://github.com/faizsameerahmed96/pipeline-parallelism-rl.git`

### Start an experiment

`run_experiment.sh` launches both RPC workers (machine0 + machine1) on a single machine. All `args.py` fields can be overridden via CLI flags.

```bash
MASTER_PORT=29500 WANDB_API_KEY=... bash run_experiment.sh --cuda --gradient_compression_technique none
```

- `MASTER_PORT` defaults to 29500. Use different ports to run multiple experiments concurrently on the same machine.
- Logs go to `/tmp/machine0_PORT.log` and `/tmp/machine1_PORT.log`.
- Run detached with `nohup ... &` so it survives SSH disconnects.

### Syncing results

Runs are saved to `/workspace/runs/` on the remote (not `~/pipeline-parallelism-rl/.../runs/`). Rsync is needed to get results locally:

```bash
rsync -avz -e "ssh -p PORT" root@HOST:/workspace/runs/ experiments/exp2-act-grad-acc/runs/
```

Set up a background loop to sync every 30 seconds during training.

### W&B

All runs log to: https://wandb.ai/faizsameerahmed96-san-jose-state-university/data-paralellism-rl-2026-02

WANDB_API_KEY: `7e17edaf69249508fbdf0464123047fd4b4d21ff`

## Key Learnings

- Only `num_envs=1, num_steps=4096` has been proven to work well. Other num_envs/num_steps combos underperformed — do not change without re-tuning.
- Decay buffer is applied `num_steps/minibatch_size * update_epochs` times per iteration (e.g., 320x). Even 0.99 decay is aggressive.
- `warm_start_steps=30000` — gradient compression only activates after this many steps.
