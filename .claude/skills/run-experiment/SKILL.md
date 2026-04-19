---
name: run-experiment
description: Run RL training experiment on a remote GPU server via SSH. Use when asked to run an experiment remotely.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
---

# Run Experiment on Remote GPU Server

Run the pipeline-parallelism RL training on a remote GPU server via SSH, then sync results back.

## Arguments

`$ARGUMENTS` = `user@host:port [extra args]`

Parse the SSH connection from `$ARGUMENTS`. If port is not specified, default to 22. Any additional arguments after the SSH target are passed to `run_experiment.sh` (and forwarded to `machine0.py`).

Examples:
- `/run-experiment root@gpu-server:40101`
- `/run-experiment root@gpu-server:40101 --cuda --total_timesteps 16384`
- `/run-experiment root@gpu-server:40101 --cuda --gradient_compression_technique none`

## Environment

- WANDB_API_KEY: `7e17edaf69249508fbdf0464123047fd4b4d21ff`
- Repo URL: `https://github.com/faizsameerahmed96/pipeline-parallelism-rl.git`
- Remote clone path: `~/pipeline-parallelism-rl`
- Experiment dir: `experiments/exp2-act-grad-acc`
- Local runs path: `/Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/`

## Step-by-step

### 1. Parse SSH args

Extract user, host, port from `$ARGUMENTS`. Format: `user@host` or `user@host:port`. Default port = 22. Everything after the SSH target is `EXTRA_ARGS`.

### 2. Verify SSH connectivity

```bash
ssh -o ConnectTimeout=10 -o BatchMode=yes -p PORT USER@HOST "echo CONNECTION_OK"
```

If this fails, stop and tell the user to check SSH config/keys.

### 3. Check remote GPU

```bash
ssh -p PORT USER@HOST "nvidia-smi && python3 -c 'import torch; print(torch.cuda.is_available())'"
```

If `nvidia-smi` fails, warn user. If torch is missing, setup will install it.

### 4. Clone or update repo

```bash
ssh -p PORT USER@HOST "if [ -d ~/pipeline-parallelism-rl ]; then cd ~/pipeline-parallelism-rl && git fetch origin && git reset --hard origin/main; else git clone https://github.com/faizsameerahmed96/pipeline-parallelism-rl.git ~/pipeline-parallelism-rl; fi"
```

### 5. Run setup script (first time or if deps are missing)

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && bash setup_remote.sh"
```

This installs system deps (`swig`, `build-essential`) and all Python packages. Safe to re-run — pip will skip already-installed packages.

### 6. Start training (detached) and background sync

Start training detached on the remote so it survives SSH disconnects:

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && nohup bash -c 'WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff bash run_experiment.sh EXTRA_ARGS' > /tmp/training.log 2>&1 &"
```

If no extra args were provided, default to `--cuda`:
```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && nohup bash -c 'WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff bash run_experiment.sh --cuda' > /tmp/training.log 2>&1 &"
```

Wait a few seconds, then verify training started:
```bash
ssh -p PORT USER@HOST "pgrep -af 'python3.*machine[01].py'"
```

Start a **background rsync loop** that syncs results every 30 seconds:

```bash
while true; do rsync -avz --quiet -e "ssh -p PORT" USER@HOST:~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/ /Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/ 2>/dev/null; sleep 30; done
```

Run this with `run_in_background: true` so it syncs continuously while training runs.

### 7. Monitor training

Tail the remote log to watch progress:

```bash
ssh -p PORT USER@HOST "tail -f /tmp/training.log"
```

**IMPORTANT:** Training is complete when you see "Machine shutting down" in the logs. machine0 sleeps for 1000 seconds after that message. Once you see it, proceed to step 8.

If SSH disconnects, check if training is still running:
```bash
ssh -p PORT USER@HOST "pgrep -af 'python3.*machine[01].py'"
```

To view recent logs after disconnect:
```bash
ssh -p PORT USER@HOST "tail -100 /tmp/training.log"
```

### 8. Cleanup

Once training is complete:

1. Kill the background rsync loop (stop the background Bash command).
2. Do one final rsync to ensure everything is synced:
```bash
rsync -avz -e "ssh -p PORT" USER@HOST:~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/ /Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/
```
3. Kill remote training processes:
```bash
ssh -p PORT USER@HOST "pkill -f 'python3.*machine[01].py'" 2>/dev/null || true
```

Tell the user:
- Results have been synced to `experiments/exp2-act-grad-acc/runs/`
- W&B metrics available at wandb.ai
- TensorBoard: `tensorboard --logdir experiments/exp2-act-grad-acc/runs/`
