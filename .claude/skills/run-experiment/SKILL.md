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

### 6. Start training

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff bash run_experiment.sh EXTRA_ARGS"
```

Where `EXTRA_ARGS` are any extra arguments from `$ARGUMENTS` (e.g., `--cuda --total_timesteps 16384`).

If no extra args were provided, default to `--cuda`:
```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff bash run_experiment.sh --cuda"
```

`run_experiment.sh` handles:
- Killing any previous training processes
- Starting machine1 (RPC worker) in background
- Starting machine0 (master) in foreground with `tee` to `/tmp/machine0.log`
- Cleaning up machine1 when machine0 finishes

**IMPORTANT:** Training is complete when you see "Machine shutting down" in the output. machine0 sleeps for 1000 seconds after that message. Once you see it, Ctrl+C the SSH session and proceed to step 7.

If SSH disconnects, check if training is still running:
```bash
ssh -p PORT USER@HOST "pgrep -af 'python3.*machine[01].py'"
```

To view logs after disconnect:
```bash
ssh -p PORT USER@HOST "tail -100 /tmp/machine0.log"
```

### 7. Sync results to local

```bash
rsync -avz -e "ssh -p PORT" USER@HOST:~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/ /Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/
```

Report what was synced (model checkpoints, grad buffers, logs).

### 8. Cleanup

```bash
ssh -p PORT USER@HOST "pkill -f 'python3.*machine[01].py'" 2>/dev/null || true
```

Tell the user:
- Results have been synced to `experiments/exp2-act-grad-acc/runs/`
- W&B metrics available at wandb.ai
- TensorBoard: `tensorboard --logdir experiments/exp2-act-grad-acc/runs/`
