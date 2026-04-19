---
name: run-experiment
description: Run RL training experiment on a remote GPU server via SSH and Docker Compose. Use when asked to run an experiment remotely.
disable-model-invocation: true
allowed-tools:
  - Bash
  - Read
---

# Run Experiment on Remote GPU Server

Run the pipeline-parallelism RL training on a remote GPU server via SSH + Docker Compose, then sync results back.

## Arguments

`$ARGUMENTS` = `user@host` or `user@host:port`

Parse the SSH connection from `$ARGUMENTS`. If port is not specified, default to 22.

## Environment

- WANDB_API_KEY: `7e17edaf69249508fbdf0464123047fd4b4d21ff`
- Repo URL: `https://github.com/faizsameerahmed96/pipeline-parallelism-rl.git`
- Remote clone path: `~/pipeline-parallelism-rl`
- Experiment dir: `experiments/exp2-act-grad-acc`
- Local runs path: `/Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/`

## Step-by-step

### 1. Parse SSH args

Extract user, host, port from `$ARGUMENTS`. Format: `user@host` or `user@host:port`. Default port = 22.

### 2. Verify SSH connectivity

```bash
ssh -o ConnectTimeout=10 -o BatchMode=yes -p PORT USER@HOST "echo CONNECTION_OK"
```

If this fails, stop and tell the user to check SSH config/keys.

### 3. Check remote prerequisites

```bash
ssh -p PORT USER@HOST "docker --version && docker compose version && nvidia-smi"
```

If any fails, report which prerequisite is missing.

### 4. Clone or update repo

```bash
ssh -p PORT USER@HOST "if [ -d ~/pipeline-parallelism-rl ]; then cd ~/pipeline-parallelism-rl && git fetch origin && git reset --hard origin/main; else git clone https://github.com/faizsameerahmed96/pipeline-parallelism-rl.git ~/pipeline-parallelism-rl; fi"
```

### 5. Build and start training

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && export WANDB_API_KEY=7e17edaf69249508fbdf0464123047fd4b4d21ff && docker compose -f docker-compose.gpu.yml up --build -d"
```

Tell the user Docker build may take several minutes on first run (pytorch/cuda base image is ~8GB).

### 6. Monitor training

Stream logs from machine0:

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && docker compose -f docker-compose.gpu.yml logs -f --tail=50 machine0"
```

**IMPORTANT:** Training is complete when you see "Machine shutting down" in the logs. Do NOT wait for the container to exit — machine0 sleeps for 1000 seconds after that message. Once you see it, proceed to step 7.

If log streaming is interrupted, check status:
```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && docker compose -f docker-compose.gpu.yml ps"
```

### 7. Sync results to local

```bash
rsync -avz -e "ssh -p PORT" USER@HOST:~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/ /Users/faizahmed/Documents/SJSU/CS297/pipeline-parallelism-rl/experiments/exp2-act-grad-acc/runs/
```

Report what was synced (model checkpoints, grad buffers, logs).

### 8. Cleanup

```bash
ssh -p PORT USER@HOST "cd ~/pipeline-parallelism-rl/experiments/exp2-act-grad-acc && docker compose -f docker-compose.gpu.yml down"
```

Tell the user:
- Results have been synced to `experiments/exp2-act-grad-acc/runs/`
- W&B metrics available at wandb.ai
- TensorBoard: `tensorboard --logdir experiments/exp2-act-grad-acc/runs/`
