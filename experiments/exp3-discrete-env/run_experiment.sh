#!/bin/bash
# Run the experiment with two RPC workers on a single machine
# Usage: MASTER_PORT=29500 bash run_experiment.sh [--cuda] [extra args for machine0.py]
#
# Examples:
#   bash run_experiment.sh --cuda
#   bash run_experiment.sh --cuda --total_timesteps 16384
#   MASTER_PORT=29501 bash run_experiment.sh --cuda --gradient_compression_technique none

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p runs

echo "=== Starting machine1 (worker1) on port $MASTER_PORT ==="
RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
    nohup python3 "$SCRIPT_DIR/machine1.py" > /tmp/machine1_${MASTER_PORT}.log 2>&1 &
MACHINE1_PID=$!
echo "machine1 PID: $MACHINE1_PID"

sleep 2

echo "=== Starting machine0 (worker0) on port $MASTER_PORT ==="
RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=$MASTER_PORT \
    WANDB_API_KEY="${WANDB_API_KEY}" \
    python3 "$SCRIPT_DIR/machine0.py" "$@" 2>&1 | tee /tmp/machine0_${MASTER_PORT}.log

echo "=== machine0 finished, cleaning up ==="
kill $MACHINE1_PID 2>/dev/null || true
wait $MACHINE1_PID 2>/dev/null || true
echo "=== Done ==="
