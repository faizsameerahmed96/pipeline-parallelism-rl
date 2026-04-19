#!/bin/bash
# Run the experiment with two RPC workers on a single machine
# Usage: bash run_experiment.sh [--cuda] [extra args for machine0.py]
#
# Examples:
#   bash run_experiment.sh --cuda
#   bash run_experiment.sh --cuda --total_timesteps 16384
#   bash run_experiment.sh --cuda --gradient_compression_technique none

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs

# Clean up any previous processes
pkill -f "python3.*machine[01].py" 2>/dev/null || true
sleep 1

echo "=== Starting machine1 (worker1) ==="
RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 \
    nohup python3 "$SCRIPT_DIR/machine1.py" > /tmp/machine1.log 2>&1 &
MACHINE1_PID=$!
echo "machine1 PID: $MACHINE1_PID"

sleep 2

echo "=== Starting machine0 (worker0) ==="
RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 \
    WANDB_API_KEY="${WANDB_API_KEY}" \
    python3 "$SCRIPT_DIR/machine0.py" "$@" 2>&1 | tee /tmp/machine0.log

echo "=== machine0 finished, cleaning up ==="
kill $MACHINE1_PID 2>/dev/null || true
wait $MACHINE1_PID 2>/dev/null || true
echo "=== Done ==="
