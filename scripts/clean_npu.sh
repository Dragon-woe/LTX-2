#!/bin/bash
# Clean residual processes on all NPU devices.
# Usage: bash scripts/clean_npu.sh

set -e

CURRENT_PID=$$
PARENT_PID=$PPID

echo "[clean_npu] Checking for residual NPU processes..."

device_count=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo "0")
if [ "$device_count" -eq 0 ]; then
    echo "[clean_npu] No NPU devices found, skipping."
    exit 0
fi

killed=0
for dev_id in $(seq 0 $((device_count - 1))); do
    pids=$(npu-smi info -t proc-mem -i "$dev_id" -c 0 2>/dev/null \
        | grep -oE 'PID[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' || true)
    for pid in $pids; do
        if [ "$pid" -ne "$CURRENT_PID" ] && [ "$pid" -ne "$PARENT_PID" ] && [ "$pid" -ne 1 ]; then
            echo "[clean_npu] Killing PID $pid on NPU $dev_id"
            kill -9 "$pid" 2>/dev/null || true
            killed=$((killed + 1))
        fi
    done
done

if [ "$killed" -gt 0 ]; then
    echo "[clean_npu] Killed $killed residual process(es). Waiting 3s for cleanup..."
    sleep 3
else
    echo "[clean_npu] No residual processes found. Devices are clean."
fi
