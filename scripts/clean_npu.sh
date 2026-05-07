#!/bin/bash
# Clean ONLY our own residual NPU processes. Wait if others are running.
# Usage: bash scripts/clean_npu.sh

CURRENT_PID=$$
PARENT_PID=$PPID
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[clean_npu] Checking NPU processes..."

device_count=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo "0")
if [ "$device_count" -eq 0 ]; then
    echo "[clean_npu] No NPU devices found, skipping."
    exit 0
fi

# Collect all PIDs on NPU cards
all_pids=""
for dev_id in $(seq 0 $((device_count - 1))); do
    pids=$(npu-smi info -t proc-mem -i "$dev_id" -c 0 2>/dev/null \
        | grep -oE 'PID[[:space:]]*:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' || true)
    all_pids="$all_pids $pids"
done
all_pids=$(echo "$all_pids" | tr ' ' '\n' | sort -u | tr '\n' ' ')

foreign_count=0
our_count=0

for pid in $all_pids; do
    [ "$pid" = "$CURRENT_PID" ] && continue
    [ "$pid" = "$PARENT_PID" ] && continue
    [ "$pid" = "1" ] && continue
    [ -z "$pid" ] && continue

    # Check if this PID belongs to our project
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || echo "")
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")

    if echo "$cmdline" | grep -q "$PROJECT_ROOT" || echo "$cwd" | grep -q "$PROJECT_ROOT"; then
        echo "[clean_npu] Killing our residual PID $pid on NPU"
        kill -9 "$pid" 2>/dev/null || true
        our_count=$((our_count + 1))
    else
        foreign_count=$((foreign_count + 1))
    fi
done

if [ $our_count -gt 0 ]; then
    echo "[clean_npu] Killed $our_count of our residual process(es)"
fi

if [ $foreign_count -gt 0 ]; then
    echo "[clean_npu] ⚠️  $foreign_count other process(es) on NPU — waiting..."
    while true; do
        still_foreign=0
        for pid in $all_pids; do
            [ "$pid" = "$CURRENT_PID" ] && continue
            [ "$pid" = "$PARENT_PID" ] && continue
            [ "$pid" = "1" ] && continue
            [ -z "$pid" ] && continue
            cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || echo "")
            cwd=$(readlink /proc/$pid/cwd 2>/dev/null || echo "")
            if ! echo "$cmdline" | grep -q "$PROJECT_ROOT" && ! echo "$cwd" | grep -q "$PROJECT_ROOT"; then
                if [ -d /proc/$pid ]; then
                    still_foreign=$((still_foreign + 1))
                fi
            fi
        done
        if [ $still_foreign -eq 0 ]; then
            echo "[clean_npu] All other processes finished — cards free."
            break
        fi
        sleep 10
    done
else
    echo "[clean_npu] No residual processes found. Devices are clean."
fi
