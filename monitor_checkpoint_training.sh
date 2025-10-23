#!/bin/bash

# Monitor training progress for all checkpoints

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              Training All Checkpoints - Progress Monitor                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is in progress
if ! ps aux | grep -q "[t]rain_all_checkpoints.sh"; then
    echo "⚠️  No training process running"
    echo ""
else
    echo "✅ Training process is running"
    echo ""
fi

# Count total configurations
TOTAL=20

# Count existing checkpoints
HORIZONS=(96 192 336 720)
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")
DATASET="ETTh1"

EXISTING=0
for HORIZON in "${HORIZONS[@]}"; do
    for MODEL in "${BACKBONES[@]}"; do
        if [ -f "./checkpoints/${MODEL}/${DATASET}_${HORIZON}/checkpoint_best.pth" ]; then
            ((EXISTING++))
        fi
    done
done

PROGRESS=$((EXISTING * 100 / TOTAL))

echo "📊 Progress: ${EXISTING} / ${TOTAL} checkpoints ready (${PROGRESS}%)"
echo ""

# Show progress bar
echo "Progress: "
printf "["
for i in $(seq 1 20); do
    if [ $i -le $EXISTING ]; then
        printf "█"
    else
        printf "░"
    fi
done
printf "] ${EXISTING}/20\n"
echo ""

# Show checkpoint status by horizon
echo "📁 Checkpoint Status:"
echo "────────────────────────────────────────────────────────────────────────────"
printf "%-15s | %-4s | %-4s | %-4s | %-4s\n" "Model" "H=96" "H=192" "H=336" "H=720"
echo "────────────────────────────────────────────────────────────────────────────"

for MODEL in "${BACKBONES[@]}"; do
    printf "%-15s |" "${MODEL}"
    for HORIZON in "${HORIZONS[@]}"; do
        if [ -f "./checkpoints/${MODEL}/${DATASET}_${HORIZON}/checkpoint_best.pth" ]; then
            printf " ✅  |"
        else
            printf " ⏸️  |"
        fi
    done
    printf "\n"
done

echo "────────────────────────────────────────────────────────────────────────────"
echo ""

# Show recent training activity
echo "📝 Recent Training Activity:"
echo "────────────────────────────────────────────────────────────────────────────"
tail -20 train_all_checkpoints.log 2>/dev/null || echo "No log file yet"
echo ""

# Find current training log directory
TRAIN_LOG_DIR=$(ls -td training_logs_* 2>/dev/null | head -1)

if [ -n "$TRAIN_LOG_DIR" ]; then
    echo "📂 Training logs directory: ${TRAIN_LOG_DIR}/"
    
    # Check if any training is currently happening
    CURRENT_TRAINING=$(ps aux | grep "main.py" | grep "TRAIN.ENABLE True" | grep -v grep | wc -l)
    
    if [ $CURRENT_TRAINING -gt 0 ]; then
        echo "🔄 Currently training: $CURRENT_TRAINING model(s)"
        
        # Try to find which model is training
        LATEST_LOG=$(ls -t ${TRAIN_LOG_DIR}/*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "📄 Latest log: $(basename ${LATEST_LOG})"
            echo ""
            echo "Recent metrics:"
            tail -10 "${LATEST_LOG}" | grep -E "Epoch:|Loss|MSE|MAE" | tail -5
        fi
    else
        echo "⏸️  No active training (between jobs or finished)"
    fi
fi

echo ""
echo "────────────────────────────────────────────────────────────────────────────"
echo "💡 To monitor live: tail -f train_all_checkpoints.log"
echo "💡 To check specific model log: tail -f training_logs_*/MODEL_ETTh1_HORIZON_train.log"
echo ""

# Estimate remaining time
REMAINING=$((TOTAL - EXISTING))
if [ $REMAINING -gt 0 ]; then
    # Rough estimate: 1-2 minutes per model
    EST_MINUTES=$((REMAINING * 2))
    EST_HOURS=$((EST_MINUTES / 60))
    EST_MIN=$((EST_MINUTES % 60))
    
    if [ $EST_HOURS -gt 0 ]; then
        echo "⏱️  Estimated time remaining: ~${EST_HOURS}h ${EST_MIN}m"
    else
        echo "⏱️  Estimated time remaining: ~${EST_MIN}m"
    fi
else
    echo "✅ All checkpoints complete!"
fi

echo ""
