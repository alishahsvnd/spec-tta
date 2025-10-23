#!/bin/bash

# Train All Checkpoints for All Horizons and Backbones
# Dataset: ETTh1

DATASET="ETTh1"
HORIZONS=(96 192 336 720)
BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

# Training hyperparameters
BASE_LR=0.001
WEIGHT_DECAY=0.0
MAX_EPOCH=30

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG_DIR="training_logs_${TIMESTAMP}"

mkdir -p "${TRAIN_LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              Train All Checkpoints - ETTh1 Dataset                           ║"
echo "║              Horizons: 96, 192, 336, 720                                     ║"
echo "║              Backbones: iTransformer, DLinear, PatchTST, MICN, FreTS        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL_CONFIGS=$((${#HORIZONS[@]} * ${#BACKBONES[@]}))
COMPLETED=0
SKIPPED=0
FAILED=0

echo "📊 Total configurations to train: ${TOTAL_CONFIGS}"
echo "⚙️  Training settings: LR=${BASE_LR}, Epochs=${MAX_EPOCH}, Weight Decay=${WEIGHT_DECAY}"
echo ""

START_TIME=$(date +%s)

for HORIZON in "${HORIZONS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📏 HORIZON: ${HORIZON}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for MODEL in "${BACKBONES[@]}"; do
        echo "┌─────────────────────────────────────────────────────────────────────────┐"
        echo "│ 🔨 Training: ${MODEL} (H=${HORIZON})                                    │"
        echo "└─────────────────────────────────────────────────────────────────────────┘"
        
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
        
        # Check if checkpoint already exists
        if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
            CHECKPOINT_SIZE=$(du -h "${CHECKPOINT_DIR}/checkpoint_best.pth" | cut -f1)
            echo "   ✅ Checkpoint already exists: ${CHECKPOINT_DIR}"
            echo "      Size: ${CHECKPOINT_SIZE}"
            ((SKIPPED++))
            ((COMPLETED++))
            PROGRESS=$((COMPLETED * 100 / TOTAL_CONFIGS))
            echo "   📊 Progress: ${COMPLETED}/${TOTAL_CONFIGS} (${PROGRESS}%) | ✅ ${SKIPPED} skipped, ❌ ${FAILED} failed"
            echo ""
            continue
        fi
        
        echo "   🚀 Starting training..."
        echo "      Checkpoint: ${CHECKPOINT_DIR}"
        echo "      Log: ${TRAIN_LOG_DIR}/${MODEL}_${DATASET}_${HORIZON}_train.log"
        
        TRAIN_START=$(date +%s)
        
        # Train the model
        python main.py \
            DATA.NAME ${DATASET} \
            DATA.PRED_LEN ${HORIZON} \
            MODEL.NAME ${MODEL} \
            MODEL.pred_len ${HORIZON} \
            TRAIN.ENABLE True \
            SOLVER.MAX_EPOCH ${MAX_EPOCH} \
            SOLVER.BASE_LR ${BASE_LR} \
            SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
            TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
            TEST.ENABLE False \
            TTA.ENABLE False \
            > "${TRAIN_LOG_DIR}/${MODEL}_${DATASET}_${HORIZON}_train.log" 2>&1
        
        TRAIN_EXIT_CODE=${PIPESTATUS[0]}
        TRAIN_END=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
        TRAIN_MINUTES=$((TRAIN_DURATION / 60))
        TRAIN_SECONDS=$((TRAIN_DURATION % 60))
        
        # Check if training succeeded
        if [ $TRAIN_EXIT_CODE -eq 0 ] && [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
            CHECKPOINT_SIZE=$(du -h "${CHECKPOINT_DIR}/checkpoint_best.pth" | cut -f1)
            
            # Extract final metrics from log
            TRAIN_LOG="${TRAIN_LOG_DIR}/${MODEL}_${DATASET}_${HORIZON}_train.log"
            FINAL_VAL_MSE=$(grep -E "Validation.*MSE|val_mse:" "${TRAIN_LOG}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
            FINAL_VAL_MAE=$(grep -E "Validation.*MAE|val_mae:" "${TRAIN_LOG}" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
            
            echo "   ✅ Training completed successfully!"
            echo "      Duration: ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s"
            echo "      Checkpoint: ${CHECKPOINT_DIR}/checkpoint_best.pth (${CHECKPOINT_SIZE})"
            if [ -n "$FINAL_VAL_MSE" ]; then
                echo "      Final Val MSE: ${FINAL_VAL_MSE}"
            fi
            if [ -n "$FINAL_VAL_MAE" ]; then
                echo "      Final Val MAE: ${FINAL_VAL_MAE}"
            fi
        else
            echo "   ❌ Training failed!"
            echo "      Exit code: ${TRAIN_EXIT_CODE}"
            echo "      Check log: ${TRAIN_LOG_DIR}/${MODEL}_${DATASET}_${HORIZON}_train.log"
            ((FAILED++))
        fi
        
        ((COMPLETED++))
        PROGRESS=$((COMPLETED * 100 / TOTAL_CONFIGS))
        echo "   📊 Progress: ${COMPLETED}/${TOTAL_CONFIGS} (${PROGRESS}%) | ✅ ${SKIPPED} skipped, ❌ ${FAILED} failed"
        echo ""
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         TRAINING COMPLETE                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📊 Summary:"
echo "   Total configurations: ${TOTAL_CONFIGS}"
echo "   Already existed: ${SKIPPED}"
echo "   Newly trained: $((COMPLETED - SKIPPED - FAILED))"
echo "   Failed: ${FAILED}"
echo "   Success rate: $(( (COMPLETED - FAILED) * 100 / TOTAL_CONFIGS ))%"
echo ""

# Generate checkpoint summary
echo "📁 Checkpoint Summary:"
echo "────────────────────────────────────────────────────────────────────────────"
printf "%-15s | %-8s | %-12s | %s\n" "Model" "Horizon" "Status" "Size"
echo "────────────────────────────────────────────────────────────────────────────"

for HORIZON in "${HORIZONS[@]}"; do
    for MODEL in "${BACKBONES[@]}"; do
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
        CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint_best.pth"
        
        if [ -f "${CHECKPOINT_FILE}" ]; then
            SIZE=$(du -h "${CHECKPOINT_FILE}" | cut -f1)
            printf "%-15s | %-8s | %-12s | %s\n" "${MODEL}" "${HORIZON}" "✅ EXISTS" "${SIZE}"
        else
            printf "%-15s | %-8s | %-12s | %s\n" "${MODEL}" "${HORIZON}" "❌ MISSING" "N/A"
        fi
    done
done

echo "────────────────────────────────────────────────────────────────────────────"
echo ""
echo "📝 Training logs saved to: ${TRAIN_LOG_DIR}/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All checkpoints ready! You can now run comparison experiments."
    echo ""
    echo "Next step: bash run_all_horizons_all_backbones.sh"
else
    echo "⚠️  Some training jobs failed. Check logs in ${TRAIN_LOG_DIR}/"
    echo ""
    echo "You can still run experiments with available checkpoints."
fi

echo ""
