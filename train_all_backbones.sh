#!/bin/bash
# Train all 5 backbones from scratch on ETTh1 H=96

cd /home/alishah/PETSA || exit 1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Training All Backbones from Scratch (ETTh1 H=96)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  WARNING: This will delete existing checkpoints!"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

DATASET="ETTh1"
PRED_LEN=96
BASE_LR=0.001
WEIGHT_DECAY=0.0
GATING_INIT=0.01

BACKBONES=("iTransformer" "DLinear" "PatchTST" "MICN" "FreTS")

echo ""
echo "📋 Training Plan:"
for BACKBONE in "${BACKBONES[@]}"; do
    echo "  • $BACKBONE → ./checkpoints/${BACKBONE}/${DATASET}_${PRED_LEN}/"
done
echo ""

# Function to train a single backbone
train_backbone() {
    local MODEL=$1
    local CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔨 Training: $MODEL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Checkpoint: $CHECKPOINT_DIR"
    echo "Started: $(date)"
    echo ""
    
    # Remove existing checkpoint
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "⚠️  Removing existing checkpoint: $CHECKPOINT_DIR"
        rm -rf "$CHECKPOINT_DIR"
    fi
    
    # Train
    echo "🚀 Starting training..."
    python main.py \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE True \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TTA.ENABLE False \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        2>&1 | tee "train_${MODEL}_${DATASET}_${PRED_LEN}.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ] && [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        echo "✅ $MODEL training completed successfully!"
        echo "   Checkpoint: ${CHECKPOINT_DIR}/checkpoint_best.pth"
        
        # Extract final metrics
        local train_mse=$(grep "train_mse:" "train_${MODEL}_${DATASET}_${PRED_LEN}.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        local val_mse=$(grep "val_mse:" "train_${MODEL}_${DATASET}_${PRED_LEN}.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        
        echo "   Final Train MSE: ${train_mse:-N/A}"
        echo "   Final Val MSE: ${val_mse:-N/A}"
    else
        echo "❌ $MODEL training failed! (exit code: $exit_code)"
        echo "   Check log: train_${MODEL}_${DATASET}_${PRED_LEN}.log"
    fi
    
    echo "Finished: $(date)"
    echo ""
}

# Train all backbones sequentially
START_TIME=$(date +%s)

for MODEL in "${BACKBONES[@]}"; do
    train_backbone "$MODEL"
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TRAINING COMPLETE                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📊 Training Summary:"
echo "───────────────────────────────────────────────────────────────"
printf "%-15s | %-10s | %-40s\n" "Backbone" "Status" "Checkpoint"
echo "───────────────────────────────────────────────────────────────"

for MODEL in "${BACKBONES[@]}"; do
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        printf "%-15s | %-10s | %s\n" "$MODEL" "✅ SUCCESS" "${CHECKPOINT_DIR}checkpoint_best.pth"
    else
        printf "%-15s | %-10s | %s\n" "$MODEL" "❌ FAILED" "No checkpoint"
    fi
done

echo "───────────────────────────────────────────────────────────────"
echo ""
echo "📁 Training logs saved:"
for MODEL in "${BACKBONES[@]}"; do
    if [ -f "train_${MODEL}_${DATASET}_${PRED_LEN}.log" ]; then
        echo "  • train_${MODEL}_${DATASET}_${PRED_LEN}.log"
    fi
done
echo ""

# Count successes
SUCCESS_COUNT=0
for MODEL in "${BACKBONES[@]}"; do
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
        ((SUCCESS_COUNT++))
    fi
done

echo "🎯 Result: $SUCCESS_COUNT / ${#BACKBONES[@]} backbones trained successfully"
echo ""
