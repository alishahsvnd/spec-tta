#!/bin/bash

# Train base models for ETTh2 without TTA evaluation
# This avoids the assertion error and just creates checkpoints

echo "========================================="
echo "Training Base Models for ETTh2 (No TTA)"
echo "========================================="
echo ""

DATASET="ETTh2"
MODELS=("iTransformer" "DLinear" "PatchTST")
HORIZONS=(96 192 336 720)

for MODEL in "${MODELS[@]}"; do
    for HORIZON in "${HORIZONS[@]}"; do
        CHECKPOINT_DIR="checkpoints/${MODEL}/${DATASET}_${HORIZON}/"
        
        # Check if checkpoint already exists
        if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
            echo "✓ Checkpoint exists: ${MODEL} ${DATASET} H=${HORIZON}"
            continue
        fi
        
        echo ""
        echo "Training: ${MODEL} on ${DATASET} (horizon ${HORIZON})"
        echo "----------------------------------------"
        
        # Train without TTA by setting TTA.ENABLE=False
        python main.py \
            DATA.NAME ${DATASET} \
            DATA.PRED_LEN ${HORIZON} \
            MODEL.NAME ${MODEL} \
            MODEL.pred_len ${HORIZON} \
            TRAIN.ENABLE True \
            TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
            TTA.ENABLE False \
            > "train_${MODEL}_${DATASET}_${HORIZON}_nottta.log" 2>&1
        
        # Check if training succeeded
        if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
            echo "✓ Training completed successfully"
        else
            echo "✗ Training failed - check train_${MODEL}_${DATASET}_${HORIZON}_nottta.log"
        fi
    done
done

echo ""
echo "========================================="
echo "Base Model Training Complete"
echo "========================================="
echo ""
echo "Trained models:"
find checkpoints -path "*/ETTh2_*/checkpoint_best.pth" | sort
echo ""
echo "Now you can run SPEC-TTA experiments with:"
echo "./run_single_dataset.sh ETTh2"
