#!/bin/bash

# Train base models for ETTh2 dataset
# This needs to be done before running SPEC-TTA experiments

echo "========================================="
echo "Training Base Models for ETTh2"
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
        
        # Run training script
        SCRIPT_PATH="scripts/${MODEL}/${DATASET}_${HORIZON}/train.sh"
        if [ -f "$SCRIPT_PATH" ]; then
            bash "$SCRIPT_PATH" > "train_${MODEL}_${DATASET}_${HORIZON}.log" 2>&1
            
            # Check if training succeeded
            if [ -f "${CHECKPOINT_DIR}/checkpoint_best.pth" ]; then
                echo "✓ Training completed successfully"
            else
                echo "✗ Training failed - check train_${MODEL}_${DATASET}_${HORIZON}.log"
            fi
        else
            echo "✗ Training script not found: $SCRIPT_PATH"
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
