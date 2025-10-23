#!/bin/bash
# Test single backbone with SPEC-TTA Phase 1+2

cd /home/alishah/PETSA || exit 1

BACKBONE=$1
if [ -z "$BACKBONE" ]; then
    echo "Usage: $0 <backbone_name>"
    echo "Available: iTransformer, DLinear, PatchTST, MICN, FreTS"
    exit 1
fi

echo "Testing $BACKBONE on ETTh1 H=96 with SPEC-TTA Phase 1+2"

python main.py \
    DATA.NAME ETTh1 \
    DATA.PRED_LEN 96 \
    MODEL.NAME $BACKBONE \
    MODEL.pred_len 96 \
    TRAIN.ENABLE False \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SPEC_TTA.K_BINS 32 \
    TRAIN.CHECKPOINT_DIR "./checkpoints/$BACKBONE/ETTh1_96/"
