#!/bin/bash
# Quick test: Run comparison on just iTransformer (fastest to verify everything works)

DATASET="ETTh1"
PRED_LEN=96
GPU_ID=0
MODEL="iTransformer"

# PETSA configuration
PETSA_RANK=4
PETSA_LOSS_ALPHA=0.1
PETSA_GATING_INIT=0.01

# Output directory
RESULTS_BASE="./quick_test_results"
mkdir -p ${RESULTS_BASE}

BASE_LR=0.001
WEIGHT_DECAY=0.0

export CUDA_VISIBLE_DEVICES=${GPU_ID}

CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"

echo "======================================================================"
echo "Quick Test: Running all configs on ${MODEL}"
echo "======================================================================"

####################################################################################
# 1. PETSA Baseline
####################################################################################
echo ""
echo "[1/4] Running PETSA baseline..."
python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.PETSA.RANK ${PETSA_RANK} \
    TTA.PETSA.LOSS_ALPHA ${PETSA_LOSS_ALPHA} \
    TTA.PETSA.GATING_INIT ${PETSA_GATING_INIT} \
    RESULT_DIR "./results/PETSA_RANK${PETSA_RANK}/" \
    2>&1 | tee ${RESULTS_BASE}/PETSA_RANK${PETSA_RANK}_${MODEL}_${DATASET}_${PRED_LEN}.txt

####################################################################################
# 2. SPEC-TTA HC Medium (12K params)
####################################################################################
echo ""
echo "[2/4] Running SPEC-TTA HC Medium (12K params)..."
python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.SPEC_TTA_HC.MODE medium \
    TTA.SPEC_TTA_HC.K_LOW 6 \
    TTA.SPEC_TTA_HC.K_MID 12 \
    TTA.SPEC_TTA_HC.K_HIGH 20 \
    TTA.SPEC_TTA_HC.RANK 8 \
    TTA.SPEC_TTA_HC.GATING_DIM 32 \
    TTA.SPEC_TTA_HC.LR 2e-3 \
    RESULT_DIR "./results/SPEC_TTA_HC_Medium/" \
    2>&1 | tee ${RESULTS_BASE}/SPEC_TTA_HC_Medium_${MODEL}_${DATASET}_${PRED_LEN}.txt

####################################################################################
# 3. SPEC-TTA HC High (24K params)
####################################################################################
echo ""
echo "[3/4] Running SPEC-TTA HC High (24K params)..."
python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.SPEC_TTA_HC.MODE high \
    TTA.SPEC_TTA_HC.K_LOW 8 \
    TTA.SPEC_TTA_HC.K_MID 16 \
    TTA.SPEC_TTA_HC.K_HIGH 25 \
    TTA.SPEC_TTA_HC.RANK 16 \
    TTA.SPEC_TTA_HC.GATING_DIM 64 \
    TTA.SPEC_TTA_HC.LR 2e-3 \
    RESULT_DIR "./results/SPEC_TTA_HC_High/" \
    2>&1 | tee ${RESULTS_BASE}/SPEC_TTA_HC_High_${MODEL}_${DATASET}_${PRED_LEN}.txt

####################################################################################
# 4. SPEC-TTA HC Ultra (36K params)
####################################################################################
echo ""
echo "[4/4] Running SPEC-TTA HC Ultra (36K params)..."
python main.py \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.SPEC_TTA_HC.MODE ultra \
    TTA.SPEC_TTA_HC.K_LOW 10 \
    TTA.SPEC_TTA_HC.K_MID 20 \
    TTA.SPEC_TTA_HC.K_HIGH 19 \
    TTA.SPEC_TTA_HC.RANK 24 \
    TTA.SPEC_TTA_HC.GATING_DIM 128 \
    TTA.SPEC_TTA_HC.LR 2e-3 \
    RESULT_DIR "./results/SPEC_TTA_HC_Ultra/" \
    2>&1 | tee ${RESULTS_BASE}/SPEC_TTA_HC_Ultra_${MODEL}_${DATASET}_${PRED_LEN}.txt

####################################################################################
# Analyze results
####################################################################################
echo ""
echo "======================================================================"
echo "Quick test complete! Analyzing results..."
echo "======================================================================"

python analyze_comparison_results.py ${RESULTS_BASE}

echo ""
echo "✅ Quick test complete!"
echo "Results saved to: ${RESULTS_BASE}"
echo ""
echo "If everything looks good, run the full comparison:"
echo "  ./run_full_backbone_comparison.sh"
echo ""
