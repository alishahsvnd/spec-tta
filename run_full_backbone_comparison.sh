#!/bin/bash
# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

####################################################################################
# Comprehensive comparison: PETSA vs High-Capacity SPEC-TTA (3 configs)
# Runs on all backbones: iTransformer, PatchTST, DLinear, FreTS, MICN, OLS
####################################################################################

DATASET="ETTh1"
PRED_LEN=96
GPU_ID=0

# PETSA configuration (baseline)
PETSA_RANK=4
PETSA_LOSS_ALPHA=0.1
PETSA_GATING_INIT=0.01

# Output directory
RESULTS_BASE="./comparison_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_BASE}

# Base learning rate and weight decay (same as PETSA for fair comparison)
BASE_LR=0.001
WEIGHT_DECAY=0.0

####################################################################################
# Function to run an experiment
####################################################################################
run_experiment() {
    local MODEL=$1
    local CONFIG_TYPE=$2  # "PETSA" or "SPEC_TTA_HC"
    local CONFIG_NAME=$3  # e.g., "Medium", "High", "Ultra"
    local EXTRA_ARGS=$4
    
    echo "======================================================================"
    echo "Running: ${MODEL} with ${CONFIG_TYPE} (${CONFIG_NAME})"
    echo "======================================================================"
    
    local CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    local RESULT_DIR="./results/${CONFIG_TYPE}_${CONFIG_NAME}/"
    local OUTPUT="${RESULTS_BASE}/${CONFIG_TYPE}_${CONFIG_NAME}_${MODEL}_${DATASET}_${PRED_LEN}.txt"
    
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    
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
        RESULT_DIR ${RESULT_DIR} \
        ${EXTRA_ARGS} 2>&1 | tee ${OUTPUT}
    
    echo "Results saved to: ${OUTPUT}"
    echo ""
}

####################################################################################
# List of all backbones to test
####################################################################################
BACKBONES=("iTransformer" "PatchTST" "DLinear" "FreTS" "MICN" "OLS")

####################################################################################
# Run PETSA baseline on all backbones
####################################################################################
echo "###################################################################"
echo "PHASE 1: Running PETSA baseline on all backbones"
echo "###################################################################"

for MODEL in "${BACKBONES[@]}"; do
    run_experiment \
        "${MODEL}" \
        "PETSA" \
        "RANK${PETSA_RANK}" \
        "TTA.PETSA.RANK ${PETSA_RANK} TTA.PETSA.LOSS_ALPHA ${PETSA_LOSS_ALPHA} TTA.PETSA.GATING_INIT ${PETSA_GATING_INIT}"
done

####################################################################################
# Run SPEC-TTA HC Medium (12K params) on all backbones
####################################################################################
echo "###################################################################"
echo "PHASE 2: Running SPEC-TTA HC Medium (12K params) on all backbones"
echo "###################################################################"

for MODEL in "${BACKBONES[@]}"; do
    run_experiment \
        "${MODEL}" \
        "SPEC_TTA_HC" \
        "Medium" \
        "TTA.SPEC_TTA_HC.MODE medium \
         TTA.SPEC_TTA_HC.K_LOW 6 \
         TTA.SPEC_TTA_HC.K_MID 12 \
         TTA.SPEC_TTA_HC.K_HIGH 20 \
         TTA.SPEC_TTA_HC.RANK 8 \
         TTA.SPEC_TTA_HC.GATING_DIM 32 \
         TTA.SPEC_TTA_HC.LR 2e-3"
done

####################################################################################
# Run SPEC-TTA HC High (24K params) on all backbones
####################################################################################
echo "###################################################################"
echo "PHASE 3: Running SPEC-TTA HC High (24K params) on all backbones"
echo "###################################################################"

for MODEL in "${BACKBONES[@]}"; do
    run_experiment \
        "${MODEL}" \
        "SPEC_TTA_HC" \
        "High" \
        "TTA.SPEC_TTA_HC.MODE high \
         TTA.SPEC_TTA_HC.K_LOW 8 \
         TTA.SPEC_TTA_HC.K_MID 16 \
         TTA.SPEC_TTA_HC.K_HIGH 25 \
         TTA.SPEC_TTA_HC.RANK 16 \
         TTA.SPEC_TTA_HC.GATING_DIM 64 \
         TTA.SPEC_TTA_HC.LR 2e-3"
done

####################################################################################
# Run SPEC-TTA HC Ultra (36K params) on all backbones
####################################################################################
echo "###################################################################"
echo "PHASE 4: Running SPEC-TTA HC Ultra (36K params) on all backbones"
echo "###################################################################"

for MODEL in "${BACKBONES[@]}"; do
    run_experiment \
        "${MODEL}" \
        "SPEC_TTA_HC" \
        "Ultra" \
        "TTA.SPEC_TTA_HC.MODE ultra \
         TTA.SPEC_TTA_HC.K_LOW 10 \
         TTA.SPEC_TTA_HC.K_MID 20 \
         TTA.SPEC_TTA_HC.K_HIGH 19 \
         TTA.SPEC_TTA_HC.RANK 24 \
         TTA.SPEC_TTA_HC.GATING_DIM 128 \
         TTA.SPEC_TTA_HC.LR 2e-3"
done

####################################################################################
# Summary
####################################################################################
echo "###################################################################"
echo "All experiments completed!"
echo "Results directory: ${RESULTS_BASE}"
echo "###################################################################"
echo ""
echo "To analyze results, run:"
echo "  python analyze_comparison_results.py ${RESULTS_BASE}"
echo ""
