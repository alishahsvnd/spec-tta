# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env bash
# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage: bash scripts/PatchTST/ETTh1_96/run_spec_tta.sh <gpu> [k_bins] [beta_freq] [drift_thresh]
# Example: bash scripts/PatchTST/ETTh1_96/run_spec_tta.sh 0 16 0.05 0.01

GPU=${1:-0}
K=${2:-16}
BETA=${3:-0.05}
DRIFT=${4:-0.01}

# Derived settings
TTA=SPEC_TTA_KBINS_${K}_BETA_${BETA}_DRIFT_${DRIFT}
DATASET="ETTh1"
PRED_LEN=96
MODEL="PatchTST"
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
RESULT_DIR="./results/${TTA}/"
OUTPUT="${TTA}_${MODEL}_${DATASET}_${PRED_LEN}.txt"

# SPEC-TTA parameters
SPEC_LR=0.001
PATCH_LEN=24
HUBER_DELTA=0.5
LAMBDA_PW=1.0
LAMBDA_PROX=0.0001
LAMBDA_HC=0.1
GRAD_CLIP=1.0

export CUDA_VISIBLE_DEVICES=$GPU

python -u main.py \
  DATA.NAME ${DATASET} \
  DATA.PRED_LEN ${PRED_LEN} \
  MODEL.NAME ${MODEL} \
  MODEL.pred_len ${PRED_LEN} \
  TRAIN.ENABLE False \
  TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
  TTA.ENABLE True \
  TTA.SPEC_TTA.K_BINS ${K} \
  TTA.SPEC_TTA.PATCH_LEN ${PATCH_LEN} \
  TTA.SPEC_TTA.HUBER_DELTA ${HUBER_DELTA} \
  TTA.SPEC_TTA.BETA_FREQ ${BETA} \
  TTA.SPEC_TTA.LAMBDA_PW ${LAMBDA_PW} \
  TTA.SPEC_TTA.LAMBDA_PROX ${LAMBDA_PROX} \
  TTA.SPEC_TTA.LAMBDA_HC ${LAMBDA_HC} \
  TTA.SPEC_TTA.DRIFT_THRESHOLD ${DRIFT} \
  TTA.SPEC_TTA.LR ${SPEC_LR} \
  TTA.SPEC_TTA.GRAD_CLIP ${GRAD_CLIP} \
  RESULT_DIR ${RESULT_DIR} | tee ${OUTPUT}
