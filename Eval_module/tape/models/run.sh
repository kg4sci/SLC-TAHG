#!/usr/bin/env bash
set -euo pipefail

# 需要的环境变量（可选覆盖）：
#   CUDA_VISIBLE_DEVICES, TAPE_DEVICE, TAPE_FORCE_CPU
#   EPOCHS, LR, EMB_DIM, HID_DIM, DROPOUT, BATCH_SIZE, VAL_EVERY, OUTPUT_BASE

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
EMB_DIM=${EMB_DIM:-384}
HID_DIM=${HID_DIM:-384}
DROPOUT=${DROPOUT:-0.3}
BATCH_SIZE=${BATCH_SIZE:-128}
VAL_EVERY=${VAL_EVERY:-5}
OUTPUT_BASE=${OUTPUT_BASE:-$(pwd)/Eval_module/tape/models/output_data}

echo "[TAPE] [1/3] Training GNN embeddings..."
python3 -m Eval_module.tape.models.core.trainGNN "$@"

echo "[TAPE] [2/3] Training LM embeddings..."
python3 -m Eval_module.tape.models.core.trainLM "$@"

echo "[TAPE] [3/3] Training/Evaluating cascade model..."
python3 -m Eval_module.tape.models.core.trainCascading \
  --output_base "${OUTPUT_BASE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --emb_dim "${EMB_DIM}" \
  --hidden_dim "${HID_DIM}" \
  --dropout "${DROPOUT}" \
  --batch_size "${BATCH_SIZE}" \
  --val_every "${VAL_EVERY}" \
  --use_node_feat 1 \
  --use_text_feature 1 \
  "$@"

echo "[TAPE] Pipeline done. Outputs under ${OUTPUT_BASE}"

