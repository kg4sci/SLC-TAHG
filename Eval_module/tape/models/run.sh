#!/usr/bin/env bash
set -euo pipefail

# Step 1: Train TAPE GNN to refresh node embeddings
echo "[1/4] Training GNN embeddings..."
python3 -m Eval_module.tape.models.core.trainGNN "$@"

# Step 2: Train TAPE LM to refresh text embeddings
echo "[2/4] Training LM embeddings..."
python3 -m Eval_module.tape.models.core.trainLM "$@"

# Step 3: Run Optuna hyperparameter tuning for TAPE cascade model
echo "[3/4] Running Optuna tuning..."
python3 -m Eval_module.optuna_hyperparameter_tuning --model TAPE --n_trials 200 "$@"

# Step 4: Perform 10-fold cross-validation with the best parameters
# echo "[4/4] Running 10-fold cross-validation..."
# python3 -m Eval_module.cross_validation --model TAPE --folds 10 "$@"

echo "Pipeline completed."

