# SLC-TAHG Benchmark Usage Protocol

## Overview

This document describes how to evaluate a new model on the SLC-TAHG benchmark. The benchmark tests cascade reasoning on a text-attributed hyper-relational knowledge graph of SLC tumor biology, where models must predict two dependent relation labels (rel_AB and rel_BC) in sequence.

## 1. Dataset Access

The dataset is published on ScienceDB:
**DOI: 10.57760/sciencedb.36429**

url：**https://www.scidb.cn/en/detail?dataSetId=d2461f91de8c49e5b545241119d41f1c**

Download the following files:
- `.dump` file — Neo4j database dump for graph restoration
- `.json` files — pre-extracted path data and data splits

## 2. Environment Setup

```bash
git clone https://github.com/kg4sci/SLC-TAHG.git
cd SLC-TAHG
```

Three conda environments are required for different model families:

**Other models (GRAN, RGCN, N-ComplEx, StarE, HypE, NaLP, RAM, NS-HART):**
```bash
cd Eval_module
conda create --name slcdb python=3.10
conda activate slcdb
pip install -r requirements.txt
```

**For TAPE:**
```bash
conda create --name TAPE python=3.8
conda activate TAPE
pip install -r requirements_tape.txt
```

**For GraphGPT:**
```bash
conda create --name graphgpt python=3.9
conda activate graphgpt
pip install -r requirements_graphgpt.txt
```

## 3. Neo4j Database Restoration

After downloading the `.dump` file from ScienceDB, restore it to your Neo4j database:

```bash
# Stop Neo4j service, then load the dump
neo4j-admin database load --from-path=/path/to/neo4j.dump --database=slcdb
# Or use neo4j-admin load (version-dependent)
```

Start Neo4j and verify the database is loaded correctly.

## 4. Configuration

Edit `Eval_module/config.py` to set your Neo4j connection parameters:

```python
NEO4J_URI = "bolt://localhost:xxx"     # Your Neo4j URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

MongoDB configuration is optional (text features are pre-extracted).

## 5. Path Data Format

Each path is retrieved from Neo4j by the `enumerate_graph_paths()` function in `Eval_module/path_data.py`. The query follows the cascade structure:

```
(SLCGene) --[IS_SOURCE]--> (RelaEvent) --[IS_TARGET]--> (Disease)
                                ^
                                | [IS_MEDIATOR]
                             (Pathway)
```

Each path sample is a dictionary:

```python
{
    "A": int,           # SLCGene node ID
    "A_name": str,      # SLCGene name (e.g., "SLC7A11")
    "B": int,           # Pathway node ID
    "B_name": str,      # Pathway name (e.g., "Ferroptosis")
    "C": int,           # Disease node ID
    "C_name": str,      # Disease name (e.g., "Liver Cancer")
    "Event": int,       # RelaEvent node ID (hyper-relational hub)
    "rel_AB": str,      # "promotion" or "suppression"
    "rel_BC": str,      # "promotion" or "suppression"
}
```

## 6. Evaluation Steps for a New Model

### Step 1: Load paths

```python
from Eval_module.path_data import enumerate_graph_paths

all_paths = enumerate_graph_paths()  # Loads from Neo4j
```

### Step 2: Split into train/val/test

```python
from Eval_module.path_data import split_paths

train_paths, val_paths, test_paths = split_paths(
    all_paths, train_ratio=0.7, val_ratio=0.15, seed=42
)
```

For few-shot evaluation, use `select_few_shot()` with K = {1, 5, 10, 20, 50, 100}.

### Step 3: Build relation mapping

```python
name_to_id = {"promotion": 0, "suppression": 1}
id_to_name = {0: "promotion", 1: "suppression"}
```

### Step 4: Implement your model interface

Your model should accept path batches (or individual paths) and predict two labels:

```python
def predict(self, paths: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Args:
        paths: list of path dictionaries with "A", "B", "C", "Event" keys
    Returns:
        pred_AB: list of "promotion"/"suppression"
        pred_BC: list of "promotion"/"suppression" (conditioned on pred_AB)
    """
    # Your implementation here
    return pred_AB, pred_BC
```

### Step 5: Evaluate

```python
from sklearn.metrics import accuracy_score, f1_score

# Generate predictions
pred_AB, pred_BC = model.predict(test_paths)

# True labels
true_AB = [p["rel_AB"] for p in test_paths]
true_BC = [p["rel_BC"] for p in test_paths]

# Edge-level metrics
ab_acc = accuracy_score(true_AB, pred_AB)
ab_f1 = f1_score(true_AB, pred_AB, average="macro")
bc_acc = accuracy_score(true_BC, pred_BC)
bc_f1 = f1_score(true_BC, pred_BC, average="macro")

# Path-level metrics
TT = sum(1 for t_ab, p_ab, t_bc, p_bc
         in zip(true_AB, pred_AB, true_BC, pred_BC)
         if t_ab == p_ab and t_bc == p_bc)
total = len(test_paths)
path_acc = TT / total
```

## 7. Existing Baselines

The repository includes 10 baselines in `Eval_module/`:

| Category | Models | Run command |
|----------|--------|-------------|
| Tensor Factorization | HypE, N-ComplEx, NaLP | `python -m Eval_module.{Model}.{Model}` |
| Relational GNN | RGCN, RAM | `python -m Eval_module.{Model}.{Model}` |
| Hypergraph-specific | StarE, GRAN, NS-HART | `python -m Eval_module.{Model}.{Model}` |
| LLM + Graph | TAPE, GraphGPT | See respective `run.sh` scripts |

Example for GRAN:
```bash
# Activate the slcdb environment, then:
python -m Eval_module.GRAN.GRAN
```

## 8. Hyperparameter Tuning

```bash
python optuna_hyperparameter_tuning.py --model YOUR_MODEL --n_trials 200
```

## 9. Evaluation Metrics Reference

| Metric | Formula |
|--------|---------|
| Path_ACC | TT / (TT + TF + FT + FF) |
| Path_F1 | F1 with TT as positive class |
| AB_ACC | Standard accuracy for AB edge |
| AB_F1 | Macro F1 = (F1_P + F1_S) / 2 |
| AB_recall_P | TP_P / (TP_P + FN_P) for AB, promotion |
| AB_recall_S | TP_S / (TP_S + FN_S) for AB, suppression |
| BC_* | Same as AB_* for BC edge |

TT = both predictions correct, TF = AB correct only, FT = BC correct only, FF = both wrong.
