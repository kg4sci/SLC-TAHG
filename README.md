# SLC-TAHG: Benchmarking Few-Shot Cascade Reasoning on Text-Attributed Hyper-Relational Graphs

SLC-TAHG is an evidence-traceable, text-attributed hyper-relational graph benchmark for evaluating cascade reasoning over SLC-centered biomedical mechanisms, models SLC-related biomedical mechanisms as event-centric hyper-relational cascades.

<p align="center">
  <img src="KG Schema.png" width="720">
</p>

Each RelaEvent connects an SLC gene, a biological pathway, and a disease outcome, together with relation polarity labels and supporting evidence.

```text
SLCGene --relAB--> Pathway --relBC--> Disease
````
where both `relAB` and `relBC` are labeled as `promotion` or `suppression`.

---

## 1. Dataset Statistics

| Item               | Count / Description                     |
| ------------------ | --------------------------------------- |
| Nodes              | 2,777                                   |
| Edges              | 4,473                                   |
| Node types         | 8                                       |
| Core cascade       | SLCGene → Pathway → Disease             |
| Relation labels    | promotion / suppression                 |
| Text attributes    | Evidence text linked to RelaEvent nodes |
| Task               | Two-stage cascade relation prediction   |
| Evaluation setting | Full-data and few-shot learning         |

---

## 2. Download SLC-TAHG

We have uploaded the proposed SLC-TAHG dataset and related files to the following address.

| SLC-TAHG Address                                                                                                                                       | Description                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| [https://www.scidb.cn/en/detail?dataSetId=d2461f91de8c49e5b545241119d41f1c](https://www.scidb.cn/en/detail?dataSetId=d2461f91de8c49e5b545241119d41f1c) | Download all data files and place them into the corresponding folders before running the code. |

---

## 3. Environment Preparation

Please first clone the repository and install the required environment.

```bash
git clone https://github.com/kg4sci/SLC-TAHG.git
cd SLC-TAHG
```

Different baseline families require different environments, as described below.

---

## 4. Dataset Configuration

You need to download the corresponding Neo4j data. Whether to download the MongoDB metadata depends on your needs. The text data processed by the LLM from MongoDB is already stored in the `Eval_module/data` folder.

Please modify the following configuration according to your local environment:

```python
NEO4J_URI = "bolt://xxx"      # Replace with your Neo4j URI
NEO4J_USER = "xxx"
NEO4J_PASSWORD = "xxx"

MONGO_URI = "mongodb://xxx"   # Optional, depending on your usage
```

---

## 5. Hyperparameter Tuning

For graph-based evaluation models, we use Optuna for 200 iterations of hyperparameter tuning. The optimal parameters for each model are stored in `Best_modelPara/`, and these parameters are used to run the corresponding model and obtain evaluation metrics.

Example:

```bash
python optuna_hyperparameter_tuning.py --model GRAN --n_trials 200
```

---

## 6. Running Baselines

### 6.1 GraphGPT Model

#### Environment

```bash
cd Eval_module
conda create --name graphgpt python==3.9
conda activate graphgpt
pip install -r requirements_graphgpt.txt
```

#### Run

```bash
bash graphgpt/gr/run_stage1.sh
bash graphgpt/gr/train_gran_stage2.sh
bash graphgpt/gr/eval_gran.sh
```

#### Cascading Evaluation

```bash
python -m graphgpt.gr.eval_gran_cascading \
  --model_output_file ./graphgpt/gr/graphgpt/eval/arxiv_test_res_all.json \
  --save_path ./graphgpt/gr/graphgpt/eval/arxiv_test_res_all_metrics.json
```

The corresponding `model_output_file` and `save_path` can be modified as needed.

---

### 6.2 TAPE Model

#### Environment

```bash
cd Eval_module
conda create --name TAPE python==3.8
conda activate TAPE
pip install -r requirements_tape.txt
```

#### Run

```bash
bash tape/models/run.sh
```

---

### 6.3 Other Models

This environment is used for models such as GRAN, RGCN, N-ComplEx, and other graph-based baselines.

#### Environment

```bash
cd Eval_module
conda create --name slcdb python==3.10
conda activate slcdb
pip install -r requirements.txt
```

#### Run

```bash
# GRAN
python -m Eval_module.GRAN.GRAN

# RGCN
python -m Eval_module.RGCN.RGCN

# N-ComplEx
python -m Eval_module.NComplEx.NComplEx
```

For the remaining models, please run the corresponding executable files in their respective model folders.

---

## 7. Evaluation Metrics

SLC-TAHG reports both path-level and edge-level metrics.

Main metrics include:

* Path Accuracy;
* Path F1;
* AB Accuracy / AB F1;
* BC Accuracy / BC F1;
* class-specific recall for promotion and suppression.

* The optimal hyperparameters for graph-based models are stored in `Best_modelPara/`.
* The LLM-processed text data is already provided in `Eval_module/data`.
* Neo4j is required for graph-based data access.
* MongoDB metadata is optional depending on whether additional raw metadata access is needed.

```
```
