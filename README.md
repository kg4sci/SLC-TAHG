# SLC-TAHG: Benchmarking Few-Shot Cascade Reasoning on Text-Attributed Hyper-Relational Graphs
## 1.Download SLC-TAHG
We have uploaded the models and the proposed SLC-TAHG to Huggingface.
| Huggingface Address | Description |
| ---- | ---- |
|  |    |
|  |    |

## 2. Environment Preparation
Please first clone the repo and install the required environment, which can be done by running the following.
### Models Best Params
In the graph-based evaluation model, we used the Optuna framework for 200 iterations of parameter tuning. The optimal parameters for each model are stored in **Best_modelPara**, and these optimal parameters are then used to run each model and obtain the corresponding evaluation metrics.
e.g.
```python
python optuna_hyperparameter_tuning.py --model GRAN --n_trials 200
```
### dataset config
You need to download the corresponding Neo4j data. Whether or not you download the MongoDB metadata depends on your individual needs. The text data processed by the LLM from MongoDB is already stored in the `Eval_module/data` folder.
```python
NEO4J_URI = "bolt://xxx" #Replace with your corresponding content.
NEO4J_USER = "xxx"
NEO4J_PASSWORD = "xxx"

MONGO_URI = "mongodb://xxx"
```

### (1)GraphGPT model
**commands:**
```python
cd Eval_module
conda create --name graphgpt python==3.9
conda activate graphgpt
pip install -r requirements_graphgpt.txt
```
**How to run**
```python
bash graphgpt/gr/run_stage1.sh
bash graphgpt/gr/train_gran_stage2.sh
bash graphgpt/gr/eval_gran.sh
python -m graphgpt.gr.eval_gran_cascading \
  --model_output_file ./graphgpt/gr/graphgpt/eval/arxiv_test_res_all.json \
  --save_path ./graphgpt/gr/graphgpt/eval/arxiv_test_res_all_metrics.json  #The corresponding model_output_file and save_path can be modified as needed.
```
### (2)TAPE model
**commands:**
```python
cd Eval_module
conda create --name TAPE python==3.8
conda activate TAPE
pip install -r requirements_tape.txt
```
**How to run**
```python
bash tape/models/run.sh
```
### (3)other models
**commands:**
```python
cd Eval_module
conda create --name slcdb python==3.10
conda activate slcdb
pip install -r requirements.txt
```
**How to run**
```python
#GRAN
python -m Eval_module.GRAN.GRAN
#RGCN
python -m Eval_module.RGCN.RGCN
#N-ComplEx
python -m Eval_module.NComplEx.NComplEx
...#The remaining models can be renamed by modifying the executable files in their respective model folders.
```

