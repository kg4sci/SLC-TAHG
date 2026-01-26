# SLC-TAHG: A Text-Attribute Hyper-Relational Graph Benchmark for Cascade Reasoning in SLC–Tumor Mechanisms
## 1.Download SLC-TAHG
We have uploaded the models and the proposed SLC-TAHG to Huggingface.
| Huggingface Address | Description |
| ---- | ---- |
|  |    |
|  |    |

## 2. Environment Preparation
Please first clone the repo and install the required environment, which can be done by running the following.
### (1)GraphGPT models
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
### (1)TAPE models
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

