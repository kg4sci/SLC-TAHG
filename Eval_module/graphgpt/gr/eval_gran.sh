#!/bin/bash
# GraphGPT Evaluation Script for GRAN Cascading Prediction Task
# GRAN级联预测任务的GraphGPT评估脚本

export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 配置路径
# Stage 2 基于 Stage 1 训练完成后的模型继续微调
model_path=./graphgpt_output/gran_stage_2  # 训练好的模型路径
instruct_ds=./graphgpt/gr/data/stage_2/gran_test_instruct.json  # 测试指令数据
graph_data_path=./graphgpt/gr/graph_data/gran_graph_data.pt # 图数据路径
output_dir=./graphgpt_output  # 输出目录

mkdir -p ${output_dir}

python3 graphgpt/gr/graphgpt/eval/run_graphgpt_LP.py \
    --model-name ${model_path} \
    --prompting_file ${instruct_ds} \
    --graph_data_path ${graph_data_path} \
    --output_res_path ${output_dir} \
    --projector_path ${model_path}/graph_projector.bin

