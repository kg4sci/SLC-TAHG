#!/bin/bash
# GraphGPT Stage 2 Training Script for GRAN Cascading Prediction Task (FIXED)
# 强制将 GPU3 映射为 cuda:0
export PYTHONPATH=$(pwd)/graphgpt:$PYTHONPATH  
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_DISABLE_MIXED_PRECISION=1
export ACCELERATE_USE_CPU=False
export ACCELERATE_DISTRIBUTED_TYPE=NO
export DISTRIBUTED_WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0

# 禁用 FlashAttention
export FLASH_ATTENTION_DISABLE=1
export FLASH_ATTENTION_FORCE_DISABLED=1

# 禁用 accelerate 的 multi-GPU 自动选择
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 防止某些库偷用 GPU0
unset CUDA_DEVICE

# 显示当前 GPU 设置
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 配置路径
# Stage 2 基于 Stage 1 训练完成后的模型继续微调
model_path=./graphgpt_output/gran_stage_1
instruct_ds=./graphgpt/gr/data/stage_2/gran_train_instruct.json
graph_data_path=./graphgpt/gr/graph_data/gran_graph_data.pt
pretra_gnn=clip_gt_arxiv
tuned_proj=./graphgpt_output/gran_stage_1/graph_projector.bin 
graph_content=./graphgpt/gr/data/stage_2/gran_graph_content.json
# output_model=./checkpoints/gran_stage_2

# output_model=./checkpoints/gran_stage_2_safe

# 清理旧的坏权重（或者直接用新目录）
# rm -rf ./checkpoints/gran_stage_2

# 使用新目录，确保从头开始
output_model=./graphgpt_output/gran_stage_2

python3 graphgpt/gr/graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ${graph_content} \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj} \
    \
    --tune_graph_mlp_adapter False \
    --freeze_graph_mlp_adapter False \
    \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    \
    --bf16 False \
    --fp16 False \
    --double_quant True  \
    --quant_type "nf4" \
    \
    --output_dir ${output_model} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    \
    --learning_rate 2e-5 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.03 \
    \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --dataloader_num_workers 0\
    --report_to none
