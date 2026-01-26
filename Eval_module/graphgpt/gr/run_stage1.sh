export PYTHONPATH=$(pwd)/graphgpt:$PYTHONPATH  
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true  # 使用环境变量而不是 wandb 命令
model_path=/mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k # Vicuna 模型路径
instruct_ds=./graphgpt/gr/data/stage_2/gran_train_instruct.json  # 训练指令数据
graph_data_path=./graphgpt/gr/graph_data/gran_graph_data.pt # 图数据路径
pretra_gnn=clip_gt_arxiv   # 预训练图编码器路径 
output_model=./graphgpt_output/gran_stage_1             # 输出模型的路径
graph_content=./graphgpt/gr/data/stage_2/gran_graph_content.json

python3 graphgpt/gr/graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ${graph_content} \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 false \
    --output_dir ${output_model} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --max_grad_norm 0.05 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --freeze_graph_tower False \
    --lazy_preprocess True \
    --dataloader_num_workers 0 \
    --report_to none
	#-m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 