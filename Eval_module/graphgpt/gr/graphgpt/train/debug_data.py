#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import torch.utils._pytree

# --- 1. 修复 HuggingFace Hub (解决 sentence_transformers 报错) ---
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    print("✅ Patch 1 Applied: huggingface_hub.cached_download polyfill")
except ImportError:
    pass

# --- 2. PyTorch 2.1 兼容性 ---
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    _orig_register = torch.utils._pytree._register_pytree_node
    def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return _orig_register(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
    print("✅ Patch 2 Applied: torch.utils._pytree.register_pytree_node")

# --- 3. 覆盖 Transformers 安全检查，确定环境安全 ---
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    import transformers.tokenization_utils_base
    import transformers.processing_utils
except ImportError:
    pass

def safe_check(*args, **kwargs):
    return None

transformers.utils.import_utils.check_torch_load_is_safe = safe_check
for name, module in list(sys.modules.items()):
    if name.startswith("transformers"):
        if hasattr(module, "check_torch_load_is_safe"):
            setattr(module, "check_torch_load_is_safe", safe_check)
print("✅ Patch 3 Applied: Neutralized check_torch_load_is_safe.")

import torch
import transformers
import json
import sys
import os

# 确保能导入 graphgpt 模块
sys.path.append(os.path.abspath("graphgpt"))

from graphgpt.gr.graphgpt.train import train_graph
from graphgpt.gr.graphgpt import conversation as conversation_lib

# === 配置 ===
MODEL_PATH = "/mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k"
DATA_PATH = "./graphgpt/gr/data/stage_2/gran_train_instruct.json"

def debug_one_sample():
    print(f"Loading Tokenizer from: {MODEL_PATH}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    
    # 模拟 train_graph.py 里的特殊 token 添加
    # 这一步非常重要，否则 <g_patch> 会被拆成碎 token，导致长度计算全乱
    print("Adding special tokens...")
    special_tokens = {
        "additional_special_tokens": ["<g_patch>", "<g_start>", "<g_end>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    print(f"Vocab size: {len(tokenizer)}")

    # 设置 Conversation Template
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    print(f"Loading Data from: {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)
    
    # 取第一个样本
    sample = raw_data[0]
    print("\n" + "="*40)
    print("【原始数据样例】")
    print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")
    print("="*40)

    # 模拟 preprocess_graph 流程
    # 构造 graph_cfg
    graph_cfg = {
        'is_graph': True,
        'sep_graph_conv_front': False,
        'use_graph_start_end': True
    }
    
    # 假设图特征长度 (Stage 1 通常是固定的，比如 100 左右，这里随便设一个模拟值)
    # 注意：这个长度必须和 <g_patch> * N 的数量一致
    dummy_token_len_1 = 10 
    dummy_token_len_2 = 10

    print("Running preprocess_graph_LP ...")
    # 模拟 source 结构
    sources = [sample["conversations"]]
    sources = train_graph.preprocess_graph_LP(
        sources, graph_cfg, dummy_token_len_1, dummy_token_len_2
    )
    
    print("\n【Graph Preprocess 后文本】(前100字符)")
    print(sources[0][0]["value"][:100] + "...")

    print("\nRunning preprocess_v1 ...")
    data_dict = train_graph.preprocess_v1(sources, tokenizer)
    
    input_ids = data_dict["input_ids"][0]
    labels = data_dict["labels"][0]

    print("\n" + "="*40)
    print("【Mask 结果可视化】")
    print("蓝色 = Input (被 Mask, label=-100)")
    print("红色 = Target (参与 Loss 计算)")
    print("="*40)

    decoded_tokens = []
    current_color = None # "blue" or "red"

    # 遍历 token 进行可视化
    for idx, (tok_id, label) in enumerate(zip(input_ids, labels)):
        if tok_id == tokenizer.pad_token_id:
            continue
            
        token = tokenizer.decode([tok_id])
        
        # 判断颜色
        if label == -100:
            new_color = "blue"
        else:
            new_color = "red"
            
        # 简单打印逻辑
        if new_color == "blue":
            print(f"\033[94m{token}\033[0m", end="") # 蓝色
        else:
            print(f"\033[91m{token}\033[0m", end="") # 红色
            
    print("\n\n" + "="*40)
    
    # 统计有效 Label
    valid_labels = (labels != -100).sum().item()
    print(f"Total Tokens: {len(input_ids)}")
    print(f"Valid Labels (Red): {valid_labels}")
    
    if valid_labels == 0:
        print("❌ 严重错误: 所有 Label 都是 -100 (Loss 0.0 原因确认)")
        print("可能原因: 分隔符匹配失败 或 长度计算错位")
    else:
        print(f"✅ 数据正常: 包含 {valid_labels} 个可学习 Token。")

if __name__ == "__main__":
    debug_one_sample()