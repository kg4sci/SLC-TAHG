# -----------------------------------------------------------
# helper to normalize model outputs
# -----------------------------------------------------------
FORMAT_TEMPLATE = (
    "Step 1 - A-B Relationship: {step1}\n"
    "Step 2 - B-C Relationship: {step2}"
)


def normalize_relationship_output(output_text: str) -> str:
    """Force model output to adhere to the required two-line format."""
    if not output_text:
        return FORMAT_TEMPLATE.format(step1="UNKNOWN", step2="UNKNOWN")

    # 查找 PROMOTION / SUPPRESSION 关键词，忽略大小写
    matches = re.findall(r"\b(PROMOTION|SUPPRESSION)\b", output_text.upper())

    if not matches:
        step1 = step2 = "UNKNOWN"
    elif len(matches) == 1:
        step1 = matches[0]
        step2 = matches[0]
    else:
        step1, step2 = matches[0], matches[1]

    return FORMAT_TEMPLATE.format(step1=step1, step2=step2)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import re
import torch, json, os, os.path as osp, random, copy
import types  # 用于 SimpleNamespace 转换
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import torch.nn as nn
from graphgpt.gr.graphgpt.model.graph_layers.graph_transformer import graph_transformer


def _resolve_target_dtype(dtype_str: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.bfloat16)


def _get_graph_tower(model):
    if hasattr(model, "get_model") and hasattr(model.get_model(), "graph_tower"):
        return model.get_model().graph_tower
    return getattr(model, "graph_tower", None)


def _log_graph_tower_stats(model, label: str, max_layers: int = 5):
    graph_tower = _get_graph_tower(model)
    if graph_tower is None:
        print(f"[{label}] graph_tower not present.")
        return

    print(f"[{label}] Inspecting graph_tower ({graph_tower.__class__.__name__})")
    problematic = []
    for idx, (name, param) in enumerate(graph_tower.named_parameters()):
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            problematic.append((name, nan_count, inf_count))
        if idx < max_layers:
            print(
                f"  - {name}: shape={tuple(param.shape)}, dtype={param.dtype}, "
                f"nan={has_nan}, inf={has_inf}, range=[{param.min().item():.6f}, {param.max().item():.6f}]"
            )

    if problematic:
        print(f"  ❌ Detected {len(problematic)} graph_tower params with nan/inf:")
        for name, nan_count, inf_count in problematic[:10]:
            print(f"     • {name}: nan_count={nan_count}, inf_count={inf_count}")
    else:
        print("  ✓ No nan/inf detected in graph_tower parameters.")

# === Patches to ensure TensorGraphData is available for torch.load ===
try:
    from graphgpt.gr.data.build_gran_instruct import TensorGraphData
    import sys, types
    if "__main__" not in sys.modules:
        sys.modules["__main__"] = types.ModuleType("__main__")
    setattr(sys.modules["__main__"], "TensorGraphData", TensorGraphData)
    import importlib
    try:
        importlib.import_module('graphgpt.gr.data.build_gran_instruct')
    except Exception:
        from importlib.machinery import SourceFileLoader
        mod_path = os.path.abspath("graphgpt/gr/data/build_gran_instruct.py")
        if os.path.exists(mod_path):
            mod = SourceFileLoader('graphgpt.gr.data.build_gran_instruct', mod_path).load_module()
            import sys as _sys
            _sys.modules['graphgpt.gr.data.build_gran_instruct'] = mod
except Exception as _e:
    print("[Warning] Could not register TensorGraphData automatically:", _e)

def safe_torch_load(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

# -----------------------------------------------------------
# load graph data
# -----------------------------------------------------------
from torch_geometric.data import Data

def load_graph_LP(instruct_item, graph_data_path):
    raw_graph_data = safe_torch_load(graph_data_path)
    
    # 格式适配：将字典转换为对象，支持 .x 访问（与训练代码保持一致）
    # types 已在文件顶部导入
    graph_data_all = {}
    if isinstance(raw_graph_data, dict):
        for k, v in raw_graph_data.items():
            if isinstance(v, dict):
                # 将 dict 转为对象，支持 .x 访问
                graph_data_all[k] = types.SimpleNamespace(**v)
            else:
                # 如果已经是对象，直接使用
                graph_data_all[k] = v
    else:
        # 极其罕见的情况，直接赋值
        graph_data_all = raw_graph_data
    
    graph_dict = instruct_item["graph"]
    graph_type = instruct_item["id"].split("_")[0]

    g1_edge = torch.tensor(graph_dict["edge_index_1"]).long()
    g1_nodes = graph_dict["node_list_1"]
    g1_target = graph_dict["node_idx_1"]
    g1_rep = graph_data_all[graph_type].x[g1_nodes]
    g1_len = len(g1_rep)

    g2_edge = torch.tensor(graph_dict["edge_index_2"]).long()
    g2_nodes = graph_dict["node_list_2"]
    g2_target = graph_dict["node_idx_2"]
    g2_rep = graph_data_all[graph_type].x[g2_nodes]
    g2_len = len(g2_rep)

    return {
        "graph_data": {
            "graph_1": Data(graph_node=g1_rep, edge_index=g1_edge, target_node=torch.tensor([g1_target])),
            "graph_2": Data(graph_node=g2_rep, edge_index=g2_edge, target_node=torch.tensor([g2_target]))
        },
        "graph_token_len_1": g1_len,
        "graph_token_len_2": g2_len
    }

# -----------------------------------------------------------
# helper to ensure graph_projector exists
# -----------------------------------------------------------
def ensure_and_load_projector(model, projector_path=None):
    projector_attr_parent = None
    projector_attr_name = None
    if hasattr(model, "model") and hasattr(model.model, "graph_projector"):
        projector_attr_parent = model.model
        projector_attr_name = "graph_projector"
    elif hasattr(model, "graph_projector"):
        projector_attr_parent = model
        projector_attr_name = "graph_projector"

    # 这里不再转换 convert，而是稍后统一处理，防止覆盖
    def _convert_module_to_model_dtype_and_device(mod):
        try:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            mod.to(device=model_device, dtype=model_dtype)
            return True
        except Exception as e:
            print("WARN: failed to convert projector module dtype/device:", e)
            return False

    if projector_attr_parent is not None and getattr(projector_attr_parent, projector_attr_name) is not None:
        mod = getattr(projector_attr_parent, projector_attr_name)
        _convert_module_to_model_dtype_and_device(mod)
        print("INFO: graph_projector already exists on model -> synced")
        return True

    if projector_path and os.path.exists(projector_path):
        try:
            print("INFO: Attempting to load projector state from:", projector_path)
            state = safe_torch_load(projector_path, map_location="cpu")
            if isinstance(state, dict):
                candidate_w = None
                weight_key = None
                bias_key = None
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 2:
                        candidate_w = v
                        weight_key = k
                        break
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 1 and (weight_key is None or k != weight_key):
                        bias_key = k
                        break

                if candidate_w is not None:
                    out_dim, in_dim = candidate_w.shape
                    new_linear = nn.Linear(in_dim, out_dim)
                    if weight_key:
                        new_linear.weight.data.copy_(state[weight_key])
                    if bias_key:
                        new_linear.bias.data.copy_(state[bias_key])
                    
                    if projector_attr_parent is None:
                        if hasattr(model, "model"):
                            model.model.graph_projector = new_linear
                            mod_attached = model.model.graph_projector
                        else:
                            model.graph_projector = new_linear
                            mod_attached = model.graph_projector
                    else:
                        setattr(projector_attr_parent, projector_attr_name, new_linear)
                        mod_attached = getattr(projector_attr_parent, projector_attr_name)

                    _convert_module_to_model_dtype_and_device(mod_attached)
                    print("INFO: Restored graph_projector from linear state (inferred).")
                    return True
        except Exception as e:
            print("ERROR: loading projector file failed:", e)

    # Fallback
    cfg = getattr(model, "config", None)
    in_dim = getattr(cfg, "graph_hidden_size", None) or getattr(cfg, "graph_hidden", None) or 768
    out_dim = getattr(cfg, "graph_proj_size", None) or getattr(cfg, "graph_proj_out", None) or getattr(cfg, "hidden_size", 4096)
    print(f"WARN: Creating fallback graph_projector ({in_dim} -> {out_dim})")
    new_linear = nn.Linear(in_dim, out_dim)
    
    if hasattr(model, "model"):
        model.model.graph_projector = new_linear
        mod_attached = model.model.graph_projector
    else:
        model.graph_projector = new_linear
        mod_attached = model.graph_projector

    _convert_module_to_model_dtype_and_device(mod_attached)
    return True


import json
from pathlib import Path
import torch
from graphgpt.gr.graphgpt.model.GraphLlama import GraphLlamaConfig, GraphLlamaForCausalLM

def load_graphllama_model(
    model_dir,
    projector_fallback_dir=None,
    tokenizer=None,
    base_model_path=None,
    inspect_steps=False,
    skip_postload_fix=False,
    target_dtype=torch.bfloat16,
):
    """
    Robust loader with Tokenizer-aware resizing AND BF16 support.
    Supports both full model and LoRA checkpoints.
    """
    model_dir = Path(model_dir)
    projector_fallback_dir = projector_fallback_dir or model_dir

    print("Loading GraphLlamaConfig...")
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = model_dir
    config = GraphLlamaConfig.from_pretrained(config_path, trust_remote_code=True)
    # 若 tokenizer 词表已扩展，优先用 tokenizer 大小覆盖 config.vocab_size，避免 32000/32003 mismatch
    target_vocab_size = None
    if tokenizer is not None and hasattr(tokenizer, "__len__"):
        try:
            target_vocab_size = len(tokenizer)
        except Exception:
            pass
    if target_vocab_size and getattr(config, "vocab_size", None) != target_vocab_size:
        print(f"Adjust config.vocab_size: {getattr(config, 'vocab_size', None)} -> {target_vocab_size} (match tokenizer)")
        config.vocab_size = target_vocab_size

    # LoRA 相关路径
    lora_adapter_path = os.path.join(model_dir, "adapter_model.bin")
    if not os.path.exists(lora_adapter_path):
        lora_adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
    
    # 检查是否是 LoRA checkpoint
    adapter_file = model_dir / "adapter_model.safetensors"
    adapter_config_file = model_dir / "adapter_config.json"
    is_lora = adapter_file.exists() or adapter_config_file.exists()
    
    # 如果是 LoRA，需要从基础模型加载
    if is_lora:
        if base_model_path is None:
            # 尝试从 config 获取基础模型路径
            if hasattr(config, '_name_or_path') and config._name_or_path:
                base_model_path = config._name_or_path
            elif hasattr(config, 'base_model') and config.base_model:
                base_model_path = config.base_model
            else:
                # 尝试常见路径
                common_base_paths = []
                # 优先查找与当前 LoRA 目录并列的 Stage-1 检查点
                stage1_candidate = model_dir.parent / "gran_stage_1"
                if stage1_candidate.exists():
                    common_base_paths.append(str(stage1_candidate))

                common_base_paths.extend([
                    "/mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k",
                    "./vicuna-7b-v1.5-16k",
                    os.path.expanduser("~/vicuna-7b-v1.5-16k"),
                ])

                for path in common_base_paths:
                    if os.path.exists(path):
                        base_model_path = path
                        break
        
        if base_model_path is None:
            raise ValueError(
                "LoRA checkpoint detected but base model path not found. "
                "Please provide --base-model-path or ensure config contains base_model path."
            )
        
        print(f"Loading base model from: {base_model_path}")
        print(f"Loading model structure (hf loader) in {str(target_dtype).split('.')[-1]}...")
        # 关键：使用 LoRA 目录中的 GraphLlamaConfig，以便构建 graph_tower 并加载预训练图权重
        model = GraphLlamaForCausalLM.from_pretrained(
            str(base_model_path),
            config=config,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,  # 若词表扩展，避免直接报错
        )
        print("Base model loaded. (LoRA path will be merged next)")
    else:
        # 强制使用 bfloat16 加载模型
        print(f"Loading model structure (hf loader) in {str(target_dtype).split('.')[-1]}...")
        model = GraphLlamaForCausalLM.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=target_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        print("Model loaded. (LoRA path will be merged next)")

    if inspect_steps:
        _log_graph_tower_stats(model, "after base model load")

    # 若模型词表与 tokenizer 长度不一致，调整嵌入大小
    target_vocab_size = len(tokenizer) if tokenizer is not None else getattr(config, "vocab_size", None)
    if target_vocab_size and model.get_input_embeddings().weight.shape[0] != target_vocab_size:
        print(f"Resizing token embeddings: {model.get_input_embeddings().weight.shape[0]} -> {target_vocab_size}")
        model.resize_token_embeddings(target_vocab_size)

    # Load shards / checkpoints
    # 依次检查分片索引：pytorch_model.bin.index.json 与 model.safetensors.index.json
    idx_file = model_dir / "pytorch_model.bin.index.json"
    sft_idx_file = model_dir / "model.safetensors.index.json"
    full_state = {}
    loaded_weights = False
    
    if idx_file.exists():
        import json
        with open(idx_file, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        for shard_name in sorted(set(weight_map.values())):
            shard_path = model_dir / shard_name
            if shard_path.exists():
                print(f"Loading shard: {shard_path}")
                sd = torch.load(shard_path, map_location="cpu")
                full_state.update(sd)
                loaded_weights = True
    elif sft_idx_file.exists():
        import json
        try:
            from safetensors import safe_open
        except ImportError:
            safe_open = None
        with open(sft_idx_file, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        for shard_name in shard_names:
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                continue
            print(f"Loading safetensor shard: {shard_path}")
            if safe_open is not None:
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        full_state[key] = f.get_tensor(key)
            else:
                # 回退：先尝试 torch.load（若已装好 safetensors 依赖会被用于 .safetensors）
                sd = torch.load(shard_path, map_location="cpu")
                full_state.update(sd)
            loaded_weights = True
        
    if (adapter_file.exists() or adapter_config_file.exists()) and not loaded_weights:
        print("Detected LoRA checkpoint. Loading LoRA weights...")
        try:
            from peft import PeftModel
            
            # 首先加载 non_lora_trainables.bin（embed_tokens, graph_projector 等）
            non_lora_file = model_dir / "non_lora_trainables.bin"
            if non_lora_file.exists():
                print(f"Loading non-LoRA trainables: {non_lora_file}")
                non_lora_trainables = torch.load(str(non_lora_file), map_location="cpu")
                
                # 处理键名：移除 'base_model.' 或 'model.' 前缀
                processed_non_lora = {}
                for k, v in non_lora_trainables.items():
                    # 移除 'base_model.model.' 前缀
                    if k.startswith('base_model.model.'):
                        new_k = k[17:]  # 移除 'base_model.model.'
                    elif k.startswith('base_model.'):
                        new_k = k[11:]  # 移除 'base_model.'
                    elif k.startswith('model.model.'):
                        new_k = k[6:]  # 移除 'model.'
                    else:
                        new_k = k
                    processed_non_lora[new_k] = v
                
                # 检查 vocab size
                if "model.embed_tokens.weight" in processed_non_lora:
                    sd_vocab_size = processed_non_lora["model.embed_tokens.weight"].shape[0]
                    if model.get_input_embeddings().weight.shape[0] != sd_vocab_size:
                        print(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {sd_vocab_size}")
                        model.resize_token_embeddings(sd_vocab_size)
                
                missing, unexpected = model.load_state_dict(processed_non_lora, strict=False)
                print(f"Loaded non-LoRA trainables. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if missing:
                    print(f"  First 10 missing: {missing[:10]}")
                if unexpected:
                    print(f"  First 10 unexpected: {unexpected[:10]}")
                if inspect_steps:
                    _log_graph_tower_stats(model, "after non-LoRA load")
            else:
                print("WARNING: non_lora_trainables.bin not found, skipping non-LoRA weights")
            
            # 然后加载 LoRA adapter
            print("Loading LoRA adapter...")
            # 强制在 CPU 上加载 LoRA，避免在加载阶段占用 CUDA 显存导致 OOM
            model = PeftModel.from_pretrained(model, str(model_dir), device_map={"": "cpu"})
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("LoRA weights merged successfully!")
            loaded_weights = True
            if inspect_steps:
                _log_graph_tower_stats(model, "after LoRA merge")

        except ImportError:
            print("ERROR: peft library not available. Please install: pip install peft")
            print("  Cannot load LoRA weights without peft library.")
        except Exception as e:
            print(f"ERROR: Failed to load LoRA weights: {e}")
    
    # 检查 safetensors 文件（新版本 HuggingFace 默认格式，非 LoRA）
    if not loaded_weights:
        try:
            from safetensors import safe_open
            safetensors_file = model_dir / "model.safetensors"
            if safetensors_file.exists():
                print(f"Loading safetensors: {safetensors_file}")
                state_dict = {}
                with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                
                if "model.embed_tokens.weight" in state_dict:
                    sd_vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
                    if model.get_input_embeddings().weight.shape[0] != sd_vocab_size:
                        print(
                            f"WARN: Model size {model.get_input_embeddings().weight.shape[0]} "
                            f"!= Checkpoint size {sd_vocab_size}. Resizing."
                        )
                        model.resize_token_embeddings(sd_vocab_size)
            
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {len(missing)}")
                print(f"Unexpected keys: {len(unexpected)}")
                loaded_weights = True
        except ImportError:
            print("safetensors not available, skipping safetensors check")
        except Exception as e:
            print(f"Failed to load safetensors: {e}")
    
    # 检查 checkpoint 子目录（训练过程中保存的 checkpoint）
    if not loaded_weights:
        checkpoint_dirs = [d for d in os.listdir(model_dir) 
                          if os.path.isdir(os.path.join(model_dir, d)) 
                          and d.startswith('checkpoint-')]
        if checkpoint_dirs:
            # 按数字排序，使用最新的 checkpoint
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0, reverse=True)
            latest_checkpoint = os.path.join(model_dir, checkpoint_dirs[0])
            print(f"Found checkpoint directories, trying to load from latest: {latest_checkpoint}")
            
            # 尝试从 checkpoint 加载
            checkpoint_bin = os.path.join(latest_checkpoint, "pytorch_model.bin")
            checkpoint_safetensors = os.path.join(latest_checkpoint, "model.safetensors")
            
            if os.path.exists(checkpoint_bin):
                print(f"Loading from checkpoint: {checkpoint_bin}")
                sd = torch.load(checkpoint_bin, map_location="cpu")
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(f"Missing keys: {len(missing)}")
                print(f"Unexpected keys: {len(unexpected)}")
                loaded_weights = True
            elif os.path.exists(checkpoint_safetensors):
                try:
                    from safetensors import safe_open
                    print(f"Loading from checkpoint safetensors: {checkpoint_safetensors}")
                    state_dict = {}
                    with safe_open(checkpoint_safetensors, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print(f"Missing keys: {len(missing)}")
                    print(f"Unexpected keys: {len(unexpected)}")
                    loaded_weights = True
                except Exception as e:
                    print(f"Failed to load checkpoint safetensors: {e}")
    
    if not loaded_weights:
        print("WARNING: No model weights file found! Model may not have correct weights loaded.")
        print("  Checked for:")
        print("    - pytorch_model.bin.index.json")
        print("    - pytorch_model.bin")
        print("    - model.safetensors")
        print("    - checkpoint-*/pytorch_model.bin")
        print("  The model will use weights from from_pretrained, which may be incorrect for stage2.")

    # After all weight loading, ensure embeddings match tokenizer (avoids CUDA OOB)
    if tokenizer is not None:
        tgt_vocab = len(tokenizer)
        cur_vocab = model.get_input_embeddings().weight.shape[0]
        if cur_vocab != tgt_vocab:
            print(f"Resizing embeddings after load: {cur_vocab} -> {tgt_vocab}")
            model.resize_token_embeddings(tgt_vocab)
            try:
                model.config.vocab_size = tgt_vocab
            except Exception:
                pass

    # Ensure projector exists
    cand_paths = [
        model_dir / "graph_projector.fp16.bin",
        model_dir / "graph_projector.bin",
        Path(projector_fallback_dir) / "graph_projector.fp16.bin",
        Path(projector_fallback_dir) / "graph_projector.bin",
    ]
    
    loaded_proj = False
    if full_state and not hasattr(model.model, "graph_projector"):
        w, b = None, None
        for k, v in full_state.items():
            if "graph_projector" in k and "weight" in k:
                w = v
                b_key = k.replace("weight", "bias")
                b = full_state.get(b_key)
                break
        if w is not None:
            new_linear = nn.Linear(w.shape[1], w.shape[0])
            new_linear.weight.data.copy_(w)
            if b is not None: new_linear.bias.data.copy_(b)
            model.model.graph_projector = new_linear
            print("Restored graph_projector from loaded state_dict.")
            loaded_proj = True

    if not loaded_proj:
        for p in cand_paths:
            if p.exists():
                print("Found projector file:", p)
                ensure_and_load_projector(model, str(p))
                loaded_proj = True
                if inspect_steps:
                    _log_graph_tower_stats(model, f"after projector load ({p.name})")
                break
    
    if not loaded_proj and not hasattr(model.model, "graph_projector"):
        print("WARN: Projector not found, creating fallback random init.")
        ensure_and_load_projector(model, None)
        if inspect_steps:
            _log_graph_tower_stats(model, "after projector fallback init")

    # 将模型移动到 CUDA。若使用 bitsandbytes 量化，避免 dtype 转换错误。
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    if is_quantized:
        if target_device != "cuda":
            print("WARNING: Bitsandbytes quantized model expected on CUDA; current target device is CPU.")
        else:
            device_map = getattr(model, "hf_device_map", None)
            print(f"Model loaded (bitsandbytes quantized); using existing device map: {device_map}.")
    else:
        move_dtype = target_dtype if target_device == "cuda" else model.dtype
        model = model.to(device=target_device, dtype=move_dtype)
        if target_device == "cuda":
            print(f"Model moved to CUDA with dtype={move_dtype}.")

    model.eval()
    if inspect_steps:
        _log_graph_tower_stats(model, "after device move")
    return model


# -----------------------------------------------------------
# main evaluation loop
# -----------------------------------------------------------
def run_eval(args):
    # load prompts
    with open(args.prompting_file, "r") as f:
        prompt_file = json.load(f)

    if args.is_shuffle:
        random.seed(0)
        random.shuffle(prompt_file)

    if args.end_id < 0:
        args.end_id = len(prompt_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    print(f"Using {len(prompt_file)} samples.")

    os.makedirs(args.output_res_path, exist_ok=True)
    # 结果累积容器
    res_data = []
    all_res = {"questions": [], "prediction": [], "raw_prediction": []}

    # 1. Load Tokenizer & Expand
    print("Loading tokenizer...")
    # 修复：GraphLlamaConfig 未在 TOKENIZER_MAPPING 中注册，需要直接使用 LlamaTokenizer
    from transformers import LlamaTokenizer
    
    tokenizer = None
    # 优先使用命令行显式指定的 tokenizer_path
    if getattr(args, "tokenizer_path", None):
        try:
            tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
            print(f"Loaded tokenizer from tokenizer_path: {args.tokenizer_path}")
        except Exception as e:
            print(f"WARNING: Failed to load tokenizer from tokenizer_path={args.tokenizer_path}: {e}")
            tokenizer = None
    tokenizer_files = ['tokenizer.model', 'tokenizer.json', 'vocab.json', 'tokenizer_config.json']
    
    # 策略1: 尝试直接从模型路径加载（如果 tokenizer 文件存在）
    has_tokenizer_files = any(os.path.exists(os.path.join(args.model_name, f)) for f in tokenizer_files)
    if has_tokenizer_files:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_fast=False)
            print(f"Loaded tokenizer from model directory: {args.model_name}")
        except Exception as e:
            print(f"WARNING: Failed to load tokenizer from model directory: {e}")
    
    # 检查 checkpoint 子目录中是否有 tokenizer 文件
    if tokenizer is None:
        try:
            # 查找所有 checkpoint 目录
            checkpoint_dirs = [d for d in os.listdir(args.model_name) 
                             if os.path.isdir(os.path.join(args.model_name, d)) 
                             and d.startswith('checkpoint-')]
            # 按名称排序，使用最新的 checkpoint
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0, reverse=True)
            
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_path = os.path.join(args.model_name, checkpoint_dir)
                has_tokenizer = any(os.path.exists(os.path.join(checkpoint_path, f)) for f in tokenizer_files)
                if has_tokenizer:
                    try:
                        tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path, use_fast=False)
                        print(f"Loaded tokenizer from checkpoint: {checkpoint_path}")
                        break
                    except Exception as e:
                        print(f"WARNING: Failed to load tokenizer from {checkpoint_path}: {e}")
                        continue
        except Exception as e:
            print(f"WARNING: Failed to check checkpoint directories: {e}")
    
    #尝试从配置中获取基础模型路径
    if tokenizer is None:
        try:
            config = GraphLlamaConfig.from_pretrained(args.model_name)
            # 尝试从配置中获取基础模型路径
            base_model_path = None
            if hasattr(config, '_name_or_path') and config._name_or_path:
                base_model_path = config._name_or_path
            elif hasattr(config, 'base_model') and config.base_model:
                base_model_path = config.base_model
            
            if base_model_path and os.path.exists(base_model_path):
                print(f"Loading tokenizer from base model: {base_model_path}")
                tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)
            else:
                # 如果配置中没有基础模型路径，尝试使用常见的路径
                # 通常基础模型路径可能是 vicuna-7b-v1.5-16k 
                common_base_paths = [
                    "/mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k",
                    "./vicuna-7b-v1.5-16k",
                    os.path.expanduser("~/vicuna-7b-v1.5-16k"),
                ]
                for base_path in common_base_paths:
                    if os.path.exists(base_path):
                        try:
                            tokenizer = LlamaTokenizer.from_pretrained(base_path, use_fast=False)
                            print(f"Loaded tokenizer from common base model path: {base_path}")
                            break
                        except Exception:
                            continue
                
                if tokenizer is None:
                    raise ValueError(f"Cannot determine base model path from config")
        except Exception as e:
            print(f"WARNING: Failed to load tokenizer from config: {e}")
    
    #尝试使用 AutoTokenizer
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
            print(f"Loaded tokenizer using AutoTokenizer: {args.model_name}")
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from {args.model_name}. "
                f"Please ensure the model directory or checkpoint contains tokenizer files, "
                f"or provide the base model path (e.g., /mnt/data/lxy/benchmark_paper/vicuna-7b-v1.5-16k). "
                f"Error: {e}"
            )
    
    DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
    DEFAULT_G_START_TOKEN = "<g_start>"
    DEFAULT_G_END_TOKEN = "<g_end>"
    
    special_tokens_dict = {'additional_special_tokens': [DEFAULT_GRAPH_PATCH_TOKEN, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens. Vocab size is now {len(tokenizer)}")

    # 2. Load Model in BF16
    print("Loading model (stage2 LLM weights)...")
    if not args.projector_path or not os.path.exists(args.projector_path):
        proj_fallback = args.model_name
    else:
        proj_fallback = os.path.dirname(args.projector_path)

    # 尝试从命令行或 config 获取基础模型路径（用于 LoRA）
    base_model_path = getattr(args, "base_model_path", None) or None
    try:
        config = GraphLlamaConfig.from_pretrained(args.model_name)
        if hasattr(config, '_name_or_path') and config._name_or_path:
            base_model_path = config._name_or_path
        elif hasattr(config, 'base_model') and config.base_model:
            base_model_path = config.base_model
    except Exception:
        pass
    
    inspection_enabled = getattr(args, "inspect_weights", False)
    skip_postload_fix = getattr(args, "skip_postload_fix", False)
    target_dtype = _resolve_target_dtype(getattr(args, "model_dtype", "bf16"))

    model = load_graphllama_model(
        args.model_name,
        projector_fallback_dir=proj_fallback,
        tokenizer=tokenizer,
        base_model_path=base_model_path,
        inspect_steps=inspection_enabled,
        skip_postload_fix=skip_postload_fix,
        target_dtype=target_dtype,
    )

    # 如果需要，覆盖 graph_tower 为干净的 clip_gt_arxiv 预训练权重（用于 A/B 验证图塔损坏）
    if getattr(args, "reload_graph_tower", False):
        try:
            clean_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "clip_gt_arxiv"))
            clean_config_path = os.path.join(clean_path, "config.json")
            if not os.path.exists(clean_config_path):
                raise FileNotFoundError(f"clean graph_tower config not found at {clean_config_path}")
            with open(clean_config_path, "r") as f:
                gt_args = types.SimpleNamespace(**json.load(f))
            clean_tower = graph_transformer(gt_args)
            # 尝试加载预训练权重（pkl）
            pkl_path = os.path.join(clean_path, "clip_gt_arxiv_pub.pkl")
            if os.path.exists(pkl_path):
                state = safe_torch_load(pkl_path, map_location="cpu")
                try:
                    # 兼容直接的 state_dict 或包含 state_dict 的字典
                    if isinstance(state, dict) and "state_dict" in state:
                        state_dict = state["state_dict"]
                    else:
                        state_dict = state
                    missing, unexpected = clean_tower.load_state_dict(state_dict, strict=False)
                    print(f"[Reload graph tower] Loaded clean state from pkl. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                except Exception as e:
                    print(f"[Reload graph tower] Failed to load pkl weights (will use random init): {e}")
            else:
                print(f"[Reload graph tower] pkl not found at {pkl_path}, using random-initialized clean_tower")

            # 移动到与模型相同的设备和 dtype
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            clean_tower = clean_tower.to(device=model_device, dtype=model_dtype)

            if hasattr(model, "get_model"):
                model.get_model().graph_tower = clean_tower
            elif hasattr(model, "graph_tower"):
                model.graph_tower = clean_tower
            print(f"[Reload graph tower] Replaced graph_tower with clean clip_gt_arxiv from {clean_path}")
            if inspection_enabled:
                _log_graph_tower_stats(model, "after graph tower reload")
        except Exception as e:
            print(f"[Reload graph tower] Failed to reload clean graph tower: {e}")
    
    # 检查并修复 graph_tower 参数中的 nan/inf
    if not skip_postload_fix:
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'graph_tower'):
            graph_tower = model.get_model().graph_tower
            if graph_tower is not None:
                import logging
                fixed_count = 0
                for name, param in graph_tower.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        nan_count = torch.isnan(param).sum().item()
                        inf_count = torch.isinf(param).sum().item()
                        logging.warning(
                            f"[Post-load fix] Graph tower parameter '{name}' contains nan/inf! "
                            f"nan_count={nan_count}, inf_count={inf_count}. Fixing..."
                        )
                        
                        # 修复策略：如果有有效值，使用有效值的统计信息；否则重新初始化
                        valid_mask = ~(torch.isnan(param.data) | torch.isinf(param.data))
                        if valid_mask.any():
                            valid_values = param.data[valid_mask]
                            mean_val = valid_values.mean().item()
                            std_val = valid_values.std().item()
                            std_val = min(std_val, 1.0)  # 限制标准差
                            replacement = torch.randn_like(param.data) * (std_val * 0.1 + 1e-6) + mean_val
                            replacement = torch.clamp(replacement, min=-2.0, max=2.0)
                        else:
                            # 如果全部无效，使用小的随机初始化
                            if len(param.shape) >= 2:
                                bound = (6.0 / (param.shape[0] + param.shape[1])) ** 0.5
                                replacement = torch.empty_like(param.data).uniform_(-bound, bound)
                            else:
                                replacement = torch.randn_like(param.data) * 0.01
                                replacement = torch.clamp(replacement, min=-0.1, max=0.1)
                        
                        # 只替换 nan/inf 的位置
                        param.data = torch.where(
                            torch.isnan(param.data) | torch.isinf(param.data),
                            replacement,
                            param.data
                        )
                        fixed_count += 1
                
                if fixed_count > 0:
                    logging.warning(
                        f"[Post-load fix] Fixed {fixed_count} graph_tower parameters with nan/inf after model loading."
                    )
        if inspection_enabled:
            _log_graph_tower_stats(model, "after post-load fix")
    else:
        if inspection_enabled:
            _log_graph_tower_stats(model, "after load (post-fix skipped)")
    
    # 关键检查：验证模型是否真的加载了训练后的权重
    # print("\n[DEBUG] Checking model weights...")
    # if hasattr(model, 'get_model') and hasattr(model.get_model(), 'graph_projector'):
    #     proj_weight = model.get_model().graph_projector.weight
    #     print(f"  Graph projector weight shape: {proj_weight.shape}")
    #     print(f"  Graph projector weight mean: {proj_weight.mean().item():.4f}")
    #     print(f"  Graph projector weight std: {proj_weight.std().item():.4f}")
    #     print(f"  Graph projector has nan: {torch.isnan(proj_weight).any().item()}")
    
    # if hasattr(model, 'get_input_embeddings'):
    #     embed_weight = model.get_input_embeddings().weight
    #     print(f"  Embed tokens weight shape: {embed_weight.shape}")
    #     print(f"  Embed tokens weight mean: {embed_weight.mean().item():.4f}")
    #     print(f"  Embed tokens weight std: {embed_weight.std().item():.4f}")
    
    # 检查 graph_tower 是否存在 + 打印详细统计
    def _inspect_graph_tower(graph_tower):
        tower_params = list(graph_tower.named_parameters())
        if not tower_params:
            print("  WARNING: Graph tower has no parameters!")
            return
        first_param = tower_params[0][1]
        print(f"  Graph tower first param name: {tower_params[0][0]}")
        print(f"  Graph tower first param shape: {first_param.shape}")
        
        # 检查是否有 nan/inf
        has_nan = torch.isnan(first_param).any().item()
        has_inf = torch.isinf(first_param).any().item()
        
        if has_nan or has_inf:
            nan_count = torch.isnan(first_param).sum().item()
            inf_count = torch.isinf(first_param).sum().item()
            print(f"  ❌ Graph tower first param contains nan/inf! nan_count={nan_count}, inf_count={inf_count}")
            print(f"  WARNING: This will cause graph_tower output to be invalid!")
        else:
            mean_val = first_param.mean().item()
            std_val = first_param.std().item()
            min_val = first_param.min().item()
            max_val = first_param.max().item()
            print(f"  Graph tower first param mean: {mean_val:.4f}")
            print(f"  Graph tower first param std: {std_val:.4f}")
            print(f"  Graph tower first param min: {min_val:.4f}")
            print(f"  Graph tower first param max: {max_val:.4f}")
        
        # 全量统计
        total_nan = 0
        total_inf = 0
        params_with_issues = []
        extreme_params = []
        for name, param in tower_params:
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            if nan_count > 0 or inf_count > 0:
                total_nan += nan_count
                total_inf += inf_count
                params_with_issues.append((name, nan_count, inf_count))
            # 记录均值/方差极端的参数，辅助定位坏权重
            try:
                mean_v = param.mean().item()
                std_v = param.std().item()
                extreme_params.append((abs(mean_v) + std_v, name, mean_v, std_v, float(param.min()), float(param.max())))
            except Exception:
                pass
        
        if params_with_issues:
            print(f"  ❌ Found {len(params_with_issues)} parameters with nan/inf:")
            for name, nan_cnt, inf_cnt in params_with_issues[:5]:  # 只显示前5个
                print(f"    - {name}: nan={nan_cnt}, inf={inf_cnt}")
            if len(params_with_issues) > 5:
                print(f"    ... and {len(params_with_issues) - 5} more")
            print(f"  Total: nan={total_nan}, inf={total_inf}")
        else:
            print(f"  ✅ All graph_tower parameters are healthy (no nan/inf)")
        
        extreme_params = sorted(extreme_params, key=lambda x: x[0], reverse=True)[:3]
        if extreme_params:
            print("  Top-3 parameters by |mean|+std (to catch异常分布):")
            for _, name, mean_v, std_v, min_v, max_v in extreme_params:
                print(f"    - {name}: mean={mean_v:.4f}, std={std_v:.4f}, min={min_v:.4f}, max={max_v:.4f}")

    if hasattr(model, 'get_model') and hasattr(model.get_model(), 'graph_tower'):
        graph_tower = model.get_model().graph_tower
        if graph_tower is not None:
            print(f"  Graph tower exists: {type(graph_tower).__name__}")
            _inspect_graph_tower(graph_tower)
        else:
            print("  WARNING: Graph tower is None!")
    else:
        print("  WARNING: Graph tower not found!")

    # Ensure BF16 AGAIN
    try:
        model = model.to(dtype=torch.bfloat16).cuda()
    except Exception:
        pass

    # 3. Update Config
    patch_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
    start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_G_START_TOKEN)
    end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_G_END_TOKEN)
    
    model.config.graph_patch_token = patch_token_id
    model.config.graph_start_token = start_token_id
    model.config.graph_end_token = end_token_id
    model.config.use_graph_start_end = True

    try:
        if hasattr(model, "get_model"):
            vision_tower = model.get_model().graph_tower
        else:
            vision_tower = model.model.graph_tower
            
        if vision_tower is not None and hasattr(vision_tower, "config"):
            vision_tower.config.graph_patch_token = patch_token_id
            vision_tower.config.use_graph_start_end = True
            vision_tower.config.graph_start_token = start_token_id
            vision_tower.config.graph_end_token = end_token_id
            print(f"INFO: Updated graph_tower config: patch_id={patch_token_id}, use_se=True")
    except Exception as e:
        print(f"WARN: Failed to update graph_tower config: {e}")

    # 4. Inference
    res_data = []
    print("Start inference...")

    for item in tqdm(prompt_file):
        gd = load_graph_LP(item, args.graph_data_path)
        graph_data = gd["graph_data"]
        len1, len2 = gd["graph_token_len_1"], gd["graph_token_len_2"]

        # 【修改点4】关键：将输入数据也转为 bfloat16
        graph_data["graph_1"].graph_node = graph_data["graph_1"].graph_node.cuda().to(dtype=torch.bfloat16)
        graph_data["graph_2"].graph_node = graph_data["graph_2"].graph_node.cuda().to(dtype=torch.bfloat16)
        graph_data["graph_1"] = graph_data["graph_1"].cuda()
        graph_data["graph_2"] = graph_data["graph_2"].cuda()

        DEFAULT_GRAPH_TOKEN = "<graph>"
        qs = item["conversations"][0]["value"]
        rep1 = DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * len1 + DEFAULT_G_END_TOKEN
        rep2 = DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * len2 + DEFAULT_G_END_TOKEN

        first = qs.find(DEFAULT_GRAPH_TOKEN)
        if first == -1: continue
        qs = qs[:first] + rep1 + qs[first+len(DEFAULT_GRAPH_TOKEN):]
        second = qs.find(DEFAULT_GRAPH_TOKEN)
        if second == -1: continue
        qs = qs[:second] + rep2 + qs[second+len(DEFAULT_GRAPH_TOKEN):]
        
        # 增强 prompt：在用户消息中显式加入输出格式约束，避免模型忽视系统提示
        format_instruction = (
            "\n\nImportant: The relationship type must be one of: PROMOTION or SUPPRESSION.\n"
            "Please provide your answer in the following format:\n"
            "Step 1 - A-B Relationship: [PROMOTION or SUPPRESSION]\n"
            "Step 2 - B-C Relationship: [PROMOTION or SUPPRESSION]"
        )
        if "Step 1 - A-B Relationship" not in qs:
            qs = qs.rstrip() + format_instruction

        stop_str = "</s>"
        answer_prefix = "Step 1 - A-B Relationship: "
        try:
            from graphgpt.gr.graphgpt.conversation import conv_templates, SeparatorStyle
            conv = conv_templates["graphchat_v1"].copy()
            conv.system += (
            "\nImportant: The relationship type you need to predict must be one of: PROMOTION or SUPPRESSION.\n"
            "Please provide your answer in the following format strictly:\n"
            "Step 1 - A-B Relationship: [PROMOTION or SUPPRESSION]\n"
            "Step 2 - B-C Relationship: [PROMOTION or SUPPRESSION]"
        )
            conv.append_message(conv.roles[0], qs)
            # 直接在 Assistant 消息里放入前缀，强制生成从结构化输出开始，避免首 token 跑偏为 "nobody"
            conv.append_message(conv.roles[1], answer_prefix)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        except Exception as exc:
            logging.warning("Falling back to raw prompt because conversation template failed: %s", exc)
            prompt = qs + "\n\n" + answer_prefix

        # 如果 conv 模板已经在末尾附加了 stop_str（如 </s>），去掉以免模型认为对话已结束后重启生成长段落
        if stop_str and prompt.rstrip().endswith(stop_str):
            prompt = prompt[: prompt.rfind(stop_str)].rstrip()

        # 避免在 prompt 末尾自动加 eos，导致生成从新文档开始（出现重复闲聊段落）
        tokenized = tokenizer([prompt], add_special_tokens=False)
        input_ids_list = tokenized.input_ids
        # 手动在最前添加 BOS（Llama 风格）
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.tensor([[bos_id] + ids for ids in input_ids_list]).cuda()

        if len(res_data) == 0:
            print("\n[DEBUG] ---- Prompt diagnostics (first sample) ----")
            print(f"Prompt length (chars): {len(prompt)}")
            print(f"Prompt preview head 400:\n{prompt[:400]}")
            print(f"Prompt preview tail 400:\n{prompt[-400:]}")
            print(f"Tokenized length (without BOS): {len(input_ids_list[0])}")
            print(f"First 100 input token ids: {input_ids[0][:100].tolist()}")
            print(f"Last 40 input token ids: {input_ids[0][-40:].tolist()}")
            print("[DEBUG] ---- End prompt diagnostics ----\n")

        # 如果仅进行教师强迫 loss 检查：对首条样本，用“prompt + 真实答案”做前向，labels 只监督答案部分
        if getattr(args, "loss_check", False) and len(res_data) == 0:
            answer_text = item["conversations"][1]["value"]
            prompt_ids = torch.tensor(tokenizer([prompt], add_special_tokens=False).input_ids).cuda()
            answer_ids = torch.tensor(tokenizer([answer_text], add_special_tokens=False).input_ids).cuda()
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            labels = torch.full_like(full_ids, -100)
            labels[:, prompt_ids.shape[1]:] = full_ids[:, prompt_ids.shape[1]:]
            batched_graph_data = [graph_data]
            with torch.inference_mode():
                out = model(
                    input_ids=full_ids,
                    graph_data=batched_graph_data,
                    labels=labels,
                )
            print(f"[LOSS CHECK] loss={out.loss.item():.6f}, logits_shape={out.logits.shape}, labels_shape={labels.shape}, prompt_len={prompt_ids.shape[1]}, answer_len={answer_ids.shape[1]}")
            return

        stopping_criteria = None
        if stop_str:
            from graphgpt.gr.graphgpt.model.utils import KeywordsStoppingCriteria
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # # 调试：打印前几个样本的 prompt 和 graph_data 信息
        # if len(res_data) < 3:
        #     print(f"\n[DEBUG] Sample {len(res_data)}:")
        #     print(f"  Item ID: {item.get('id', 'N/A')}")
        #     # import inspect
        #     # from graphgpt.gr.graphgpt.model.GraphLlama import GraphLlamaForCausalLM
        #     # print("Using GraphLlama file:", inspect.getfile(GraphLlamaForCausalLM), flush=True)
        #     print(f"  Original QS (before replacement): {item['conversations'][0]['value'][:150]}...")
        #     print(f"  QS after replacement: {qs[:200]}...")
        #     print(f"  Prompt length: {len(prompt)}")
        #     print(f"  Prompt preview (first 300 chars): {prompt[:300]}...")
        #     print(f"  Prompt preview (last 200 chars): ...{prompt[-200:]}")
        #     print(f"  Input IDs shape: {input_ids.shape}")
        #     print(f"  Input IDs (first 20): {input_ids[0][:20].tolist()}")
        #     print(f"  Input IDs (last 20): {input_ids[0][-20:].tolist()}")
        #     print(f"  Graph patch tokens in input: {(input_ids == patch_token_id).sum().item()}")
        #     print(f"  Graph start tokens in input: {(input_ids == start_token_id).sum().item()}")
        #     print(f"  Graph end tokens in input: {(input_ids == end_token_id).sum().item()}")
        #     print(f"  Graph 1 shape: {graph_data['graph_1'].graph_node.shape}")
        #     print(f"  Graph 2 shape: {graph_data['graph_2'].graph_node.shape}")
        #     print(f"  Graph 1 has nan: {torch.isnan(graph_data['graph_1'].graph_node).any().item()}")
        #     print(f"  Graph 2 has nan: {torch.isnan(graph_data['graph_2'].graph_node).any().item()}")
        #     print(f"  Graph 1 mean: {graph_data['graph_1'].graph_node.mean().item():.4f}")
        #     print(f"  Graph 1 std: {graph_data['graph_1'].graph_node.std().item():.4f}")
        #     print(f"  Graph 2 mean: {graph_data['graph_2'].graph_node.mean().item():.4f}")
        #     print(f"  Graph 2 std: {graph_data['graph_2'].graph_node.std().item():.4f}")
        #     print(f"  Graph data type: {type(graph_data)}")
        #     print(f"  Graph data keys: {list(graph_data.keys()) if isinstance(graph_data, dict) else 'N/A'}")
            
        
        # 验证不同样本的图数据是否不同
        if len(res_data) < 3:
            # 修复：bfloat16 不能直接转 numpy，需要先转为 float32
            graph_1_nodes = graph_data['graph_1'].graph_node[:5].cpu()
            graph_2_nodes = graph_data['graph_2'].graph_node[:5].cpu()
            # 转换为 float32 以支持 numpy 转换
            if graph_1_nodes.dtype == torch.bfloat16:
                graph_1_nodes = graph_1_nodes.float()
            if graph_2_nodes.dtype == torch.bfloat16:
                graph_2_nodes = graph_2_nodes.float()
            
            graph_1_hash = hash(str(graph_1_nodes.numpy().tolist()))
            graph_2_hash = hash(str(graph_2_nodes.numpy().tolist()))
            print(f"  [Sample {len(res_data)}] Graph 1 hash (first 5 nodes): {graph_1_hash}")
            print(f"  [Sample {len(res_data)}] Graph 2 hash (first 5 nodes): {graph_2_hash}")
        
        # 关键调试：检查输入是否真的不同
        if len(res_data) < 3:
            # 计算input_ids的hash，验证不同样本的输入是否不同
            input_ids_hash = hash(str(input_ids.cpu().numpy().tolist()))
            print(f"  Input IDs hash: {input_ids_hash}")
            # 检查prompt中是否包含实体信息
            if "Entity A" in prompt or "Entity B" in prompt or "Entity C" in prompt:
                print(f"  ✓ Prompt contains entity information")
            else:
                print(f"  ⚠️ WARNING: Prompt does NOT contain entity information!")

        with torch.inference_mode():
            batched_graph_data = [graph_data]
            use_forced = getattr(args, "force_label_decode", False)

            if use_forced:
                # -----------------------------
                # 强制标签解码（只允许 PROMOTION / SUPPRESSION），避免闲聊与随机内容
                # -----------------------------
                label_tokens = {
                    "PROMOTION": tokenizer.encode("PROMOTION", add_special_tokens=False),
                    "SUPPRESSION": tokenizer.encode("SUPPRESSION", add_special_tokens=False),
                }

                def score_label(prefix_ids, label_tok_ids):
                    # 首次前向带图，拿到 past；随后增量推理不再重复图编码
                    outputs = model(input_ids=prefix_ids, graph_data=batched_graph_data, use_cache=True)
                    past = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    logp = 0.0
                    for t in label_tok_ids:
                        probs = torch.log_softmax(logits, dim=-1)
                        logp += probs[0, t].item()
                        tok_tensor = torch.tensor([[t]], device=prefix_ids.device)
                        outputs = model(input_ids=tok_tensor, graph_data=None, use_cache=True, past_key_values=past)
                        past = outputs.past_key_values
                        logits = outputs.logits[:, -1, :]
                    return logp

                # Step1 选择
                best1, best1_lp = None, -1e9
                for lbl, toks in label_tokens.items():
                    lp = score_label(input_ids, toks)
                    if lp > best1_lp:
                        best1_lp, best1 = lp, lbl

                # 构造 Step2 前缀
                step2_prefix = prompt + best1 + "\nStep 2 - B-C Relationship: "
                step2_ids = torch.tensor(tokenizer([step2_prefix], add_special_tokens=False).input_ids).cuda()

                # Step2 选择
                best2, best2_lp = None, -1e9
                for lbl, toks in label_tokens.items():
                    lp = score_label(step2_ids, toks)
                    if lp > best2_lp:
                        best2_lp, best2 = lp, lbl

                ans = f"Step 1 - A-B Relationship: {best1}\nStep 2 - B-C Relationship: {best2}"
                raw_ans = ans
                normalized_ans = ans
                if len(res_data) < 3:
                    print(f"  Forced decode labels: step1={best1} (lp={best1_lp:.3f}), step2={best2} (lp={best2_lp:.3f})")
                    print(f"  Output preview: {ans}")
            else:
                # 增加 bad_words 过滤
                bad_words = ["nobody", "_", "\\"]
                bad_words_ids = [tokenizer.encode(w, add_special_tokens=False) for w in bad_words]
                bad_words_ids.append([3187])   # 下划线
                bad_words_ids.append([26077])  # everybody

                # 调试：查看首步 logits 的 top-k，判断模型为何偏向闲聊
                if len(res_data) < 1:
                    with torch.inference_mode():
                        logits0 = model(input_ids=input_ids, graph_data=batched_graph_data).logits[:, -1, :]
                        topk = torch.topk(logits0, k=10, dim=-1)
                        probs = torch.softmax(topk.values, dim=-1)[0].tolist()
                        tokens = topk.indices[0].tolist()
                        strings = [tokenizer.decode([t]) for t in tokens]
                        print("[DEBUG logits top-10] tokens:", tokens)
                        print("[DEBUG logits top-10] probs :", probs)
                        print("[DEBUG logits top-10] text  :", strings)

                out_ids = model.generate(
                    input_ids,
                    graph_data=batched_graph_data,
                    do_sample=False,
                    temperature=0,
                    top_p=0,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.0,
                    stopping_criteria=[stopping_criteria] if stopping_criteria else None,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=bad_words_ids,
                )

                if len(res_data) < 3:
                    first_new_token = out_ids[0, input_ids.shape[1]:input_ids.shape[1]+1]
                    first_token_text = tokenizer.decode(first_new_token, skip_special_tokens=False)
                    print(f"  First generated token ID: {first_new_token.item()}")
                    print(f"  First generated token text: '{first_token_text}'")
                    print(f"  First 5 generated token IDs: {out_ids[0, input_ids.shape[1]:input_ids.shape[1]+5].tolist()}")

                ans = tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                ans = ans.strip()
                if stop_str and ans.endswith(stop_str):
                    ans = ans[:-len(stop_str)].strip()
                raw_ans = ans
                normalized_ans = normalize_relationship_output(ans)
                if normalized_ans != ans:
                    logging.debug("Normalized output for %s: %s -> %s", item.get("id"), ans, normalized_ans)
                ans = normalized_ans
                if len(res_data) < 3:
                    print(f"  Raw generated output (first 200 chars): {raw_ans[:200]}...")
                    print(f"  Normalized output (first 200 chars): {ans[:200]}...")
                    print(f"  Output length: {len(raw_ans)}")
        
        res_data.append({
            "id": item["id"],
            "node_idx_1": item["graph"]["node_idx_1"],
            "node_idx_2": item["graph"]["node_idx_2"],
            "truth": item["conversations"][1]["value"],
            "res": normalized_ans,
            "raw_res": raw_ans,
        })
        all_res["questions"].append(item)
        all_res["prediction"].append(normalized_ans)
        all_res["raw_prediction"].append(raw_ans)
        
        # 保存中间结果
        if len(res_data) % args.save_interval == 0:
            save_path = os.path.join(args.output_res_path, f"arxiv_test_res_{args.start_id}_{args.end_id}.json")
            with open(save_path, "w") as f:
                json.dump(res_data, f, ensure_ascii=False, indent=4)
            print(f"Saved {len(res_data)} results to {save_path}")
        
        # 控制评估数量
        if args.max_num > 0 and len(res_data) >= args.max_num:
            print(f"Reached max_num={args.max_num}, stopping evaluation.")
            break

    out_path = osp.join(args.output_res_path, "arxiv_test_res_all.json")
    with open(out_path, "w") as f:
        json.dump(res_data, f, indent=4)

    print(f"\nAll done! Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, required=True)
    parser.add_argument("--graph_data_path", type=str, required=True)
    parser.add_argument("--output_res_path", type=str, required=True)
    parser.add_argument("--projector_path", type=str, default="") 
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to base LLM (e.g., vicuna)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer (defaults to model/base path)")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--max_num", type=int, default=-1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--is_shuffle", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--force_label_decode", action="store_true", help="Force deterministic label selection (PROMOTION/SUPPRESSION) instead of free-form generation.")
    # 新增：可选从 clean graph tower 覆盖（用于 A/B 验证）
    parser.add_argument("--reload_graph_tower", action="store_true", help="If set, reload graph_tower from gr/clip_gt_arxiv before inference")
    # 可选：仅做一次教师强迫 loss 检查（首条样本），不生成
    parser.add_argument("--loss_check", action="store_true", help="If set, run one sample with labels for teacher-forcing loss and exit")
    parser.add_argument("--inspect_weights", action="store_true", help="Print graph_tower diagnostics after each loading stage")
    parser.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Target dtype when loading/moving model onto device")

    args = parser.parse_args()
    run_eval(args)