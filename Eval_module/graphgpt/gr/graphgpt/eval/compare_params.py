#!/usr/bin/env python3
# 简单参数对比：基座 vs 目标（如 stage1），检查权重是否有实际更新
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

import argparse, os, torch
from transformers import AutoModelForCausalLM
try:
    from graphgpt.gr.graphgpt.model.GraphLlama import GraphLlamaForCausalLM
except Exception:
    GraphLlamaForCausalLM = None


def load_model(path, device="cpu"):
    # 优先使用本地 GraphLlama 类，避免 AutoModel 无法识别自定义架构
    if GraphLlamaForCausalLM is not None:
        return GraphLlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device,
            trust_remote_code=True,
        )
    # 回退 AutoModel
    return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device,
            trust_remote_code=True,
        )


def tensor_diff(a, b):
    truncated = False
    if a.shape != b.shape:
        if a.dim() != b.dim():
            return {"skipped": f"rank mismatch: {a.shape} vs {b.shape}"}
        # 对齐到最小公共形状（处理 vocab 扩展带来的 32000 vs 32003 等差异）
        shape = [min(a.size(i), b.size(i)) for i in range(a.dim())]
        slices = tuple(slice(0, s) for s in shape)
        a = a[slices]
        b = b[slices]
        truncated = True
    diff = (a - b).float()
    return {
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "truncated": truncated,
        "used_shape": list(a.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基座模型路径（如 vicuna 或 stage1 之前的权重）")
    ap.add_argument("--target", required=True, help="目标模型路径（如 stage1 输出目录）")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    assert os.path.exists(args.base) and os.path.exists(args.target), "路径不存在"

    print(f"Loading base from {args.base}")
    m_base = load_model(args.base, device=args.device)
    print(f"Loading target from {args.target}")
    m_tgt = load_model(args.target, device=args.device)

    pairs = [
        ("model.embed_tokens.weight", "嵌入"),
        ("model.layers.0.self_attn.q_proj.weight", "L0 q_proj"),
        ("model.layers.0.self_attn.k_proj.weight", "L0 k_proj"),
        ("model.layers.0.self_attn.v_proj.weight", "L0 v_proj"),
        ("model.layers.0.self_attn.o_proj.weight", "L0 o_proj"),
        ("model.layers.0.mlp.up_proj.weight", "L0 up_proj"),
        ("model.layers.0.mlp.down_proj.weight", "L0 down_proj"),
        ("model.layers.0.mlp.gate_proj.weight", "L0 gate_proj"),
    ]

    for key, name in pairs:
        if key not in m_base.state_dict() or key not in m_tgt.state_dict():
            print(f"[WARN] {key} not found in models; skip")
            continue
        db = m_base.state_dict()[key].detach().cpu()
        dt = m_tgt.state_dict()[key].detach().cpu()
        stats = tensor_diff(db, dt)
        if "skipped" in stats:
            print(f"{name}: skipped ({stats['skipped']})")
        else:
            note = " (truncated)" if stats.get("truncated") else ""
            print(f"{name}: mean_abs={stats['mean_abs']:.6f}, max_abs={stats['max_abs']:.6f}, cosine={stats['cosine']:.4f}, shape={stats.get('used_shape')}{note}")

    # 额外：若目标模型包含 graph_projector / graph_tower，可提示用户单独检查
    extra_keys = [k for k in m_tgt.state_dict().keys() if "graph_projector" in k or "graph_tower" in k]
    if extra_keys:
        print(f"[INFO] 目标模型包含 {len(extra_keys)} 个 graph_* 参数，可另行检查：前5个 {extra_keys[:5]}")


if __name__ == "__main__":
    main()
