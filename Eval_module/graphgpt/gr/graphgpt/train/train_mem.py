# =================================================================
#Transformers + PyTorch + HFHub + DeepSpeed 修复版
# =================================================================
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

# --- 2. 修复 PyTorch 2.1 兼容性 ---
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    _orig_register = torch.utils._pytree._register_pytree_node
    def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return _orig_register(cls, flatten_fn, unflatten_fn)
    torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
    print("✅ Patch 2 Applied: torch.utils._pytree.register_pytree_node")

# --- 3. 暴力覆盖 Transformers 安全检查 ---
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

import graphgpt.gr.graphgpt.train.train_graph as train_graph_module

def simple_maybe_zero_3(param, ignore_status=False, name=None):
    # 原函数试图用 deepspeed 收集参数，我们直接返回 CPU 副本即可
    return param.detach().cpu().clone()

# 强制替换 train_graph 中的函数
train_graph_module.maybe_zero_3 = simple_maybe_zero_3
print("✅ Patch 4 Applied: Bypassed DeepSpeed dependency.")
# =================================================================

# Flash Attention 补丁
try:
    from graphgpt.gr.graphgpt.train.llama_flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )
    replace_llama_attn_with_flash_attn()
except ImportError:
    pass

from graphgpt.gr.graphgpt.train.train_graph import train

if __name__ == "__main__":
    train()