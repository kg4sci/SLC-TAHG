#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from graphgpt.gr.graphgpt.model.graph_layers import MPNN, GNN, CLIP, graph_transformer
from torch_geometric.data import Data
import json
import os.path as osp
import glob

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"




class GraphLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"#继承自 LlamaConfig，并将 model_type 设置为 GraphLlama
    # 可以看到它支持不同的 graph_tower 类型（如 MPNN、GNN、Graph Transformer 等）
class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)#用于从配置字典中加载图模型的相关参数。它将配置项转化为类属性，以便后续模型初始化时使用。

def load_model_pretrained(model_name, pretrain_model_path): 
    # load conig json
    # 加载预训练的图模型（如 CLIP），并将其参数转移到 GraphLlama 的图编码器（如 MPNN、GNN 或 Graph Transformer）
    
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
        # print(config_dict,"config_dict********")
    args = GraphPretrainConfig(config_dict)
    print(args)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pretrain_model_path}")
    
    state_dict = torch.load(pkl_files[0], map_location="cpu")
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')

    # 检查加载的权重是否包含 nan/inf
    import logging
    has_invalid_values = False
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # 确保在 CPU 上检查（从 pkl 加载的应该已经在 CPU 上）
            v_cpu = v.cpu() if v.device.type != 'cpu' else v
            nan_count = torch.isnan(v_cpu).sum().item()
            inf_count = torch.isinf(v_cpu).sum().item()
            if nan_count > 0 or inf_count > 0:
                total_elements = v_cpu.numel()
                invalid_ratio = (nan_count + inf_count) / total_elements
                
                logging.warning(
                    f"[load_model_pretrained] Weight '{k}' contains nan/inf! "
                    f"nan_count={nan_count}, inf_count={inf_count}, "
                    f"invalid_ratio={invalid_ratio:.2%}, "
                    f"Shape: {v.shape}, dtype: {v.dtype}"
                )
                has_invalid_values = True
                
                # 如果超过 50% 的值是无效的，完全重新初始化该权重
                if invalid_ratio > 0.5:
                    logging.warning(
                        f"[load_model_pretrained] Weight '{k}' has >50% invalid values ({invalid_ratio:.2%}). "
                        f"Completely reinitializing this weight."
                    )
                    # 使用 Xavier/Glorot 初始化（对于线性层）或 Kaiming 初始化
                    if len(v.shape) >= 2:
                        # 线性层：使用 Xavier uniform
                        bound = (6.0 / (v.shape[0] + v.shape[1])) ** 0.5
                        fixed_tensor = torch.empty_like(v_cpu).uniform_(-bound, bound)
                    else:
                        # 其他层：使用小的随机值
                        fixed_tensor = torch.randn_like(v_cpu) * 0.01
                else:
                    # 修复：将 nan/inf 替换为基于有效值的随机值
                    valid_mask = ~(torch.isnan(v_cpu) | torch.isinf(v_cpu))
                    if valid_mask.any():
                        valid_values = v_cpu[valid_mask]
                        mean_val = valid_values.mean().item()
                        std_val = valid_values.std().item()
                        # 使用更保守的修复：限制标准差，避免过大值
                        std_val = min(std_val, 1.0)  # 限制标准差
                        replacement = torch.randn_like(v_cpu) * (std_val * 0.1 + 1e-6) + mean_val
                        # 进一步 clamp 到合理范围
                        replacement = torch.clamp(replacement, min=-2.0, max=2.0)
                    else:
                        # 如果全部无效，使用小的随机初始化
                        replacement = torch.randn_like(v_cpu) * 0.01
                        replacement = torch.clamp(replacement, min=-0.1, max=0.1)
                    
                    # 修复后的张量
                    fixed_tensor = torch.where(
                        torch.isnan(v_cpu) | torch.isinf(v_cpu),
                        replacement,
                        v_cpu
                    )
                
                # 如果原始张量不在 CPU，移回原设备
                if v.device.type != 'cpu':
                    fixed_tensor = fixed_tensor.to(v.device)
                state_dict[k] = fixed_tensor
    
    if has_invalid_values:
        logging.warning(
            f"[load_model_pretrained] Found and fixed invalid values in pretrained weights from {pkl_files[0]}. "
            f"This may indicate issues with the pretrained model file."
        )

    # 兼容输入维度变化：只加载形状匹配的权重，其他层保持随机初始化
    model_sd = model.state_dict()
    filtered_sd = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f"[load_model_pretrained] skipped {len(skipped)} keys due to shape mismatch: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    model.load_state_dict(filtered_sd, strict=False)

    return model, args
def transfer_param_tograph(clip_graph, gnn):
    
    print(clip_graph)
    gnn_state_dict = clip_graph.gnn.state_dict()
    
    # 检查转换的权重是否包含 nan/inf
    import logging
    has_invalid_values = False
    processed_dict = {}
    for k, v in gnn_state_dict.items():
        if isinstance(v, torch.Tensor):
            # 如果张量在 Meta 设备上，无法直接检查，直接使用（通常是新初始化的，不会有问题）
            if v.device.type == 'meta':
                processed_dict[k] = v
                continue
            
            # 确保在 CPU 上进行检查
            v_cpu = v.cpu() if v.device.type != 'cpu' else v
            try:
                nan_count = torch.isnan(v_cpu).sum().item()
                inf_count = torch.isinf(v_cpu).sum().item()
            except Exception as e:
                # 如果无法检查（例如在 Meta 设备上），直接使用原值
                logging.debug(f"[transfer_param_tograph] Cannot check weight '{k}' (device: {v.device}): {e}")
                processed_dict[k] = v
                continue
            
            if nan_count > 0 or inf_count > 0:
                total_elements = v_cpu.numel()
                invalid_ratio = (nan_count + inf_count) / total_elements
                
                logging.warning(
                    f"[transfer_param_tograph] Weight '{k}' contains nan/inf! "
                    f"nan_count={nan_count}, inf_count={inf_count}, "
                    f"invalid_ratio={invalid_ratio:.2%}, "
                    f"Shape: {v.shape}, dtype: {v.dtype}, device: {v.device}"
                )
                has_invalid_values = True
                
                # 确保在正确的设备上操作
                device = v.device
                
                # 如果超过 50% 的值是无效的，完全重新初始化该权重
                if invalid_ratio > 0.5:
                    logging.warning(
                        f"[transfer_param_tograph] Weight '{k}' has >50% invalid values ({invalid_ratio:.2%}). "
                        f"Completely reinitializing this weight."
                    )
                    # 使用 Xavier/Glorot 初始化（对于线性层）或 Kaiming 初始化
                    if len(v.shape) >= 2:
                        # 线性层：使用 Xavier uniform
                        bound = (6.0 / (v.shape[0] + v.shape[1])) ** 0.5
                        fixed_v = torch.empty_like(v_cpu).uniform_(-bound, bound)
                    else:
                        # 其他层：使用小的随机值
                        fixed_v = torch.randn_like(v_cpu) * 0.01
                else:
                    # 修复：将 nan/inf 替换为基于有效值的随机值
                    v_fixed = v.cpu() if v.device.type != 'cpu' else v.clone()
                    valid_mask = ~(torch.isnan(v_fixed) | torch.isinf(v_fixed))
                    if valid_mask.any():
                        valid_values = v_fixed[valid_mask]
                        mean_val = valid_values.mean().item()
                        std_val = valid_values.std().item()
                        # 使用更保守的修复：限制标准差，避免过大值
                        std_val = min(std_val, 1.0)  # 限制标准差
                        replacement = torch.randn_like(v_fixed) * (std_val * 0.1 + 1e-6) + mean_val
                        # 进一步 clamp 到合理范围
                        replacement = torch.clamp(replacement, min=-2.0, max=2.0)
                    else:
                        # 如果全部无效，使用小的随机初始化
                        replacement = torch.randn_like(v_fixed) * 0.01
                        replacement = torch.clamp(replacement, min=-0.1, max=0.1)
                    
                    fixed_v = torch.where(
                        torch.isnan(v_fixed) | torch.isinf(v_fixed),
                        replacement,
                        v_fixed
                    )
                
                # 移回原始设备
                if device.type != 'cpu':
                    fixed_v = fixed_v.to(device)
                # 移回原始设备
                if device.type != 'cpu':
                    fixed_v = fixed_v.to(device)
                processed_dict[k] = fixed_v
            else:
                processed_dict[k] = v
        else:
            processed_dict[k] = v
    
    gnn_state_dict = processed_dict
    
    if has_invalid_values:
        logging.warning(
            f"[transfer_param_tograph] Found and fixed invalid values during weight transfer. "
            f"This may indicate issues with the pretrained CLIP model."
        )
    
    gnn.load_state_dict(gnn_state_dict)
    return gnn


class GraphLlamaModel(LlamaModel):
    config_class = GraphLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)

        if hasattr(config, "graph_tower"):
            # 根据配置文件中的 graph_tower 参数，选择不同的图网络结构（如 MPNN、clip_gcn_arxiv、graph_transformer）
            # HACK: for FSDP
            # self.vision_tower = [CLIPVisionModel.from_pretrained(config.graph_tower)]
            # self.arxiv_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)
            if config.graph_tower == 'MPNN': 
                self.graph_tower = MPNN(in_channels = config.graph_hidden_size, hidden_channels = config.graph_hidden_size * 2, out_channels = config.graph_hidden_size, dropout = 0.1, num_layers = 2, if_param = False)
            elif config.graph_tower == "clip_gcn_arxiv": 

                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path)
                self.graph_tower = GNN(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt":
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv": 
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)
            elif config.graph_tower == "clip_gt_arxiv_pub": 
                clip_graph, args= load_model_pretrained(CLIP, config.pretrain_graph_model_path) 
                self.graph_tower = graph_transformer(args)
                self.graph_tower = transfer_param_tograph(clip_graph, self.graph_tower)

            

            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_graph_proj"):
            # graph_projector 在 initialize_graph_modules 里根据实际图塔输出维度重新创建
            # 这里占个位，避免属性缺失
            self.graph_projector = None
    # 通过 get_graph_tower 和 initialize_graph_modules 函数，图模型模块会初始化并加载预训练权重
    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,
                                  pretrain_graph_mlp_adapter=None, fsdp=None): # TODO: modify this function
        self.config.graph_tower = graph_tower


        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower == 'MPNN': 
                graph_tower = MPNN(in_channels = self.config.graph_hidden_size, hidden_channels = self.config.graph_hidden_size * 2, out_channels = self.config.graph_hidden_size, dropout = 0.1, num_layers = 2, if_param = False)
            elif self.config.graph_tower == "clip_gcn_arxiv": 

                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = GNN(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            # graph_tower = MPNN(in_channels = self.config.graph_hidden_size, hidden_channels = self.config.graph_hidden_size * 2, out_channels = self.config.graph_hidden_size, dropout = 0.1, num_layers = 2)
            elif self.config.graph_tower == "clip_gt_arxiv":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path)
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
            elif self.config.graph_tower == "clip_gt_arxiv_pub":
                clip_graph, args= load_model_pretrained(CLIP, self.config.pretrain_graph_model_path) 
                graph_tower = graph_transformer(args)
                graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        else:
            graph_tower = self.graph_tower
        graph_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower

        

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        # 确定图塔输出维度（优先从预训练图配置里取 gnn_output）
        graph_hidden_size = getattr(getattr(graph_tower, "args", None), "gnn_output", None)
        if graph_hidden_size is None:
            graph_hidden_size = getattr(self.config, "graph_hidden_size", None)
        if graph_hidden_size is None:
            # 回退：若仍为空，则用 LLM hidden_size（不理想但可运行）
            graph_hidden_size = self.config.hidden_size
        self.config.graph_hidden_size = graph_hidden_size

        # 始终用当前的 graph_hidden_size 重新创建 projector，确保形状匹配
        self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 基本前向调试：打印前几次调用的 graph_data / seq_len / past_key_values 状态
        if not self.training:
            if not hasattr(self, "_dbg_forward_calls"):
                self._dbg_forward_calls = 0
            if self._dbg_forward_calls < 3:
                gd_type = type(graph_data).__name__ if graph_data is not None else "None"
                gd_len = len(graph_data) if hasattr(graph_data, "__len__") else "N/A"
                print(f"[forward dbg] call={self._dbg_forward_calls}, seq_len={input_ids.shape[1]}, "
                      f"past_key_values is None={past_key_values is None}, graph_data type={gd_type}, len={gd_len}", flush=True)
                self._dbg_forward_calls += 1

        graph_tower = self.get_graph_tower()
        # 关键修复：在生成过程中，第一次 forward 时（input_ids.shape[1] > 1）需要处理图数据
        # 后续步骤（input_ids.shape[1] == 1）时，图信息已经在第一次 forward 时被嵌入到序列中
        should_process_graph = (
            graph_tower is not None 
            and graph_data is not None 
            and (input_ids.shape[1] != 1 or self.training)
        )
        # 调试：更详细地打印 graph_data 状态，确认是否会进入图处理分支
        if not self.training and past_key_values is None:
            import logging
            if graph_data is None:
                msg = f"!! graph not processed: graph_data is None, seq_len={input_ids.shape[1]}"
                logging.warning(msg)
                print(msg, flush=True)
            elif not isinstance(graph_data, list):
                msg = f"!! graph not processed: graph_data type={type(graph_data)}, seq_len={input_ids.shape[1]}"
                logging.warning(msg)
                print(msg, flush=True)
            elif len(graph_data) == 0:
                msg = f"!! graph not processed: graph_data list is empty, seq_len={input_ids.shape[1]}"
                logging.warning(msg)
                print(msg, flush=True)
            elif not should_process_graph:
                msg = f"!! graph not processed: seq_len={input_ids.shape[1]}, training={self.training}"
                logging.warning(msg)
                print(msg, flush=True)
            else:
                first_item = graph_data[0]
                keys = list(first_item.keys()) if isinstance(first_item, dict) else "N/A"
                msg = (f"!! graph WILL be processed, batch={len(graph_data)}, "
                       f"seq_len={input_ids.shape[1]}, first_item_type={type(first_item)}, keys={keys}")
                logging.warning(msg)
                print(msg, flush=True)
        # 调试信息：记录图数据处理情况
        if graph_tower is not None and graph_data is not None and not self.training:
            import logging
            if input_ids.shape[1] == 1:
                logging.debug(
                    f"Generation step: input_ids.shape={input_ids.shape}, "
                    f"skipping graph processing (graph info already embedded in sequence)"
                )
            else:
                graph_data_type = type(graph_data).__name__
                graph_data_len = len(graph_data) if isinstance(graph_data, (list, dict)) else 'N/A'
                if isinstance(graph_data, list) and len(graph_data) > 0:
                    first_item_type = type(graph_data[0]).__name__
                    logging.info(
                        f"[GRAPH DEBUG] First forward: input_ids.shape={input_ids.shape}, "
                        f"graph_data type={graph_data_type}, length={graph_data_len}, "
                        f"first_item type={first_item_type}"
                    )
                    if isinstance(graph_data[0], dict):
                        logging.info(
                            f"[GRAPH DEBUG] Graph dict keys: {list(graph_data[0].keys())}, "
                            f"graph_1 has graph_node: {hasattr(graph_data[0].get('graph_1'), 'graph_node')}, "
                            f"graph_2 has graph_node: {hasattr(graph_data[0].get('graph_2'), 'graph_node')}"
                        )
                else:
                    logging.info(
                        f"[GRAPH DEBUG] First forward: input_ids.shape={input_ids.shape}, "
                        f"graph_data type={graph_data_type}, length={graph_data_len}"
                    )
        
        if should_process_graph:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            requires_grad_graph = self.training and any(
                param.requires_grad for param in graph_tower.parameters()
            )
            grad_context = nullcontext if requires_grad_graph else torch.no_grad
            with grad_context():
                # 保存原始 dtype 和设备，用于后续恢复
                original_dtype = next(graph_tower.parameters()).dtype
                original_device = next(graph_tower.parameters()).device
                # 临时转换为 fp32 以提高数值稳定性（如果当前不是 fp32）
                use_fp32_for_stability = (original_dtype != torch.float32)
                if use_fp32_for_stability:
                    graph_tower = graph_tower.float()
                if type(graph_data) is list:
                    # variable length images
                    graph_node_features = []
                    if type(graph_data[0]) is Data:
                        # 检查并修复 graph_tower 权重是否有 nan/inf（仅检查一次）
                        import logging
                        if not hasattr(graph_tower, '_weight_checked'):
                            has_nan_weight = False
                            fixed_count = 0
                            for name, param in graph_tower.named_parameters():
                                nan_count = torch.isnan(param).sum().item()
                                inf_count = torch.isinf(param).sum().item()
                                if nan_count > 0 or inf_count > 0:
                                    logging.warning(
                                        f"Graph tower parameter '{name}' contains nan/inf! "
                                        f"nan_count={nan_count}, inf_count={inf_count}. "
                                        f"Attempting to fix by replacing with small random values..."
                                    )
                                    has_nan_weight = True
                                    # 修复：将 nan/inf 替换为小的随机值
                                    with torch.no_grad():
                                        # 获取参数的统计信息
                                        valid_mask = ~(torch.isnan(param) | torch.isinf(param))
                                        if valid_mask.any():
                                            # 如果有有效值，使用有效值的统计信息
                                            valid_values = param[valid_mask]
                                            mean_val = valid_values.mean().item()
                                            std_val = valid_values.std().item()
                                            # 限制 std_val 避免过大值导致数值不稳定
                                            std_val = min(std_val, 1.0)  # 限制标准差
                                            # 用小的随机值替换 nan/inf，使用更保守的初始化
                                            replacement = torch.randn_like(param) * (std_val * 0.1 + 1e-6) + mean_val
                                            # 进一步 clamp 到合理范围
                                            replacement = torch.clamp(replacement, min=-2.0, max=2.0)
                                        else:
                                            # 如果全部是 nan/inf，使用非常小的随机初始化
                                            replacement = torch.randn_like(param) * 0.001
                                            replacement = torch.clamp(replacement, min=-0.1, max=0.1)
                                        
                                        # 只替换 nan/inf 的位置
                                        param.data = torch.where(
                                            torch.isnan(param) | torch.isinf(param),
                                            replacement,
                                            param
                                        )
                                        # 再次检查，确保修复后没有 nan/inf
                                        remaining_nan = torch.isnan(param.data).sum().item()
                                        remaining_inf = torch.isinf(param.data).sum().item()
                                        if remaining_nan > 0 or remaining_inf > 0:
                                            # 如果还有问题，直接用 0 替换
                                            param.data = torch.where(
                                                torch.isnan(param.data) | torch.isinf(param.data),
                                                torch.zeros_like(param.data),
                                                param.data
                                            )
                                        fixed_count += 1
                            
                            if has_nan_weight:
                                logging.warning(
                                    f"Fixed {fixed_count} graph tower parameters with nan/inf. "
                                    f"This may indicate issues with pretrained weights or numerical instability."
                                )
                            graph_tower._weight_checked = True
                        
                        for g in graph_data:
                            # 检查输入数据是否有 nan/inf
                            if hasattr(g, 'graph_node'):
                                input_nan = torch.isnan(g.graph_node).sum().item()
                                input_inf = torch.isinf(g.graph_node).sum().item()
                                if input_nan > 0 or input_inf > 0:
                                    logging.warning(
                                        f"Graph input contains nan/inf! "
                                        f"nan_count={input_nan}, inf_count={input_inf}, "
                                        f"input_stats: min={g.graph_node.min().item():.4f}, "
                                        f"max={g.graph_node.max().item():.4f}"
                                    )
                                    # 修复输入
                                    g.graph_node = torch.where(
                                        torch.isnan(g.graph_node) | torch.isinf(g.graph_node),
                                        torch.zeros_like(g.graph_node),
                                        g.graph_node
                                    )
                            
                            node_forward_out = graph_tower(g)
                            # 如果使用了 fp32，转换回原始精度
                            if use_fp32_for_stability:
                                node_forward_out = node_forward_out.to(dtype=original_dtype)
                            
                            # 检查 graph_tower 输出是否有 nan/inf
                            if torch.isnan(node_forward_out).any() or torch.isinf(node_forward_out).any():
                                logging.warning(
                                    f"Graph tower output contains nan/inf! "
                                    f"nan_count={torch.isnan(node_forward_out).sum().item()}, "
                                    f"inf_count={torch.isinf(node_forward_out).sum().item()}, "
                                    f"output_shape={node_forward_out.shape}"
                                )
                                # 修复：将 nan/inf 替换为 0（因为输入和输出维度可能不同）
                                node_forward_out = torch.where(
                                    torch.isnan(node_forward_out) | torch.isinf(node_forward_out),
                                    torch.zeros_like(node_forward_out),
                                    node_forward_out
                                )
                                # clamp 确保数值稳定
                                node_forward_out = torch.clamp(node_forward_out, min=-50.0, max=50.0)
                            graph_node_features.append(node_forward_out)
                    elif type(graph_data[0]) is dict:
                        # 保存原始 dtype 和设备（如果还没有保存）
                        if 'original_dtype' not in locals():
                            original_dtype = next(graph_tower.parameters()).dtype
                            original_device = next(graph_tower.parameters()).device
                            use_fp32_for_stability = (original_dtype != torch.float32)
                            if use_fp32_for_stability:
                                graph_tower = graph_tower.float()
                        
                        for g_dict in graph_data:
                            # 使用 fp32 进行前向传播以提高数值稳定性
                            graph_dtype = torch.float32 if use_fp32_for_stability else original_dtype
                            # 转换 graph_1
                            if hasattr(g_dict['graph_1'], 'graph_node'):
                                g_dict['graph_1'].graph_node = g_dict['graph_1'].graph_node.to(graph_dtype)
                            # 转换 graph_2
                            if hasattr(g_dict['graph_2'], 'graph_node'):
                                g_dict['graph_2'].graph_node = g_dict['graph_2'].graph_node.to(graph_dtype)
                            
                            # 检查输入数据是否有 nan/inf
                            import logging
                            for g_name, g_data in [("graph_1", g_dict['graph_1']), ("graph_2", g_dict['graph_2'])]:
                                if hasattr(g_data, 'graph_node'):
                                    input_nan = torch.isnan(g_data.graph_node).sum().item()
                                    input_inf = torch.isinf(g_data.graph_node).sum().item()
                                    if input_nan > 0 or input_inf > 0:
                                        logging.warning(
                                            f"Graph input ({g_name}) contains nan/inf! "
                                            f"nan_count={input_nan}, inf_count={input_inf}, "
                                            f"input_stats: min={g_data.graph_node.min().item():.4f}, "
                                            f"max={g_data.graph_node.max().item():.4f}, "
                                            f"mean={g_data.graph_node.mean().item():.4f}"
                                        )
                                        # 修复输入：将 nan/inf 替换为 0
                                        g_data.graph_node = torch.where(
                                            torch.isnan(g_data.graph_node) | torch.isinf(g_data.graph_node),
                                            torch.zeros_like(g_data.graph_node),
                                            g_data.graph_node
                                        )
                            
                            # 检查并修复 graph_tower 权重是否有 nan/inf（仅检查一次，避免重复）
                            if not hasattr(graph_tower, '_weight_checked'):
                                has_nan_weight = False
                                fixed_count = 0
                                for name, param in graph_tower.named_parameters():
                                    nan_count = torch.isnan(param).sum().item()
                                    inf_count = torch.isinf(param).sum().item()
                                    if nan_count > 0 or inf_count > 0:
                                        logging.warning(
                                            f"Graph tower parameter '{name}' contains nan/inf! "
                                            f"nan_count={nan_count}, inf_count={inf_count}. "
                                            f"Attempting to fix by replacing with small random values..."
                                        )
                                        has_nan_weight = True
                                        # 修复：将 nan/inf 替换为小的随机值（保持参数的可训练性）
                                        with torch.no_grad():
                                            # 获取参数的统计信息
                                            valid_mask = ~(torch.isnan(param) | torch.isinf(param))
                                            if valid_mask.any():
                                                # 如果有有效值，使用有效值的统计信息
                                                valid_values = param[valid_mask]
                                                mean_val = valid_values.mean().item()
                                                std_val = valid_values.std().item()
                                                # 限制 std_val 避免过大值导致数值不稳定
                                                std_val = min(std_val, 1.0)  # 限制标准差
                                                # 用更保守的随机值替换 nan/inf
                                                replacement = torch.randn_like(param) * (std_val * 0.1 + 1e-6) + mean_val
                                                # 进一步 clamp 到合理范围
                                                replacement = torch.clamp(replacement, min=-2.0, max=2.0)
                                            else:
                                                # 如果全部是 nan/inf，使用非常小的随机初始化
                                                replacement = torch.randn_like(param) * 0.001
                                                replacement = torch.clamp(replacement, min=-0.1, max=0.1)
                                            
                                            # 只替换 nan/inf 的位置
                                            param.data = torch.where(
                                                torch.isnan(param) | torch.isinf(param),
                                                replacement,
                                                param
                                            )
                                            # 再次检查，确保修复后没有 nan/inf
                                            remaining_nan = torch.isnan(param.data).sum().item()
                                            remaining_inf = torch.isinf(param.data).sum().item()
                                            if remaining_nan > 0 or remaining_inf > 0:
                                                # 如果还有问题，直接用 0 替换
                                                param.data = torch.where(
                                                    torch.isnan(param.data) | torch.isinf(param.data),
                                                    torch.zeros_like(param.data),
                                                    param.data
                                                )
                                            fixed_count += 1
                                
                                if has_nan_weight:
                                    logging.warning(
                                        f"Fixed {fixed_count} graph tower parameters with nan/inf. "
                                        f"This may indicate issues with pretrained weights or numerical instability."
                                    )
                                graph_tower._weight_checked = True
                            
                            node_forward_out_1 = graph_tower(g_dict['graph_1'])
                            node_forward_out_2 = graph_tower(g_dict['graph_2'])
                            
                            # 如果使用了 fp32，转换回原始精度
                            if use_fp32_for_stability:
                                node_forward_out_1 = node_forward_out_1.to(dtype=original_dtype)
                                node_forward_out_2 = node_forward_out_2.to(dtype=original_dtype)
                            
                            # 检查 graph_tower 输出是否有 nan/inf
                            for out_name, out_val, g_data in [("graph_1", node_forward_out_1, g_dict['graph_1']), 
                                                               ("graph_2", node_forward_out_2, g_dict['graph_2'])]:
                                if torch.isnan(out_val).any() or torch.isinf(out_val).any():
                                    logging.warning(
                                        f"Graph tower output ({out_name}) contains nan/inf! "
                                        f"nan_count={torch.isnan(out_val).sum().item()}, "
                                        f"inf_count={torch.isinf(out_val).sum().item()}, "
                                        f"output_shape={out_val.shape}, "
                                        f"input_shape={g_data.graph_node.shape if hasattr(g_data, 'graph_node') else 'N/A'}"
                                    )
                                    # 尝试更智能的修复：使用输入的平均值而不是0
                                    if hasattr(g_data, 'graph_node'):
                                        # 使用输入特征的均值作为fallback（如果输入是有效的）
                                        # 注意：输入维度可能和输出维度不同，需要投影或使用零
                                        # 简单方案：使用零，因为维度不匹配
                                        out_val = torch.where(
                                            torch.isnan(out_val) | torch.isinf(out_val),
                                            torch.zeros_like(out_val),
                                            out_val
                                        )
                                    else:
                                        # 如果没有输入信息，使用0
                                        out_val = torch.where(
                                            torch.isnan(out_val) | torch.isinf(out_val),
                                            torch.zeros_like(out_val),
                                            out_val
                                        )
                                    
                                    # 再次clamp确保数值稳定
                                    out_val = torch.clamp(out_val, min=-50.0, max=50.0)
                                    
                                    if out_name == "graph_1":
                                        node_forward_out_1 = out_val
                                    else:
                                        node_forward_out_2 = out_val
                            
                            graph_node_features.append(node_forward_out_1)
                            graph_node_features.append(node_forward_out_2)
                            
                            # 调试：记录图特征信息
                            import logging
                            logging.info(
                                f"[GRAPH DEBUG] Processed graph_1: shape={node_forward_out_1.shape}, "
                                f"mean={node_forward_out_1.mean().item():.4f}, "
                                f"std={node_forward_out_1.std().item():.4f}"
                            )
                            logging.info(
                                f"[GRAPH DEBUG] Processed graph_2: shape={node_forward_out_2.shape}, "
                                f"mean={node_forward_out_2.mean().item():.4f}, "
                                f"std={node_forward_out_2.std().item():.4f}"
                            )
                else:
                    raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            if type(graph_data) is list:
                # 获取 projector 权重的精度 (fp32)，并将输入强转为该精度
                proj_dtype = self.graph_projector.weight.dtype
                graph_node_features_projected = []
                for node_feature in graph_node_features:
                    # 检查输入是否有 nan/inf
                    if torch.isnan(node_feature).any() or torch.isinf(node_feature).any():
                        import logging
                        logging.warning(
                            f"Graph node features contain nan/inf before projection! "
                            f"nan_count={torch.isnan(node_feature).sum().item()}, "
                            f"inf_count={torch.isinf(node_feature).sum().item()}"
                        )
                        # 修复：将 nan/inf 替换为 0
                        node_feature = torch.where(
                            torch.isnan(node_feature) | torch.isinf(node_feature),
                            torch.zeros_like(node_feature),
                            node_feature
                        )
                    
                    projected = self.graph_projector(node_feature.to(proj_dtype))
                    
                    # 检查投影后的输出是否有 nan/inf
                    if torch.isnan(projected).any() or torch.isinf(projected).any():
                        import logging
                        logging.warning(
                            f"Graph projector output contains nan/inf! "
                            f"nan_count={torch.isnan(projected).sum().item()}, "
                            f"inf_count={torch.isinf(projected).sum().item()}, "
                            f"input_stats: min={node_feature.min().item():.4f}, max={node_feature.max().item():.4f}, "
                            f"projector_weight_stats: min={self.graph_projector.weight.min().item():.4f}, "
                            f"max={self.graph_projector.weight.max().item():.4f}"
                        )
                        # 修复：clamp 到合理范围
                        projected = torch.clamp(projected, min=-100.0, max=100.0)
                        # 如果还有 nan，替换为 0
                        projected = torch.where(
                            torch.isnan(projected) | torch.isinf(projected),
                            torch.zeros_like(projected),
                            projected
                        )
                    
                    graph_node_features_projected.append(projected)
                graph_node_features = graph_node_features_projected
                # else: 
                #     graph_node_features = [{'graph_1': self.graph_projector(node_feature['graph_1']), 'graph_2': self.graph_projector(node_feature['graph_2'])} for node_feature in graph_node_features]
            else:
                raise ValueError(f'graph_node_reps is expected to be a list but got {type(graph_data)}')
            dummy_graph_features = torch.zeros(256, 128, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)

            new_input_embeds = []
            cur_graph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == graph_tower.config.graph_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                if graph_tower.config.use_graph_start_end:#**************
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_start_token).sum() != (cur_input_ids == graph_tower.config.graph_end_token).sum():
                        raise ValueError("The number of graph start tokens and graph end tokens should be the same.")
                    graph_start_tokens = torch.where(cur_input_ids == graph_tower.config.graph_start_token)[0]
                    for graph_start_token_pos in graph_start_tokens:                        
                        cur_graph_features = graph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        if cur_input_ids[graph_start_token_pos + num_patches + 1] != graph_tower.config.graph_end_token:
                            raise ValueError("The graph end token should follow the graph start token.")
                        if orig_embeds_params is not None:                            
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos].detach(), cur_input_embeds[graph_start_token_pos:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:graph_start_token_pos + num_patches + 2], cur_input_embeds[graph_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:                            
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:graph_start_token_pos+1], cur_graph_features, cur_input_embeds[graph_start_token_pos + num_patches + 1:]), dim=0)
                        cur_graph_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_graph_features = graph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    if (cur_input_ids == graph_tower.config.graph_patch_token).sum() != num_patches:
                        raise ValueError("The number of graph patch tokens should be the same as the number of graph patches.")
                    masked_indices = torch.where(cur_input_ids == graph_tower.config.graph_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The graph patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_graph_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_graph_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1

            # print(cur_graph_idx)
            # print(len(graph_node_features))
            assert cur_graph_idx == len(graph_node_features)
            
            # 调试：记录嵌入信息
            if not self.training:
                import logging
                logging.info(
                    f"[GRAPH DEBUG] Graph features processed: {len(graph_node_features)} features, "
                    f"input_ids length: {input_ids.shape[1]}, "
                    f"final inputs_embeds length: {new_input_embeds[0].shape[0]}"
                )
                # 检查图特征是否真的被插入
                if len(graph_node_features) > 0:
                    expected_embeds_len = input_ids.shape[1] - (len(graph_node_features) * 2) + sum(f.shape[0] for f in graph_node_features)
                    actual_embeds_len = new_input_embeds[0].shape[0]
                    logging.info(
                        f"[GRAPH DEBUG] Expected embeds length: {expected_embeds_len}, "
                        f"Actual embeds length: {actual_embeds_len}, "
                        f"Difference: {actual_embeds_len - expected_embeds_len}"
                    )
            
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(GraphLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class GraphLlamaForCausalLM(LlamaForCausalLM):
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GraphLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_data = graph_data
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)  # 显式设置ignore_index
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            # 检查是否有有效的标签（不是所有都被mask）
            valid_labels = (shift_labels != -100)
            valid_count = valid_labels.sum().item()
            if valid_count == 0:
                # 如果所有标签都被mask，这通常表示数据预处理有问题
                # 返回None让Trainer处理，或者返回一个小的非零loss
                import logging
                logging.warning(
                    f"All labels are masked in batch! This indicates a data preprocessing issue. "
                    f"Please check preprocess_v1 function. Returning zero loss."
                )
                # 返回一个小的dummy loss以避免梯度问题
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                # 检查 logits 是否有 nan/inf，这可能是导致 loss nan 的原因
                import logging
                nan_count = torch.isnan(shift_logits).sum().item()
                inf_count = torch.isinf(shift_logits).sum().item()
                if nan_count > 0 or inf_count > 0:
                    logging.warning(
                        f"Logits contain {'nan' if nan_count > 0 else ''} "
                        f"{'inf' if inf_count > 0 else ''}! "
                        f"nan_count={nan_count}, inf_count={inf_count}, "
                        f"valid_labels={valid_count}/{len(shift_labels)}. "
                        f"Logits stats: min={shift_logits.min().item():.4f}, "
                        f"max={shift_logits.max().item():.4f}, "
                        f"mean={shift_logits.mean().item():.4f}"
                    )
                    # 尝试修复：将 nan/inf 替换为有限值
                    shift_logits = torch.where(
                        torch.isnan(shift_logits) | torch.isinf(shift_logits),
                        torch.zeros_like(shift_logits),
                        shift_logits
                    )
                    # 或者使用 clamp 限制范围
                    shift_logits = torch.clamp(shift_logits, min=-50.0, max=50.0)
                
                # 只对有效标签计算 loss
                loss = loss_fct(shift_logits, shift_labels)
                
                # 额外的检查：如果loss是nan或inf，记录警告并尝试修复
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(
                        f"Loss is {'nan' if torch.isnan(loss) else 'inf'}! "
                        f"Valid labels: {valid_count}/{len(shift_labels)}. "
                        f"Attempting to compute loss only on valid labels..."
                    )
                    # 尝试只对有效标签计算 loss
                    valid_mask = valid_labels
                    if valid_mask.sum() > 0:
                        valid_logits = shift_logits[valid_mask]
                        valid_labels_subset = shift_labels[valid_mask]
                        # 使用 reduction='mean' 但只对有效标签
                        loss_fct_valid = CrossEntropyLoss(ignore_index=-100, reduction='mean')
                        loss = loss_fct_valid(valid_logits, valid_labels_subset)
                        if torch.isnan(loss) or torch.isinf(loss):
                            logging.error(
                                f"Loss is still {'nan' if torch.isnan(loss) else 'inf'} even after filtering! "
                                f"Returning small dummy loss."
                            )
                            loss = torch.tensor(1e-6, device=shift_logits.device, requires_grad=True)
                    else:
                        loss = torch.tensor(1e-6, device=shift_logits.device, requires_grad=True)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 关键修复：确保 graph_data 在生成过程中被正确传递
        # 在第一次调用时（past_key_values is None），graph_data 会被处理并嵌入到 inputs_embeds 中
        # 在后续调用时，虽然图数据不会被再次处理（因为 input_ids.shape[1] == 1），
        # 但我们需要确保它仍然被传递，以便 forward 方法知道这是生成过程
        graph_data = kwargs.get("graph_data", None)
        if past_key_values is None:
            import logging
            if graph_data is None:
                msg = f"[prepare_inputs] graph_data is None, seq_len={input_ids.shape[1]}"
                logging.warning(msg); print(msg, flush=True)
            else:
                gd_type = type(graph_data)
                gd_len = len(graph_data) if hasattr(graph_data, '__len__') else 'N/A'
                msg = f"[prepare_inputs] graph_data type={gd_type}, len={gd_len}, seq_len={input_ids.shape[1]}"
                logging.warning(msg); print(msg, flush=True)
        if graph_data is not None:
            # 确保 graph_data 是列表格式（forward 方法期望列表）
            if not isinstance(graph_data, list):
                graph_data = [graph_data]
        else:
            graph_data = [None]

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": graph_data,
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                    tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                
                # 尝试多种可能的键名
                embed_tokens_key = None
                possible_keys = [
                    'model.embed_tokens.weight',
                    'embed_tokens.weight',
                    'model.model.embed_tokens.weight'
                ]
                
                for key in possible_keys:
                    if key in mm_projector_weights:
                        embed_tokens_key = key
                        break
                
                if embed_tokens_key is None:
                    # 如果没有找到 embed_tokens，记录警告但继续（使用默认初始化）
                    import logging
                    available_keys = list(mm_projector_weights.keys())
                    logging.warning(
                        f"[initialize_graph_tokenizer] Could not find embed_tokens.weight in pretrain_graph_mlp_adapter. "
                        f"Available keys: {available_keys[:10]}{'...' if len(available_keys) > 10 else ''}. "
                        f"Will use default initialization for new tokens."
                    )
                else:
                    embed_tokens_weight = mm_projector_weights[embed_tokens_key]
                    assert num_new_tokens == 2
                    if input_embeddings.shape == embed_tokens_weight.shape:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                    elif embed_tokens_weight.shape[0] == num_new_tokens:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight
                    else:
                        import logging
                        logging.warning(
                            f"[initialize_graph_tokenizer] Unexpected embed_tokens_weight shape. "
                            f"Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. "
                            f"Number of new tokens: {num_new_tokens}. Will use default initialization."
                        )

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]

AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)

# 注册 tokenizer 映射，使 AutoTokenizer 能够识别 GraphLlamaConfig
try:
    from transformers import AutoTokenizer, LlamaTokenizer
    # 方法1: 尝试使用 _tokenizer_mapping（如果存在）
    if hasattr(AutoTokenizer, '_tokenizer_mapping'):
        if hasattr(AutoTokenizer._tokenizer_mapping, '_extra_content'):
            AutoTokenizer._tokenizer_mapping._extra_content["GraphLlama"] = (LlamaTokenizer, None)
        else:
            # 备用方法：直接添加到映射字典
            if hasattr(AutoTokenizer, 'register'):
                AutoTokenizer.register(GraphLlamaConfig, LlamaTokenizer)
except Exception as e:
    import logging
    logging.warning(f"Failed to register GraphLlama tokenizer mapping: {e}. Will use LlamaTokenizer directly.")
