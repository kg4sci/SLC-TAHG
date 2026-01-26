"""
GRAN: Graph Recurrent Attention Networks for Cascading Path Prediction

基于GRAN的GNN架构实现级联预测模型，参考NaLP的实现方式：
- 使用GRAN的GNN对路径节点进行编码
- 支持节点特征嵌入（通过编码器投影后与实体embedding融合）
- 支持文本特征嵌入（作为额外的节点特征）
- 实现级联预测：AB段预测rel_AB，BC段使用预测的rel_AB作为条件预测rel_BC
"""
import os
os.environ["DGL_USE_GRAPHBOLT"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNN(nn.Module):

  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim

    self.update_func = nn.ModuleList([
        nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
        for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear(self.node_state_dim + self.edge_feat_dim,
                          self.msg_dim),
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear(self.node_state_dim + self.edge_feat_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, edge, edge_feat, layer_idx=0):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff

    msg = self.msg_func[layer_idx](edge_input)

    ### attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state = self.update_func[layer_idx](state_msg, state)
    return state

  def forward(self, node_feat, edge, edge_feat, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state = node_feat
    prev_state = state
    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state = self._prop(state, edge, edge_feat=edge_feat, layer_idx=ii)

    if self.has_residual:
      state = state + prev_state

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state

class GRANCascadingPredictor(nn.Module):
    """
    基于GRAN的GNN架构实现级联预测模型
    
    - AB 段：使用GNN编码完整路径 [A, Event, B, C] 节点，通过节点状态差异预测 rel_AB
    - BC 段：使用GNN编码完整路径 [A, Event, B, C] 节点，rel_AB只作为额外条件输入（不通过GNN传播），预测 rel_BC
    - 节点特征：投影后与实体embedding相加（增强实体表示）
    - 文本特征：作为额外的节点特征加入GNN输入
    
    重要设计：
    - 预测rel_AB时使用完整路径（SLC、Pathway、Disease），而不是只使用前两个节点
    - 预测rel_BC时使用完整路径（SLC、Pathway、Disease）+ 预测的rel_AB（作为条件输入）
    - 关键修复：rel_AB信息不通过GNN传播，只在预测时作为额外条件输入，避免信息泄露
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relation_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_prop: int,
        dropout: float,
        has_attention: bool,
        node_feat_dim: int,
        text_dim: int,
        use_text_features: bool
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relation_classes = num_relation_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_text_features = use_text_features
        
        # 实体embedding
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # 关系类别embedding（用于级联预测中的CondAB）
        self.relation_class_embeddings = nn.Embedding(num_relation_classes, embedding_dim)
        
        # 节点特征编码器（如果提供节点特征）
        if node_feat_dim > 0:
            self.node_feat_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.node_feat_encoder = None
        
        # 文本特征投影器（如果使用文本特征）
        if use_text_features and text_dim > 0:
            self.text_projector = nn.Sequential(
                nn.Linear(text_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.text_projector = None
        
        # 边特征维度：如果使用文本特征，边特征维度会增加
        # 基础边特征维度（用于区分不同类型的边：A-Event, Event-B, B-Event, Event-C等）
        base_edge_feat_dim = 8  # 用于区分边的类型
        edge_feat_dim = base_edge_feat_dim
        
        # AB段的GNN：编码 [A, Event, B] 节点
        # 输入维度：embedding_dim（实体embedding + 可选的节点特征 + 可选的文本特征）
        self.gnn_ab = GNN(
            msg_dim=hidden_dim,
            node_state_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            num_prop=num_prop,
            num_layer=num_layers,
            has_attention=has_attention,
            att_hidden_dim=hidden_dim // 2,
            has_residual=False,
            has_graph_output=False
        )
        
        # BC段的GNN：编码 [B, Event, C] 节点，并考虑预测的rel_AB
        # 输入维度：embedding_dim + embedding_dim（实体embedding + rel_AB embedding）
        self.gnn_bc = GNN(
            msg_dim=hidden_dim,
            node_state_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            num_prop=num_prop,
            num_layer=num_layers,
            has_attention=has_attention,
            att_hidden_dim=hidden_dim // 2,
            has_residual=False,
            has_graph_output=False
        )
        
        # 输入投影层：将embedding_dim投影到hidden_dim
        self.input_proj_ab = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.input_proj_bc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # 实体embedding + rel_AB embedding
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类头：基于节点状态差异预测关系类别
        self.head_ab = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relation_classes)
        )
        
        self.head_bc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relation_classes)
        )
        
        self._init_params()
    
    def _init_params(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_class_embeddings.weight)
    
    def _get_entity_embedding_with_features(
        self, 
        entity_ids: torch.Tensor, 
        node_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        获取实体embedding，如果提供节点特征则融合
        
        Args:
            entity_ids: [B] 实体ID
            node_features: [B, node_feat_dim] 节点特征（可选）
        Returns:
            [B, embedding_dim] 融合后的实体embedding
        """
        emb = self.entity_embeddings(entity_ids)  # [B, emb]
        if node_features is not None and self.node_feat_encoder is not None:
            feat_emb = self.node_feat_encoder(node_features)  # [B, emb]
            emb = emb + feat_emb  # 相加融合
        return emb
    
    def _build_path_graph_ab(
        self,
        a_emb: torch.Tensor,
        event_emb: torch.Tensor,
        b_emb: torch.Tensor,
        c_emb: torch.Tensor,
        text_ab: torch.Tensor = None,
        device: str = "cpu"
    ) -> tuple:
        """
        构建AB段的完整路径图：A -> Event -> B -> Event -> C（完整路径）
        
        
        Args:
            a_emb: [B, emb] A节点embedding (SLC)
            event_emb: [B, emb] Event节点embedding
            b_emb: [B, emb] B节点embedding (Pathway)
            c_emb: [B, emb] C节点embedding (Disease)
            text_ab: [B, emb] AB段文本特征（可选）
            device: 设备
        
        Returns:
            node_feat: [4*B, hidden_dim] 节点特征（A, Event, B, C）
            edges: [3*B, 2] 边索引（A->Event, Event->B, Event->C）
            edge_feat: [3*B, edge_feat_dim] 边特征
        """
        B = a_emb.size(0)
        
        # 如果使用文本特征，将其投影并加到Event节点上（作为额外的上下文）
        if text_ab is not None and self.text_projector is not None:
            text_emb = self.text_projector(text_ab)  # [B, emb]
            event_emb = event_emb + text_emb
        
        # 投影节点特征到hidden_dim
        a_proj = self.input_proj_ab(a_emb)  # [B, hidden_dim]
        event_proj = self.input_proj_ab(event_emb)  # [B, hidden_dim]
        b_proj = self.input_proj_ab(b_emb)  # [B, hidden_dim]
        c_proj = self.input_proj_ab(c_emb)  # [B, hidden_dim]
        
        # 拼接所有节点：[A, Event, B, C] - 完整路径
        node_feat = torch.cat([a_proj, event_proj, b_proj, c_proj], dim=0)  # [4*B, hidden_dim]
        
        # 构建边：A -> Event, Event -> B, Event -> C（完整路径的边）
        # 节点索引：0..B-1 是A, B..2*B-1 是Event, 2*B..3*B-1 是B, 3*B..4*B-1 是C
        a_indices = torch.arange(B, device=device)
        event_indices = torch.arange(B, B * 2, device=device)
        b_indices = torch.arange(B * 2, B * 3, device=device)
        c_indices = torch.arange(B * 3, B * 4, device=device)
        
        # A -> Event 边
        edges_a_event = torch.stack([a_indices, event_indices], dim=1)  # [B, 2]
        
        # Event -> B 边
        edges_event_b = torch.stack([event_indices, b_indices], dim=1)  # [B, 2]
        
        # Event -> C 边（完整路径）
        edges_event_c = torch.stack([event_indices, c_indices], dim=1)  # [B, 2]
        
        edges = torch.cat([edges_a_event, edges_event_b, edges_event_c], dim=0)  # [3*B, 2]
        
        # 边特征：用于区分不同类型的边
        edge_feat_dim = 8
        edge_feat = torch.zeros(3 * B, edge_feat_dim, device=device)
        edge_feat[:B, 0] = 1.0  # A->Event
        edge_feat[B:2*B, 1] = 1.0  # Event->B
        edge_feat[2*B:, 2] = 1.0  # Event->C
        
        return node_feat, edges, edge_feat
    
    def _build_path_graph_bc(
        self,
        a_emb: torch.Tensor,
        event_emb: torch.Tensor,
        b_emb: torch.Tensor,
        c_emb: torch.Tensor,
        cond_ab_emb: torch.Tensor = None,  # 可选，如果为None则不使用rel_AB
        text_bc: torch.Tensor = None,
        device: str = "cpu"
    ) -> tuple:
        """
        构建BC段的完整路径图：A -> Event -> B -> Event -> C
        
        预测rel_BC时使用完整路径（SLC、Pathway、Disease）
        重要：rel_AB信息不通过GNN传播，只在预测时作为额外条件输入（避免信息泄露）
        
        Args:
            a_emb: [B, emb] A节点embedding (SLC)
            event_emb: [B, emb] Event节点embedding
            b_emb: [B, emb] B节点embedding (Pathway)
            c_emb: [B, emb] C节点embedding (Disease)
            cond_ab_emb: [B, emb] 预测的rel_AB embedding（级联条件，可选，如果为None则不使用）
            text_bc: [B, emb] BC段文本特征（可选）
            device: 设备
        
        Returns:
            node_feat: [4*B, hidden_dim] 节点特征（A, Event, B, C）- 不包含rel_AB信息
            edges: [3*B, 2] 边索引（A->Event, Event->B, Event->C）
            edge_feat: [3*B, edge_feat_dim] 边特征
        """
        B = b_emb.size(0)
        
        # 如果使用文本特征，将其投影并加到Event节点上
        if text_bc is not None and self.text_projector is not None:
            text_emb = self.text_projector(text_bc)  # [B, emb]
            event_emb = event_emb + text_emb
        
        # 所有节点都使用原始embedding，不拼接rel_AB（避免信息泄露）
        # rel_AB只在预测时作为额外条件输入
        a_proj = self.input_proj_ab(a_emb)  # [B, hidden_dim]
        event_proj = self.input_proj_ab(event_emb)  # [B, hidden_dim]
        b_proj = self.input_proj_ab(b_emb)  # [B, hidden_dim] - 不包含rel_AB
        c_proj = self.input_proj_ab(c_emb)  # [B, hidden_dim]
        
        # 拼接所有节点：[A, Event, B, C] - 完整路径（不包含rel_AB信息）
        node_feat = torch.cat([a_proj, event_proj, b_proj, c_proj], dim=0)  # [4*B, hidden_dim]
        
        # 构建边：A -> Event, Event -> B, Event -> C（完整路径的边）
        # 节点索引：0..B-1 是A, B..2*B-1 是Event, 2*B..3*B-1 是B, 3*B..4*B-1 是C
        a_indices = torch.arange(B, device=device)
        event_indices = torch.arange(B, B * 2, device=device)
        b_indices = torch.arange(B * 2, B * 3, device=device)
        c_indices = torch.arange(B * 3, B * 4, device=device)
        
        # A -> Event
        edges_a_event = torch.stack([a_indices, event_indices], dim=1)  # [B, 2]
        # Event -> B
        edges_event_b = torch.stack([event_indices, b_indices], dim=1)  # [B, 2]
        # Event -> C
        edges_event_c = torch.stack([event_indices, c_indices], dim=1)  # [B, 2]
        
        edges = torch.cat([edges_a_event, edges_event_b, edges_event_c], dim=0)  # [3*B, 2]
        
        # 边特征
        edge_feat_dim = 8
        edge_feat = torch.zeros(3 * B, edge_feat_dim, device=device)
        edge_feat[:B, 0] = 1.0  # A->Event
        edge_feat[B:2*B, 1] = 1.0  # Event->B
        edge_feat[2*B:, 2] = 1.0  # Event->C
        
        return node_feat, edges, edge_feat
    
    def forward(
        self,
        a_ids: torch.Tensor,
        b_ids: torch.Tensor,
        c_ids: torch.Tensor,
        event_ids: torch.Tensor,
        training: bool,
        node_features: torch.Tensor = None,  # [num_nodes, node_feat_dim] 或 None
        text_features_ab: torch.Tensor = None,  # [B, text_dim] 或 None
        text_features_bc: torch.Tensor = None   # [B, text_dim] 或 None
    ):
        """
        Args:
            a_ids, b_ids, c_ids, event_ids: [B] int64
            training: 是否使用Gumbel-Softmax来获得 cond rel_AB 的可微 one-hot
            node_features: [num_nodes, node_feat_dim] 节点特征（可选，按entity_id索引）
            text_features_ab: [B, text_dim] AB段的文本特征（可选）
            text_features_bc: [B, text_dim] BC段的文本特征（可选）
        Returns:
            logits_ab, logits_bc, rel_ab_pred
        """
        B = a_ids.size(0)
        device = a_ids.device
        
        # 获取节点特征（如果提供）
        node_feat_a = node_features[a_ids] if node_features is not None else None
        node_feat_b = node_features[b_ids] if node_features is not None else None
        node_feat_c = node_features[c_ids] if node_features is not None else None
        node_feat_event = node_features[event_ids] if node_features is not None else None
        
        # 获取实体embedding（融合节点特征）
        a_emb = self._get_entity_embedding_with_features(a_ids, node_feat_a)  # [B, emb]
        b_emb = self._get_entity_embedding_with_features(b_ids, node_feat_b)  # [B, emb]
        c_emb = self._get_entity_embedding_with_features(c_ids, node_feat_c)  # [B, emb]
        event_emb = self._get_entity_embedding_with_features(event_ids, node_feat_event)  # [B, emb]
        
        # ========== AB段预测（使用完整路径） ==========
        # 构建AB段完整路径图：A -> Event -> B -> Event -> C
        node_feat_ab, edges_ab, edge_feat_ab = self._build_path_graph_ab(
            a_emb, event_emb, b_emb, c_emb, text_features_ab, device
        )
        
        # GNN编码
        node_states_ab = self.gnn_ab(node_feat_ab, edges_ab, edge_feat=edge_feat_ab)  # [4*B, hidden_dim]
        
        # 提取A和B的节点状态，通过差异预测rel_AB
        # 节点索引：0..B-1 是A, B..2*B-1 是Event, 2*B..3*B-1 是B, 3*B..4*B-1 是C
        a_states = node_states_ab[:B]  # [B, hidden_dim]
        b_states = node_states_ab[2*B:3*B]  # [B, hidden_dim]
        
        # 使用节点状态差异预测关系
        diff_ab = a_states - b_states  # [B, hidden_dim]
        logits_ab = self.head_ab(diff_ab)  # [B, num_rel_classes]
        
        # 预测 rel_AB（用于条件化）
        if training:
            soft_onehot = F.gumbel_softmax(logits_ab, tau=1.0, hard=True, dim=-1)
            cond_ab_emb = torch.matmul(soft_onehot, self.relation_class_embeddings.weight)  # [B, emb]
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
        else:
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)  # [B]
            cond_ab_emb = self.relation_class_embeddings(rel_ab_pred)  # [B, emb]
        
        # ========== BC段预测（级联，使用完整路径） ==========
        # 构建BC段完整路径图：A -> Event -> B -> Event -> C（不包含rel_AB，避免信息泄露）
        # rel_AB只作为额外的条件输入，不通过GNN传播
        node_feat_bc, edges_bc, edge_feat_bc = self._build_path_graph_bc(
            a_emb, event_emb, b_emb, c_emb, None, text_features_bc, device  # 不传入cond_ab_emb到图构建
        )
        
        # GNN编码（不包含rel_AB信息）
        node_states_bc = self.gnn_bc(node_feat_bc, edges_bc, edge_feat=edge_feat_bc)  # [4*B, hidden_dim]
        
        # 提取B和C的节点状态（这些状态不包含rel_AB信息）
        # 节点索引：0..B-1 是A, B..2*B-1 是Event, 2*B..3*B-1 是B, 3*B..4*B-1 是C
        b_states_bc = node_states_bc[2*B:3*B]  # [B, hidden_dim] - 不包含rel_AB
        c_states = node_states_bc[3*B:4*B]  # [B, hidden_dim]
        
        # 将rel_AB embedding作为额外的条件输入，与节点状态差异拼接
        # 这样rel_AB只作为条件，不会通过GNN传播到其他节点
        diff_bc = b_states_bc - c_states  # [B, hidden_dim]
        # 将rel_AB embedding投影到hidden_dim并加到差异特征上
        cond_ab_proj = self.input_proj_ab(cond_ab_emb)  # [B, hidden_dim]
        diff_bc_with_cond = diff_bc + cond_ab_proj  # [B, hidden_dim] - rel_AB作为条件加入
        
        logits_bc = self.head_bc(diff_bc_with_cond)  # [B, num_rel_classes]
        
        return logits_ab, logits_bc, rel_ab_pred
