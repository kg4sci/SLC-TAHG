import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Optional


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss


class RAM(nn.Module):
    def __init__(self, K, n_r, n_v, rdim, vdim, n_parts, max_ary, device, **kwargs):
        # n_v=n_ent, vdim=edim, n_parts=m
        super(RAM, self).__init__()
        self.loss = MyLoss()
        self.device = device
        self.n_parts = n_parts
        self.n_ary = max_ary
        self.RolU = nn.Embedding(K, embedding_dim=rdim, padding_idx=0) # role basis-vector
        self.RelV = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(n_r, arity, K, requires_grad=True)).to(device))
             for arity in range(2, max_ary + 1)])
        self.Val = nn.Embedding(n_v, embedding_dim=vdim, padding_idx=0)  
        self.Plist = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(K, arity, self.n_parts, requires_grad=True)).to(device))
             for arity in range(2, max_ary + 1)])

        self.max_ary = max_ary
        self.drop_role, self.drop_value = torch.nn.Dropout(kwargs["drop_role"]), torch.nn.Dropout(kwargs["drop_ent"])
        self.device = device

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.RolU.weight.data)
        nn.init.xavier_normal_(self.Val.weight.data)
        for i in range(2, self.max_ary + 1):
            nn.init.xavier_normal_(self.Plist[i - 2])
            nn.init.xavier_normal_(self.RelV[i-2])

    def Sinkhorn(self, X):
        S = torch.exp(X)
        S = S / S.sum(dim=[1, 2], keepdim=True).repeat(1, S.shape[1], S.shape[2])
        return S

    def forward(self, rel_idx, value_idx, miss_value_domain):
        n_b, n_v, arity = value_idx.shape[0], self.Val.weight.shape[0], value_idx.shape[1]+1
        RelV = self.RelV[arity-2][rel_idx]
        RelV = F.softmax(RelV, dim=2)
        role = torch.matmul(RelV, self.RolU.weight)
        value = self.Val(value_idx)
        role, value = self.drop_role(role), self.drop_value(value)
        value = value.reshape(n_b, arity-1, self.n_parts, -1)
        Plist = self.Sinkhorn(self.Plist[arity-2])
        P = torch.einsum('bak,kde->bade', RelV, Plist)
        idx = [i for i in range(arity) if i + 1 != miss_value_domain]
        V0 = torch.einsum('bijk,baij->baik', value, P[:, :, idx, :])
        V1 = torch.prod(V0, dim=2)
        V0_miss = torch.einsum('njk,baj->bnak', self.Val.weight.reshape(n_v, self.n_parts, -1),
                               P[:, :, miss_value_domain - 1, :])
        score = torch.einsum('bak,bnak,bak->bn', V1, V0_miss, role)
        return score


class RAMCascadingFactClassifier(nn.Module):
    """
    适配本任务（路径级联预测）的 RAM 模型封装：
    - 第一步预测 rel_AB（基于 A, Event, B, C）
    - 将预测到的 rel_AB 作为条件嵌入，参与第二步 rel_BC 的预测
    - 与 HypE/StarE 的级联头对齐，接口与 RAM/train_paths.py 一致
    """
    def __init__(self,
                 num_entities: int,
                 num_relation_classes: int,
                 embedding_dim: int,
                 n_parts: int,
                 max_ary: int,
                 dropout: float,
                 node_feat_dim: int,
                 text_dim: int,
                 use_text_features: bool,
                 device: str):
        super().__init__()
        # 确保支持 arity=5（value_idx 含4个元素 -> arity=5）
        effective_max_ary = max(max_ary, 5)

        self.ram = RAM(
            K=embedding_dim,
            n_r=num_relation_classes,
            n_v=num_entities,
            rdim=embedding_dim,
            vdim=embedding_dim * n_parts,
            n_parts=n_parts,
            max_ary=effective_max_ary,
            device=device,
            drop_role=dropout,
            drop_ent=dropout
        )
        self.num_entities = num_entities
        self.num_relation_classes = num_relation_classes
        self.embedding_dim = embedding_dim
        self.node_feat_dim = node_feat_dim
        self.text_dim = text_dim
        self.use_text_features = use_text_features
        self.device = device

        # 可选：节点与文本特征投影到 embedding 空间（保持与 HypE/StarE 接口一致，便于后续扩展）
        self.node_feat_proj = nn.Linear(node_feat_dim, embedding_dim) if node_feat_dim and node_feat_dim > 0 else None
        self.text_proj = nn.Linear(text_dim, embedding_dim) if use_text_features and text_dim and text_dim > 0 else None

        # 关系类别嵌入（用于 rel_BC 的条件）
        self.relclass_embed = nn.Embedding(num_relation_classes, embedding_dim)

        # 分类头
        self.head_ab = nn.Linear(embedding_dim, num_relation_classes)
        self.head_bc = nn.Linear(embedding_dim, num_relation_classes)

    def _scores_to_embedding(self, scores: torch.Tensor) -> torch.Tensor:
        """
        将对所有实体的打分 [B, n_v] 映射为一个 embedding 表示 [B, emb]，
        通过对实体嵌入（按 part 平均）进行加权求和，避免巨型线性层导致的内存爆炸。
        """
        # self.ram.Val.weight: [n_v, vdim]，其中 vdim = embedding_dim * n_parts
        n_v = self.num_entities
        emb_dim = self.embedding_dim
        val_weight = self.ram.Val.weight.view(n_v, self.ram.n_parts, emb_dim).mean(dim=1)  # [n_v, emb]
        probs = F.softmax(scores, dim=-1)  # [B, n_v]
        return torch.matmul(probs, val_weight)  # [B, emb]

    def _add_optional_features(self, base_emb: torch.Tensor, node_features: torch.Tensor | None, text_features: torch.Tensor | None):
        emb = base_emb
        if self.node_feat_proj is not None and node_features is not None:
            emb = emb + self.node_feat_proj(node_features)
        if self.text_proj is not None and text_features is not None:
            emb = emb + self.text_proj(text_features)
        return emb

    def forward(self,
                a_ids: torch.Tensor,
                b_ids: torch.Tensor,
                c_ids: torch.Tensor,
                event_ids: torch.Tensor,
                training: bool,
                node_features_ab: torch.Tensor = None,
                node_features_bc: torch.Tensor = None,
                text_features_ab: torch.Tensor = None,
                text_features_bc: torch.Tensor = None):
        """
        Args:
            a_ids, b_ids, c_ids, event_ids: [B]
            training: 是否训练模式（训练时使用 Gumbel-Softmax 以可微方式采样 rel_AB）
            node_features: 可选节点特征（与 ID 对齐的 [B, node_feat_dim]）
            text_features_ab / text_features_bc: 可选文本证据（[B, text_dim]）
        Returns:
            logits_ab, logits_bc, rel_ab_pred
        """
        B = a_ids.size(0)
        device = a_ids.device
        # 防御：索引范围约束，避免 embedding 越界（Event 可能为缺失值）
        max_idx = self.num_entities - 1
        a_ids = torch.clamp(a_ids, min=0, max=max_idx)
        b_ids = torch.clamp(b_ids, min=0, max=max_idx)
        c_ids = torch.clamp(c_ids, min=0, max=max_idx)
        event_ids = torch.clamp(event_ids, min=0, max=max_idx)

        # ----- 第一步：预测 rel_AB -----
        # 构造 arity=4 的主路径输入 (A, Event, B, C)；miss_value_domain=5 表示预测关系
        rel_idx_ab = torch.zeros(B, dtype=torch.long, device=device)
        value_idx_ab = torch.stack([a_ids, event_ids, b_ids, c_ids], dim=1)
        miss_value_domain_ab = 5
        score_ab = self.ram(rel_idx_ab, value_idx_ab, miss_value_domain_ab)  # [B, n_v]
        score_ab_emb = self._scores_to_embedding(score_ab)  # [B, emb]
        # 可选融合节点/文本特征（与 HypE/StarE 保持风格一致）
        score_ab_emb = self._add_optional_features(score_ab_emb, node_features_ab, text_features_ab)
        logits_ab = self.head_ab(score_ab_emb)  # [B, num_relation_classes]

        # 采样/选择 rel_AB，获得条件嵌入
        if training:
            soft_onehot = F.gumbel_softmax(logits_ab, tau=1.0, hard=True, dim=-1)
            cond_ab_emb = torch.matmul(soft_onehot, self.relclass_embed.weight)  # [B, emb]
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
        else:
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
            cond_ab_emb = self.relclass_embed(rel_ab_pred)

        # ----- 第二步：预测 rel_BC（条件依赖于 rel_AB） -----
        rel_idx_bc = torch.zeros(B, dtype=torch.long, device=device)
        value_idx_bc = torch.stack([a_ids, event_ids, b_ids, c_ids], dim=1)
        miss_value_domain_bc = 5
        score_bc = self.ram(rel_idx_bc, value_idx_bc, miss_value_domain_bc)  # [B, n_v]
        score_bc_emb = self._scores_to_embedding(score_bc)  # [B, emb]
        score_bc_emb = self._add_optional_features(score_bc_emb, node_features_bc, text_features_bc)
        # 条件融合（采用加和，保持维度与稳定性）
        bc_input = score_bc_emb + cond_ab_emb
        logits_bc = self.head_bc(bc_input)  # [B, num_relation_classes]

        return logits_ab, logits_bc, rel_ab_pred