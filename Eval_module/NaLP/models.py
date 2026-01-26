"""
NaLP: Neural-Logical Programming for N-ary Knowledge Graphs
Reference: Guan et al. "Logical Message Passing Networks with One-hop Inference on Atomic Formulas" (ICLR 2023)

Key Concepts:
- Neural-symbolic reasoning
- Logical rule learning for N-ary relations
- One-hop inference on atomic formulas

Note: This is a placeholder implementation. Full implementation requires:
1. Logical rule templates
2. Rule weight learning
3. Neural-logical inference mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaLPFactBlock(nn.Module):
    """
    原生NaLP思想的事实编码块：
    - 输入：若干 role-value 对，先拼接 [role_emb || value_emb] 形成 picture（序列）
    - 1x1 卷积（跨 width）得到每个 role-value 的相关性表示 o_i
    - g-FCN 对 (o_i, o_j) 两两组合生成 relatedness，再按 i 做 reduce_min 得到 relatedness_res
    - 输出：relatedness_res（作为分类器输入）
    """
    def __init__(
        self,
        n_roles: int,
        n_values_entities: int,
        n_relation_classes: int,
        embedding_dim: int,
        n_filters: int,
        n_gfcn: int,
        dropout: float
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.n_gfcn = n_gfcn

        # 角色与值嵌入；值分两类：实体值、关系类别值（用于CondAB）
        self.role_embeddings = nn.Embedding(n_roles, embedding_dim)
        self.value_entity_embeddings = nn.Embedding(n_values_entities, embedding_dim)
        self.value_relclass_embeddings = nn.Embedding(n_relation_classes, embedding_dim)

        # 1x1 conv over the concatenated [role||value] width
        # 输入张量形状：[B, 1, arity, 2*embedding_dim]，使用 kernel=(1, 2*emb) 使每个位置输出 n_filters
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(1, 2 * embedding_dim),
            stride=(1, 1),
            padding=0
        )
        self.bn = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # g-FCN: concat(o_i, o_j) -> n_gfcn
        self.gfcn = nn.Sequential(
            nn.Linear(2 * n_filters, n_gfcn),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.role_embeddings.weight)
        nn.init.xavier_uniform_(self.value_entity_embeddings.weight)
        nn.init.xavier_uniform_(self.value_relclass_embeddings.weight)
        nn.init.xavier_uniform_(self.conv.weight)

    def build_picture(self, role_ids: torch.Tensor, value_embeds: torch.Tensor) -> torch.Tensor:
        """
        role_ids: [B, arity]  int64
        value_embeds: [B, arity, embedding_dim]  已经根据实体或关系类别查表得到
        returns picture tensor: [B, arity, 2*embedding_dim]
        """
        role_emb = self.role_embeddings(role_ids)  # [B, arity, emb]
        picture = torch.cat([role_emb, value_embeds], dim=-1)  # [B, arity, 2*emb]
        return picture

    def forward(self, role_ids: torch.Tensor, value_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute relatedness_res for a fact (picture defined by role_ids + value_embeds)
        role_ids: [B, arity]
        value_embeds: [B, arity, emb]
        returns: relatedness_res [B, n_gfcn]
        """
        B, arity, _ = value_embeds.shape
        picture = self.build_picture(role_ids, value_embeds)  # [B, arity, 2*emb]

        # Conv over width
        x = picture.unsqueeze(1)  # [B, 1, arity, 2*emb]
        x = self.conv(x)          # [B, n_filters, arity, 1]
        x = self.bn(x)
        x = self.relu(x)
        x = x.squeeze(-1).transpose(1, 2)  # [B, arity, n_filters]
        x = self.dropout(x)

        # Pairwise g-FCN for each i against all j
        # 计算对称对 (i, j)，得到 [B, arity, arity, n_gfcn]，再对 j 聚合 min，得到 [B, arity, n_gfcn]
        o_i = x.unsqueeze(2).expand(B, arity, arity, self.n_filters)
        o_j = x.unsqueeze(1).expand(B, arity, arity, self.n_filters)
        pair = torch.cat([o_i, o_j], dim=-1)  # [B, arity, arity, 2*n_filters]
        pair = pair.reshape(B * arity * arity, 2 * self.n_filters)
        g = self.gfcn(pair)  # [B*arity*arity, n_gfcn]
        g = g.view(B, arity, arity, self.n_gfcn)
        # 对 j 取最小（原文 relatedness_res = reduce_min over j）
        relatedness_list = torch.min(g, dim=2).values  # [B, arity, n_gfcn]
        # 再对 i 取最小，得到最终 [B, n_gfcn]
        relatedness_res = torch.min(relatedness_list, dim=1).values  # [B, n_gfcn]
        return relatedness_res


class NaLPBackbone(nn.Module):
    ROLE_A = 0
    ROLE_EVENT = 1
    ROLE_B = 2
    ROLE_C = 3
    ROLE_COND_AB = 4
    ROLE_TEXT_AB = 5
    ROLE_TEXT_BC = 6

    def __init__(
        self,
        num_entities: int,
        num_relation_classes: int,
        embedding_dim: int,
        n_filters: int,
        n_gfcn: int,
        dropout: float,
        node_feat_dim: int = 0,
        text_dim: int = 0,
        use_text_features: bool = False,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.use_text_features = use_text_features

        if node_feat_dim > 0:
            self.node_feat_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.node_feat_encoder = None

        if use_text_features and text_dim > 0:
            self.text_projector = nn.Sequential(
                nn.Linear(text_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 2, embedding_dim),
            )
            n_roles = 7
        else:
            self.text_projector = None
            n_roles = 5

        self.fact_block = NaLPFactBlock(
            n_roles=n_roles,
            n_values_entities=num_entities,
            n_relation_classes=num_relation_classes,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            n_gfcn=n_gfcn,
            dropout=dropout,
        )

        self.entity_lookup = self.fact_block.value_entity_embeddings
        self.relclass_lookup = self.fact_block.value_relclass_embeddings

    def project_text(self, text_features: torch.Tensor | None) -> torch.Tensor | None:
        if text_features is None or self.text_projector is None:
            return None
        return self.text_projector(text_features)

    def relation_value_from_ids(self, rel_ids: torch.Tensor) -> torch.Tensor:
        return self.relclass_lookup(rel_ids)

    def relation_value_from_onehot(self, rel_onehot: torch.Tensor) -> torch.Tensor:
        weights = self.relclass_lookup.weight
        return rel_onehot @ weights

    def _entity_with_features(
        self,
        entity_ids: torch.Tensor,
        node_features: torch.Tensor | None,
    ) -> torch.Tensor:
        emb = self.entity_lookup(entity_ids)
        if node_features is not None and self.node_feat_encoder is not None:
            feat = node_features[entity_ids]
            emb = emb + self.node_feat_encoder(feat)
        return emb

    def encode_stage(
        self,
        stage: str,
        a_ids: torch.Tensor,
        b_ids: torch.Tensor,
        c_ids: torch.Tensor,
        event_ids: torch.Tensor,
        node_features: torch.Tensor | None = None,
        cond_rel_emb: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.entity_lookup.weight.device
        a_ids = a_ids.to(device)
        b_ids = b_ids.to(device)
        c_ids = c_ids.to(device)
        event_ids = event_ids.to(device)

        node_feat_a = node_features[a_ids] if node_features is not None else None
        node_feat_b = node_features[b_ids] if node_features is not None else None
        node_feat_c = node_features[c_ids] if node_features is not None else None
        node_feat_event = node_features[event_ids] if node_features is not None else None

        a_emb = self._entity_with_features(a_ids, node_features)
        event_emb = self._entity_with_features(event_ids, node_features)
        b_emb = self._entity_with_features(b_ids, node_features)
        c_emb = self._entity_with_features(c_ids, node_features)

        roles = [self.ROLE_A, self.ROLE_EVENT, self.ROLE_B, self.ROLE_C]
        values = [a_emb, event_emb, b_emb, c_emb]

        if stage == "bc":
            if cond_rel_emb is None:
                raise ValueError("cond_rel_emb is required for BC stage.")
            roles.append(self.ROLE_COND_AB)
            values.append(cond_rel_emb.to(device))

        if self.use_text_features and text_features is not None and self.text_projector is None:
            raise ValueError("Text features provided but text projector is not initialized.")

        if self.use_text_features and text_features is not None:
            roles.append(self.ROLE_TEXT_AB if stage == "ab" else self.ROLE_TEXT_BC)
            values.append(text_features.to(device))

        role_ids = torch.tensor(roles, device=device).unsqueeze(0).expand(a_ids.size(0), -1)
        val_tensor = torch.stack(values, dim=1)
        return self.fact_block(role_ids, val_tensor)


class NaLPCascadingHead(nn.Module):
    def __init__(self, repr_dim: int, num_relation_classes: int, dropout: float):
        super().__init__()
        self.repr_dim = repr_dim
        self.num_relation_classes = num_relation_classes

        self.head_ab = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, num_relation_classes),
        )

        self.rel_embeddings = nn.Embedding(num_relation_classes, repr_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_bc = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, num_relation_classes),
        )

    def forward_ab(self, repr_ab: torch.Tensor, training: bool):
        logits_ab = self.head_ab(repr_ab)
        if training:
            soft = F.gumbel_softmax(logits_ab, tau=1.0, hard=True, dim=-1)
            cond_vec = soft @ self.rel_embeddings.weight
            rel_ids = torch.argmax(logits_ab, dim=-1)
            cond_onehot = soft
        else:
            rel_ids = torch.argmax(logits_ab, dim=-1)
            cond_vec = self.rel_embeddings(rel_ids)
            cond_onehot = F.one_hot(rel_ids, num_classes=self.num_relation_classes).float()
        cond_signal = self.cond_proj(cond_vec)
        return logits_ab, cond_signal, rel_ids, cond_onehot

    def forward_bc(self, repr_bc: torch.Tensor, cond_signal: torch.Tensor):
        return self.head_bc(repr_bc + cond_signal)
