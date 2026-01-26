import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CascadingModel(nn.Module):
    """
    级联预测模型，支持阶段特定的节点特征：
    - base 特征：节点 id embedding + 可选的通用节点特征
    - ab_stage_feat：仅在 AB 阶段叠加（事件节点的 AB 证据编码）
    - bc_stage_feat：仅在 BC 阶段叠加
    """

    def __init__(
        self,
        num_nodes: int,
        num_rel_classes: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        node_feat_base: Optional[torch.Tensor] = None,
        node_feat_ab: Optional[torch.Tensor] = None,
        node_feat_bc: Optional[torch.Tensor] = None,
        use_node_feat: bool = True,
    ):
        super().__init__()
        self.use_node_feat = use_node_feat and node_feat_base is not None

        self.node_embeddings = nn.Embedding(num_nodes, emb_dim)

        if self.use_node_feat:
            feat_dim = node_feat_base.size(1)
            self.node_feat_base = nn.Parameter(node_feat_base, requires_grad=False)
            self.node_feat_encoder = nn.Sequential(
                nn.Linear(feat_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.node_feat_base = None
            self.node_feat_encoder = None

        # 阶段特定特征（与 base_feat 处于同一特征空间，例如同一 LM 的不同视角），
        # 在 lookup 时也复用 node_feat_encoder 投影到 emb_dim 再相加，避免与 id embedding 维度不一致。
        self.node_feat_ab = nn.Parameter(node_feat_ab, requires_grad=False) if node_feat_ab is not None else None
        self.node_feat_bc = nn.Parameter(node_feat_bc, requires_grad=False) if node_feat_bc is not None else None

        self.stage_mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        self.rel_embeddings = nn.Embedding(num_rel_classes, emb_dim)

        self.head_ab = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rel_classes),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.head_bc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rel_classes),
        )

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def _lookup(self, ids: torch.Tensor, stage: str) -> torch.Tensor:
        emb = self.node_embeddings(ids)
        if self.use_node_feat and self.node_feat_base is not None:
            feat = self.node_feat_base[ids]
            emb = emb + self.node_feat_encoder(feat)
        # 阶段特定特征也通过同一 encoder 投影到 emb_dim 后再相加
        if stage == "ab" and self.node_feat_ab is not None and self.node_feat_encoder is not None:
            feat_ab = self.node_feat_ab[ids]
            emb = emb + self.node_feat_encoder(feat_ab)
        if stage == "bc" and self.node_feat_bc is not None and self.node_feat_encoder is not None:
            feat_bc = self.node_feat_bc[ids]
            emb = emb + self.node_feat_encoder(feat_bc)
        return emb

    def encode_stage(self, stage: str, A, B, C, Event) -> torch.Tensor:
        a_emb = self._lookup(A, stage)
        b_emb = self._lookup(B, stage)
        c_emb = self._lookup(C, stage)
        e_emb = self._lookup(Event, stage)
        seq = torch.cat([a_emb, e_emb, b_emb, c_emb], dim=-1)
        return self.stage_mlp(seq)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        Event: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        repr_ab = self.encode_stage("ab", A, B, C, Event)
        repr_bc = self.encode_stage("bc", A, B, C, Event)

        logits_ab = self.head_ab(self.dropout(repr_ab))

        probs_ab = F.softmax(logits_ab, dim=-1)
        cond_emb = probs_ab @ self.rel_embeddings.weight
        cond_signal = self.cond_proj(cond_emb)

        logits_bc = self.head_bc(self.dropout(repr_bc + cond_signal))
        pred_ab = torch.argmax(logits_ab, dim=-1)
        return logits_ab, logits_bc, pred_ab

