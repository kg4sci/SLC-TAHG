"""
HART-based cascade predictor aligning with GRAN cascading logic.

This module adapts the native NS-HART transformer-style aggregator so that
it can be used for the same cascading relation prediction task implemented
in the NaLP and GRAN baselines.  The design keeps the hyper-attention flavour
of HART (sequence + positional embeddings + CLS token + Transformer encoder),
while following GRAN's inference pipeline:

- Stage AB: encode the full path [A, Event, B, C] and classify rel_AB
- Stage BC: encode the same full path, inject rel_AB only as a conditioning
  signal at the classifier (no message passing), and predict rel_BC
- Optional node features are projected and added to entity embeddings
- Optional text features provide extra context by enriching the event node

The exposed interface mirrors `GRANCascadingPredictor` to ease swapping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HARTPathEncoder(nn.Module):
    """
    Lightweight transformer encoder inspired by NS-HART.

    It augments input node embeddings with learnable role/position embeddings,
    prepends a CLS token, and applies a stack of Transformer encoder layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_roles: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.role_embeddings = nn.Embedding(num_roles, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.role_embeddings.weight)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        role_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: [B, L, embed_dim]
            role_ids: [B, L]
            mask: optional [B, L] bool tensor (True for valid tokens)

        Returns:
            encoded: [B, L+1, embed_dim] (including CLS at position 0)
            attn_mask: [B, L+1] bool tensor (True for valid tokens)
        """
        B, L, _ = node_embeddings.shape
        device = node_embeddings.device

        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=device)

        role_emb = self.role_embeddings(role_ids)
        x = node_embeddings + role_emb
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        attn_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), mask], dim=1
        )

        x = self.dropout(x)
        encoded = self.encoder(x, src_key_padding_mask=~attn_mask)
        encoded = self.layer_norm(encoded)
        return encoded, attn_mask


class HARTCascadingPredictor(nn.Module):
    """
    Cascading relation classifier that leverages an NS-HART style encoder.

    The public API is kept consistent with `GRANCascadingPredictor` so that
    experiment scripts can switch architectures by configuration only.
    """

    ROLE_A = 0
    ROLE_EVENT = 1
    ROLE_B = 2
    ROLE_C = 3

    def __init__(
        self,
        num_entities: int,
        num_relation_classes: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        encoder_layers: int,
        encoder_heads: int,
        dropout: float,
        node_feat_dim: int = 0,
        text_dim: int = 0,
        use_text_features: bool = False,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relation_classes = num_relation_classes
        self.embedding_dim = embedding_dim
        self.use_text_features = use_text_features

        # Entity embeddings (shared for all roles)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # Relation-class embeddings (used for cascading conditioning)
        self.relation_class_embeddings = nn.Embedding(
            num_relation_classes, embedding_dim
        )

        # Optional node feature encoder
        if node_feat_dim > 0:
            self.node_feat_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.node_feat_encoder = None

        # Optional text feature projector (added to the event node)
        if use_text_features and text_dim > 0:
            self.text_projector = nn.Sequential(
                nn.Linear(text_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.text_projector = None

        # HART-style encoders for AB / BC stages
        self.encoder_ab = HARTPathEncoder(
            embed_dim=embedding_dim,
            num_heads=encoder_heads,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            num_roles=4,
        )
        self.encoder_bc = HARTPathEncoder(
            embed_dim=embedding_dim,
            num_heads=encoder_heads,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            num_roles=4,
        )

        # Post-encoder heads (operate on token differences)
        self.head_ab = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes),
        )
        self.head_bc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes),
        )

        # Conditioning projection for rel_AB embeddings
        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_class_embeddings.weight)

    def _get_entity_embedding_with_features(
        self,
        entity_ids: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.entity_embeddings(entity_ids)
        if node_features is not None and self.node_feat_encoder is not None:
            emb = emb + self.node_feat_encoder(node_features)
        return emb

    def _build_path_sequence(
        self,
        a_emb: torch.Tensor,
        event_emb: torch.Tensor,
        b_emb: torch.Tensor,
        c_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble node embeddings and role ids for the path [A, Event, B, C].
        """
        nodes = torch.stack([a_emb, event_emb, b_emb, c_emb], dim=1)
        B = a_emb.size(0)
        roles = torch.tensor(
            [self.ROLE_A, self.ROLE_EVENT, self.ROLE_B, self.ROLE_C],
            device=a_emb.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, 4, dtype=torch.bool, device=a_emb.device)
        return nodes, roles, mask

    def forward(
        self,
        a_ids: torch.Tensor,
        b_ids: torch.Tensor,
        c_ids: torch.Tensor,
        event_ids: torch.Tensor,
        training: bool,
        node_features: Optional[torch.Tensor] = None,
        text_features_ab: Optional[torch.Tensor] = None,
        text_features_bc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute cascading prediction with HART-style encoders.

        Returns:
            logits_ab: [B, num_relation_classes]
            logits_bc: [B, num_relation_classes]
            rel_ab_pred: [B]
        """
        device = a_ids.device

        # Gather optional node-aligned features
        feat_a = node_features[a_ids] if node_features is not None else None
        feat_b = node_features[b_ids] if node_features is not None else None
        feat_c = node_features[c_ids] if node_features is not None else None
        feat_event = node_features[event_ids] if node_features is not None else None

        # Entity embeddings (after feature fusion)
        a_emb = self._get_entity_embedding_with_features(a_ids, feat_a)
        b_emb = self._get_entity_embedding_with_features(b_ids, feat_b)
        c_emb = self._get_entity_embedding_with_features(c_ids, feat_c)
        event_emb = self._get_entity_embedding_with_features(event_ids, feat_event)

        # ===== Stage AB =====
        if (
            self.use_text_features
            and text_features_ab is not None
            and self.text_projector is not None
        ):
            event_emb_ab = event_emb + self.text_projector(text_features_ab)
        else:
            event_emb_ab = event_emb

        nodes_ab, roles_ab, mask_ab = self._build_path_sequence(
            a_emb, event_emb_ab, b_emb, c_emb
        )
        encoded_ab, _ = self.encoder_ab(nodes_ab, roles_ab, mask_ab)
        # Positions: CLS=0, A=1, Event=2, B=3, C=4
        a_state = encoded_ab[:, 1, :]
        b_state = encoded_ab[:, 3, :]
        diff_ab = a_state - b_state
        logits_ab = self.head_ab(diff_ab)

        if training:
            onehot_ab = F.gumbel_softmax(logits_ab, tau=1.0, hard=True, dim=-1)
            cond_ab_emb = torch.matmul(
                onehot_ab, self.relation_class_embeddings.weight
            )
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
        else:
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
            cond_ab_emb = self.relation_class_embeddings(rel_ab_pred)

        # ===== Stage BC =====
        if (
            self.use_text_features
            and text_features_bc is not None
            and self.text_projector is not None
        ):
            event_emb_bc = event_emb + self.text_projector(text_features_bc)
        else:
            event_emb_bc = event_emb

        nodes_bc, roles_bc, mask_bc = self._build_path_sequence(
            a_emb, event_emb_bc, b_emb, c_emb
        )
        encoded_bc, _ = self.encoder_bc(nodes_bc, roles_bc, mask_bc)
        b_state_bc = encoded_bc[:, 3, :]
        c_state = encoded_bc[:, 4, :]

        diff_bc = b_state_bc - c_state
        cond_proj = self.cond_proj(cond_ab_emb)
        diff_bc_with_cond = diff_bc + cond_proj
        logits_bc = self.head_bc(diff_bc_with_cond)

        return logits_ab, logits_bc, rel_ab_pred


__all__ = ["HARTCascadingPredictor"]

