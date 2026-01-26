import torch
import torch.nn as nn
import torch.nn.functional as F


class HypEBackbone(nn.Module):
    """
    HypE backbone that keeps the original hypernetwork + 1D convolution spirit.

    - Each path is represented by concatenating role-specific entity embeddings
      (A, Event, B, C).
    - A stage-specific relation embedding (AB / BC) is passed through a
      hypernetwork that generates convolution kernels (1D) and biases.
    - The convolution operates along the concatenated signal, producing
      relation-aware features that are projected back to the embedding space.
    - Optional node features are linearly projected and added to entity
      embeddings before convolution.
    - Optional text features (already projected to the embedding_dim) can be
      added to stage-specific roles (A/B for AB stage, B/C for BC stage).
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        kernel_size: int,
        num_filters: int,
        dropout: float,
        node_feat_dim: int = 0,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.seq_len = 4 * embedding_dim
        self.conv_out_len = self.seq_len - kernel_size + 1
        if self.conv_out_len <= 0:
            raise ValueError("kernel_size must be smaller than the flattened sequence length.")

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        if node_feat_dim and node_feat_dim > 0:
            self.node_feat_encoder = nn.Sequential(
                nn.Linear(node_feat_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.node_feat_encoder = None

        self.stage_embeddings = nn.Embedding(2, embedding_dim)  # 0: AB, 1: BC
        self.hyper_w = nn.Linear(embedding_dim, num_filters * kernel_size, bias=False)
        self.hyper_b = nn.Linear(embedding_dim, num_filters)

        self.output_proj = nn.Linear(num_filters * self.conv_out_len, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.stage_embeddings.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def _lookup(self, entity_ids: torch.Tensor, node_features: torch.Tensor | None) -> torch.Tensor:
        emb = self.entity_embeddings(entity_ids)
        if node_features is not None and self.node_feat_encoder is not None:
            feat = node_features[entity_ids]
            emb = emb + self.node_feat_encoder(feat)
        return emb

    def _stage_kernel(self, stage_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stage_vec = self.stage_embeddings(torch.tensor(stage_idx, device=self.stage_embeddings.weight.device))
        weight = self.hyper_w(stage_vec).view(self.num_filters, 1, self.kernel_size)
        bias = self.hyper_b(stage_vec)
        return weight, bias

    def encode_stage(
        self,
        stage: str,
        a_ids: torch.Tensor,
        b_ids: torch.Tensor,
        c_ids: torch.Tensor,
        event_ids: torch.Tensor,
        node_features: torch.Tensor | None = None,
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute relation-aware representations for AB or BC stage.

        Args:
            stage: 'ab' or 'bc'
            *_ids: tensors of shape [B]
            node_features: Optional global node feature matrix to index into
            text_features: Optional tensor [B, embedding_dim] already projected
                           into the model space. Added to (A,B) for AB stage,
                           or (B,C) for BC stage.
        Returns:
            Tensor of shape [B, embedding_dim]
        """
        device = self.entity_embeddings.weight.device
        a_ids = a_ids.to(device)
        b_ids = b_ids.to(device)
        c_ids = c_ids.to(device)
        event_ids = event_ids.to(device)

        a_emb = self._lookup(a_ids, node_features)
        b_emb = self._lookup(b_ids, node_features)
        c_emb = self._lookup(c_ids, node_features)
        event_emb = self._lookup(event_ids, node_features)

        if text_features is not None:
            text_features = text_features.to(device)
            if stage == "ab":
                a_emb = a_emb + text_features
                b_emb = b_emb + text_features
            else:
                b_emb = b_emb + text_features
                c_emb = c_emb + text_features

        seq = torch.cat([a_emb, event_emb, b_emb, c_emb], dim=-1).unsqueeze(1)  # [B, 1, seq_len]
        stage_idx = 0 if stage == "ab" else 1
        weight, bias = self._stage_kernel(stage_idx)
        conv = F.conv1d(seq, weight, bias=bias)
        conv = F.relu(conv)
        conv = conv.view(conv.size(0), -1)
        repr_vec = self.output_proj(conv)
        repr_vec = self.layer_norm(repr_vec)
        repr_vec = self.dropout(repr_vec)
        return repr_vec


class CascadingHypEPredictor(nn.Module):
    """
    Task head that consumes backbone outputs for AB and BC stages and performs
    cascading classification (rel_AB first, rel_BC conditioned on predicted rel_AB).
    """

    def __init__(self, embedding_dim: int, num_relation_classes: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relation_classes = num_relation_classes

        self.rel_embeddings = nn.Embedding(num_relation_classes, embedding_dim)

        self.head_ab = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_bc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes),
        )

    def forward(self, repr_ab: torch.Tensor, repr_bc: torch.Tensor, training: bool):
        logits_ab = self.head_ab(repr_ab)

        if training:
            soft_pred = F.gumbel_softmax(logits_ab, tau=1.0, hard=True, dim=-1)
            cond_emb = soft_pred @ self.rel_embeddings.weight
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
        else:
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
            cond_emb = self.rel_embeddings(rel_ab_pred)

        cond_signal = self.cond_proj(cond_emb)
        logits_bc = self.head_bc(repr_bc + cond_signal)
        return logits_ab, logits_bc, rel_ab_pred


class HypETextProjector(nn.Module):
    """
    Projects SBERT (or other) text embeddings into the HypE embedding space.
    """

    def __init__(self, text_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, text_features: torch.Tensor | None) -> torch.Tensor | None:
        if text_features is None:
            return None
        return self.projector(text_features)
