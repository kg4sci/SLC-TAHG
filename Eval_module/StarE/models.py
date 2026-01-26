"""
StarE original backbone + cascading heads.

This file now exposes a clean Backbone + Task Head split:
1. `StarE` wraps the official `StarEEncoder` (message passing + qualifier logic)
   and outputs entity embeddings for the entire graph.
2. `CascadingStarEPredictor` implements the task-specific AB→BC cascading head.
3. `StarEWithTextProjector` keeps the optional text fusion pathway while reusing
   the original StarE backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import StarEEncoder


class StarE(nn.Module):
    """
    Wrapper around the original StarEEncoder that exposes a simple forward()
    returning the full set of entity embeddings after message passing.

    Args:
        graph_repr: Dict with edge_index / edge_type tensors as expected by StarEEncoder
        config: Full StarE config (see train_paths.build_stare_config)
    """

    def __init__(self, graph_repr, config, node_features=None, node_feat_dim: int = None):
        super().__init__()
        self.encoder = StarEEncoder(
            graph_repr,
            config,
            node_features=node_features,
            node_feat_dim=node_feat_dim,
        )
        self.hidden_drop = nn.Dropout(config['STAREARGS']['HID_DROP'])
        self.feature_drop = nn.Dropout(config['STAREARGS']['FEAT_DROP'])
        device = config['DEVICE']
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.num_entities = config['NUM_ENTITIES']

        self.register_buffer(
            'all_entity_ids',
            torch.arange(self.num_entities, dtype=torch.long, device=self.device),
            persistent=False)
        self.register_buffer(
            'dummy_rel_ids',
            torch.zeros(self.num_entities, dtype=torch.long, device=self.device),
            persistent=False)

    def forward(self):
        """
        Returns:
            Tensor of shape [num_entities, embedding_dim] containing the updated
            entity embeddings after StarE message passing.
        """
        _, _, entity_emb = self.encoder.forward_base(
            sub=self.all_entity_ids,
            rel=self.dummy_rel_ids,
            drop1=self.hidden_drop,
            drop2=self.feature_drop,
            quals=None,
            embed_qualifiers=False)
        return entity_emb


class CascadingStarEPredictor(nn.Module):
    """
    Cascading Prediction head for StarE model.
    
    This predictor implements CASCADING prediction:
    1. First predict rel_AB using (h_a, h_b, h_event)
    2. Then predict rel_BC using (h_b, h_c, h_event, h_rel_ab)
       where h_rel_ab is the embedding of PREDICTED rel_AB (not ground truth!)
    
    This is crucial for realistic evaluation since in real scenarios
    we don't know the ground truth rel_AB when predicting rel_BC.
    """
    
    def __init__(self, embedding_dim: int, num_relation_classes: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relation_classes = num_relation_classes
        
        # Relation embeddings (for cascading feature)
        self.rel_embeddings = nn.Embedding(num_relation_classes, embedding_dim)
        
        # MLP for rel_AB prediction (A -> B via RelaEvent)
        self.predictor_ab = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),  # [h_a, h_b, h_event]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes)
        )
        
        # MLP for rel_BC prediction (B -> C via RelaEvent, CONDITIONED on predicted rel_AB)
        self.predictor_bc = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),  # [h_b, h_c, h_event, h_rel_ab]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_relation_classes)
        )
        
    def forward(self, h_a, h_b, h_c, h_event, training: bool):
        """
        Args:
            h_a: SLCGene embeddings [batch_size, embedding_dim]
            h_b: Pathway embeddings [batch_size, embedding_dim]
            h_c: Disease embeddings [batch_size, embedding_dim]
            h_event: RelaEvent embeddings [batch_size, embedding_dim]
            training: bool，区分前向时是否用Gumbel-Softmax（可微）
        Returns:
            logits_ab: Logits for rel_AB prediction
            logits_bc: Logits for rel_BC prediction
            rel_ab_pred: rel_AB类别（分析/评估用）
        """
        x_ab = torch.cat([h_a, h_b, h_event], dim=-1)
        logits_ab = self.predictor_ab(x_ab)
        if training:
            soft_pred_ab = F.gumbel_softmax(logits_ab, tau=1.0, hard=True)
            h_rel_ab = soft_pred_ab @ self.rel_embeddings.weight
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
        else:
            rel_ab_pred = torch.argmax(logits_ab, dim=-1)
            h_rel_ab = self.rel_embeddings(rel_ab_pred)
        x_bc = torch.cat([h_b, h_c, h_event, h_rel_ab], dim=-1)
        logits_bc = self.predictor_bc(x_bc)
        return logits_ab, logits_bc, rel_ab_pred


class StarEWithTextProjector(nn.Module):
    """
    Extended StarE model with text evidence projection.
    
    This allows incorporating textual evidence (from MongoDB) into the model.
    Combines structural embeddings with text features from entity pairs.
    """
    
    def __init__(
        self,
        graph_repr,
        config,
        num_relation_classes: int,
        text_dim: int,
        dropout: float,
        node_features=None,
        node_feat_dim: int = None,
    ):
        super().__init__()
        self.backbone = StarE(
            graph_repr,
            config,
            node_features=node_features,
            node_feat_dim=node_feat_dim,
        )
        embedding_dim = config['EMBEDDING_DIM']

        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        self.predictor = CascadingStarEPredictor(
            embedding_dim=embedding_dim,
            num_relation_classes=num_relation_classes,
            dropout=dropout
        )

    def forward(self, text_features_ab=None, text_features_bc=None):
        """
        Returns:
            Tuple: (node_embeddings, (proj_ab, proj_bc))
        """
        node_embeddings = self.backbone()
        proj_ab = self.text_projector(text_features_ab) if text_features_ab is not None else None
        proj_bc = self.text_projector(text_features_bc) if text_features_bc is not None else None
        return node_embeddings, (proj_ab, proj_bc)

