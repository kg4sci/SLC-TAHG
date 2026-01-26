import torch
import os
import torch.nn as nn
os.environ["DGL_USE_GRAPHBOLT"] = "0"
from dgl.nn.pytorch import RelGraphConv

from ..config import SBERT_DIM


class RGCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_rels: int):
        super(RGCN, self).__init__()
        self.rgcn_layer1 = RelGraphConv(in_dim, hid_dim, num_rels, num_bases=-1)
        self.rgcn_layer2 = RelGraphConv(hid_dim, out_dim, num_rels, num_bases=-1)

    def forward(self, g, features, rel_types):
        h = self.rgcn_layer1(g, features, rel_types)
        h = torch.relu(h)
        h = self.rgcn_layer2(g, h, rel_types)
        return h


class RelationEmbeddings(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_relations, dim)

    def forward(self, rel_ids):
        return self.emb(rel_ids)


class PredictorAB(nn.Module):
    def __init__(self, dim: int, num_relations: int, use_slc_neighbors: bool):
        super().__init__()
        # If use_slc_neighbors=True, input is [h_a(dim), h_a_neighbors_agg(SBERT_DIM), h_b(dim)]
        # Otherwise, input is [h_a(dim), h_b(dim)]
        input_dim = (dim * 2 + SBERT_DIM) if use_slc_neighbors else (dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim), nn.ReLU(), nn.Linear(dim, num_relations)
        )
        self.use_slc_neighbors = use_slc_neighbors

    def forward(self, h_a, h_b, h_a_neighbors_agg=None):
        if self.use_slc_neighbors and h_a_neighbors_agg is not None:
            x = torch.cat([h_a, h_a_neighbors_agg, h_b], dim=-1)
        else:
            x = torch.cat([h_a, h_b], dim=-1)
        return self.mlp(x)


class PredictorBC(nn.Module):
    def __init__(self, dim: int, num_relations: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, num_relations)
        )

    def forward(self, h_b, h_c, e_rel_ab):
        x = torch.cat([h_b, h_c, e_rel_ab], dim=-1)
        return self.mlp(x)


class TextProjector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.proj(x)
