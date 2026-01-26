# HypE Model Package
# HypE: Hypernetwork-based Embedding for N-ary Knowledge Graphs
# Reference: Fatemi et al. "Knowledge Hypergraphs: Prediction Beyond Binary Relations" (AAAI 2020)
# This model uses hypernetwork and convolution for N-ary relations

from .models import HypEBackbone, CascadingHypEPredictor, HypETextProjector
from .train_paths import train_pipeline_from_graph

__all__ = [
    "HypEBackbone",
    "CascadingHypEPredictor",
    "HypETextProjector",
    "train_pipeline_from_graph",
]
