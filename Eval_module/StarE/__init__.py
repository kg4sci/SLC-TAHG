# StarE Model Package
# StarE: Structure-Aware Encoder for N-ary Knowledge Graph
# Reference: Galkin et al. "Message Passing for Hyper-Relational Knowledge Graphs" (EMNLP 2020)
# This model uses star graph decomposition for N-ary relations

from .models import StarE, CascadingStarEPredictor, StarEWithTextProjector
from .train_paths import train_pipeline_from_graph

__all__ = [
    'StarE',
    'CascadingStarEPredictor',
    'StarEWithTextProjector',
    'train_pipeline_from_graph',
]
