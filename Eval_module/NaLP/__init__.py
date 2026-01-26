# NaLP Model Package
# NaLP: Neural-Logical Programming for N-ary Knowledge Graphs
# Reference: Guan et al. "Logical Message Passing Networks with One-hop Inference on Atomic Formulas" (ICLR 2023)
# This model combines neural and logical reasoning for N-ary relations

from .models import NaLPBackbone, NaLPCascadingHead, NaLPFactBlock
from .train_paths import train_pipeline_from_graph

__all__ = [
    "NaLPFactBlock",
    "NaLPBackbone",
    "NaLPCascadingHead",
    "train_pipeline_from_graph",
]
