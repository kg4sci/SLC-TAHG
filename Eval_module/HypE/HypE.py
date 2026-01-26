"""
HypE Model Entry Point
Run this module to train and evaluate HypE on N-ary Knowledge Graph.

Usage:
    python -m HypE.HypE
"""

if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph

    print("Starting HypE training pipeline...")
    print("Model: HypE (Hypernetwork-based Embedding for N-ary KG)")
    print("Reference: Fatemi et al. AAAI 2020")
    print()
    
    target_device = resolve_device()

    # Run training pipeline
    results = train_pipeline_from_graph(
        epochs=100,
        lr=0.001,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        val_every=10,
        use_text_features=True,
        use_node_features=True,
        device=target_device
    )
    

