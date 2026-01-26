"""
StarE Model Entry Point
Run this module to train and evaluate StarE on N-ary Knowledge Graph.

Usage:
    python -m benchmark_paper.Eval_module.StarE.StarE
"""

if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph
    
    print("Starting StarE training pipeline...")
    print("Model: StarE (Structure-Aware Encoder for N-ary KG)")
    print("Reference: Galkin et al. EMNLP 2020")
    print()
    
    target_device = resolve_device()

    # Run training pipeline
    results = train_pipeline_from_graph(
        epochs=180,
        lr=0.002,
        embedding_dim=192,
        hidden_dim=448,
        num_layers=2,
        dropout=0.1,
        val_every=10,
        use_text_features=True,
        use_node_features=True,
        device=target_device
    )
    

