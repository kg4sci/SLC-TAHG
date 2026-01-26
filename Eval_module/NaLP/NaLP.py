if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph

    print("Starting NaLP training pipeline...")
    print("Model: NaLPBackbone + Cascading Head (fact-style NaLP with cascading)")
    print()

    target_device = resolve_device()

    results = train_pipeline_from_graph(
        epochs=100,
        lr=0.01,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        val_every=10,
        use_text_features=True,
        use_node_features=True,
        device=target_device
    )