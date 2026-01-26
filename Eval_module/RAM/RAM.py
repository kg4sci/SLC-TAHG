if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph
    print("Starting RAM training pipeline...")
    print("Model: RAM (Relational Attention Mechanism, Adapted for N-ary Path Cascading)")
    print()
    target_device = resolve_device()

    results = train_pipeline_from_graph(
        epochs=150,
        lr=0.001,
        embedding_dim=128,
        n_parts=4,
        max_ary=5,  # 至少需要5，因为输入是4元组(A, Event, B, C)，arity=5
        dropout=0.1,
        val_every=10,
        use_node_features=True,
        use_text_features=True,
        device=target_device  # 自动优先使用GPU
    )
