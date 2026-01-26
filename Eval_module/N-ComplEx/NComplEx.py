"""
N-ComplEx: Main Entry for N-ary ComplEx Model
训练/评测统一入口，兼容python -m ...直接启动
"""
if __name__ == "__main__":
    from .train_paths import train_pipeline_from_graph
    print("Starting N-ComplEx training pipeline...")
    print("Model: N-ComplEx (Extension of ComplEx for N-ary KGs)")
    print()
    results = train_pipeline_from_graph(
        epochs=160,
        lr=0.002,
        embedding_dim=192,
        dropout=0.1,
        val_every=10,
        use_node_features=True,
        use_text_features=True,
        device="cuda"  # 可替换为 "cuda"
    )
