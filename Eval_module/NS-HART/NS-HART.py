"""
NS-HART (Neural Symbolic Reasoning with Hierarchical Attention and Relation Trees) Training Entry Point

基于NS-HART的Transformer架构实现级联预测模型的训练入口
"""

if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph

    print("=" * 80)
    print("Starting NS-HART training pipeline...")
    print("Model: HARTCascadingPredictor (HART Transformer with cascading)")
    print("=" * 80)
    print()

    target_device = resolve_device()

    results = train_pipeline_from_graph(
        epochs=100,
        lr=0.001,
        embedding_dim=128,
        encoder_hidden_dim=256,
        encoder_layers=2,
        encoder_heads=4,
        dropout=0.1,
        val_every=10,
        use_text_features=True,
        use_node_features=True,
        device=target_device,
        weight_decay=1e-5
    )

