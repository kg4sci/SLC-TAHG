"""
GRAN (Graph Recurrent Attention Networks) Training Entry Point

基于GRAN的GNN架构实现级联预测模型的训练入口
"""

if __name__ == "__main__":
    from ..device_utils import resolve_device
    from .train_paths import train_pipeline_from_graph

    print("=" * 80)
    print("Starting GRAN training pipeline...")
    print("Model: GRANCascadingPredictor (GRAN GNN with cascading)")
    print("=" * 80)
    print()

    target_device = resolve_device()

    results = train_pipeline_from_graph(
        epochs=100,
        lr=0.01,#0.001:0.4110
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_prop=1,
        dropout=0.1,
        val_every=10,
        use_text_features=True,
        use_node_features=True,
        has_attention=True,
        device=target_device
    )

