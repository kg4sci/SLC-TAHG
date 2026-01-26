from ..config import SBERT_DIM
from ..device_utils import resolve_device
from .train_paths import train_pipeline_from_graph

# 入口：支持参数全部调参
if __name__ == '__main__':
    device = resolve_device()
    train_pipeline_from_graph(
        sbert_dim=SBERT_DIM,
        epochs=100,              # 训练轮数
        lr=0.0001028,               # 学习率
        batch_size=112,           # 批次大小
        hidden_dim=128,          # RGCN隐藏层/MLP中间层宽度
        dropout_rate=0.1,        # Dropout 比例
        weight_decay=0.00025,       # Adam正则
        use_slc_neighbors=True,  # SLC邻居特征
        device=device,
    )
# 参数含义与用法：
# epochs:         训练轮数
# lr:             学习率
# batch_size:     批次大小（当前单卡1次epoch遍历全部）
# hidden_dim:     RGCN第1层宽度，对性能和显存有影响
# dropout_rate:   Dropout强度，防止过拟合
# weight_decay:   权重正则化项，防止过拟合
# use_slc_neighbors: 是否用SLC邻域信息
