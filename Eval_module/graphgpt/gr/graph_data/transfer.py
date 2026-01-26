import torch
import pandas as pd

# 加载图数据
graph_data_path = 'gran_graph_data.pt'
graph_data = torch.load(graph_data_path)

# 假设你已经加载了 graph_data，并且有 arxiv 数据
arxiv_data = graph_data['arxiv']

# 1. 保存节点特征（node features）到 CSV
node_features = arxiv_data.x  # 节点特征
node_features_df = pd.DataFrame(node_features.numpy())  # 转换为 DataFrame
node_features_df.to_csv('arxiv_node_features.csv', index=False)

print("节点特征已保存为 arxiv_node_features.csv")

# 2. 保存边索引（edge_index）到 CSV
edge_index = arxiv_data.edge_index  # 边索引
edge_index_df = pd.DataFrame(edge_index.numpy().T, columns=['source', 'target'])  # 转换为 DataFrame
edge_index_df.to_csv('arxiv_edge_index.csv', index=False)

print("边索引已保存为 arxiv_edge_index.csv")

# 3. 保存节点标签（labels）到 CSV
labels = arxiv_data.y  # 节点标签
labels_df = pd.DataFrame(labels.numpy(), columns=['label'])  # 转换为 DataFrame
labels_df.to_csv('arxiv_node_labels.csv', index=False)

print("节点标签已保存为 arxiv_node_labels.csv")

# 4. 保存节点年份（node_year）到 CSV
node_year = arxiv_data.node_year  # 节点年份
node_year_df = pd.DataFrame(node_year.numpy(), columns=['year'])  # 转换为 DataFrame
node_year_df.to_csv('arxiv_node_year.csv', index=False)

print("节点年份已保存为 arxiv_node_year.csv")

# 5. 保存训练、验证、测试掩码（train_mask, val_mask, test_mask）到 CSV
train_mask = arxiv_data.train_mask  # 训练掩码
val_mask = arxiv_data.val_mask  # 验证掩码
test_mask = arxiv_data.test_mask  # 测试掩码

train_mask_df = pd.DataFrame(train_mask.numpy(), columns=['train_mask'])  # 转换为 DataFrame
val_mask_df = pd.DataFrame(val_mask.numpy(), columns=['val_mask'])  # 转换为 DataFrame
test_mask_df = pd.DataFrame(test_mask.numpy(), columns=['test_mask'])  # 转换为 DataFrame

train_mask_df.to_csv('arxiv_train_mask.csv', index=False)
val_mask_df.to_csv('arxiv_val_mask.csv', index=False)
test_mask_df.to_csv('arxiv_test_mask.csv', index=False)

print("训练、验证和测试掩码已保存为 CSV 文件！")

# 6. 保存邻接矩阵（adj_t）到 CSV
adj_t = arxiv_data.adj_t  # 邻接矩阵（稀疏矩阵）
adj_t_csr = adj_t.to_scipy().tocoo()  # 转为 COO 格式的稀疏矩阵
adj_t_df = pd.DataFrame({
    'row': adj_t_csr.row,
    'col': adj_t_csr.col,
    'data': adj_t_csr.data
})  # 转换为 DataFrame
adj_t_df.to_csv('arxiv_adjacency_matrix.csv', index=False)

print("邻接矩阵已保存为 arxiv_adjacency_matrix.csv")

# 7. 保存图的基本信息
num_nodes = arxiv_data.num_nodes  # 节点数量
graph_info = {'num_nodes': num_nodes}
graph_info_df = pd.DataFrame([graph_info])
graph_info_df.to_csv('arxiv_graph_info.csv', index=False)

print("图的基本信息已保存为 arxiv_graph_info.csv")
